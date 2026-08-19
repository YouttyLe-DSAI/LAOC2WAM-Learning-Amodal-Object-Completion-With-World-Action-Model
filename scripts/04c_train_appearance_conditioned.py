"""
Fine-tune Stable Diffusion UNet nhận ĐÚNG 2 input theo slide:
  - Object's mask (amodal mask, nối thêm kênh vào input UNet)
  - Object's appearance feature vector (CLIP image embedding, tiêm qua
    cross-attention THAY cho text embedding)
Output: amodal RGB.

Training kiểu SELF-SUPERVISED (giống PCNet/SeGAN/SynergyAmodal): lấy ảnh
vật thể KHÔNG bị che từ COCOA làm "đáp án đúng", TỰ che 1 phần đi, trích
feature vector từ phần còn thấy, dạy model tái tạo lại đúng ảnh gốc.

Usage:
    python scripts/04c_train_appearance_conditioned.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor

IMG_EXTS = (".jpg", ".jpeg", ".png")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def estimate_object_mask(image_bgr: np.ndarray, border_ratio: float = 0.06) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    bx, by = int(w * border_ratio), int(h * border_ratio)
    rect = (bx, by, w - 2 * bx, h - 2 * by)
    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        fb = np.zeros((h, w), np.uint8)
        fb[by:h - by, bx:w - bx] = 1
        return fb
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    if fg.sum() < 0.02 * h * w:
        fb = np.zeros((h, w), np.uint8)
        fb[by:h - by, bx:w - bx] = 1
        return fb
    return fg


def add_synthetic_occluder(image_bgr: np.ndarray, amodal_mask: np.ndarray, rng: random.Random):
    h, w = amodal_mask.shape
    ys, xs = np.where(amodal_mask > 0)
    if len(xs) == 0:
        return image_bgr.copy(), np.zeros((h, w), np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    obj_w, obj_h = x1 - x0, y1 - y0
    occ_w = max(int(obj_w * rng.uniform(0.25, 0.45)), 10)
    occ_h = max(int(obj_h * rng.uniform(0.25, 0.45)), 10)
    cx, cy = rng.randint(x0, x1), rng.randint(y0, y1)
    occluder_mask = np.zeros((h, w), np.uint8)
    color = tuple(rng.randint(0, 255) for _ in range(3))
    out = image_bgr.copy()
    axes_e = (occ_w // 2, occ_h // 2)
    cv2.ellipse(out, (cx, cy), axes_e, 0, 0, 360, color, -1)
    cv2.ellipse(occluder_mask, (cx, cy), axes_e, 0, 0, 360, 1, -1)
    return out, occluder_mask


class SelfSupervisedOcclusionDataset(Dataset):
    def __init__(self, image_dir: str, resolution: int, clip_processor, seed: int = 42):
        self.paths = sorted(
            p for p in glob.glob(os.path.join(image_dir, "*")) if p.lower().endswith(IMG_EXTS)
        )
        if not self.paths:
            raise SystemExit(f"Không có ảnh trong {image_dir}")
        self.resolution = resolution
        self.latent_size = resolution // 8
        self.clip_processor = clip_processor
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.paths[idx])
        img_bgr = cv2.resize(img_bgr, (self.resolution, self.resolution))

        amodal_mask = estimate_object_mask(img_bgr)
        occluded_bgr, _ = add_synthetic_occluder(img_bgr, amodal_mask, self.rng)

        gt_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        occluded_rgb = cv2.cvtColor(occluded_bgr, cv2.COLOR_BGR2RGB)

        gt_pixel = self.tf(Image.fromarray(gt_rgb))

        clip_inputs = self.clip_processor(images=Image.fromarray(occluded_rgb), return_tensors="pt")
        clip_pixel = clip_inputs["pixel_values"][0]

        mask_small = cv2.resize(amodal_mask.astype(np.float32), (self.latent_size, self.latent_size))
        mask_latent = torch.from_numpy(mask_small).unsqueeze(0)

        return {"gt_pixel": gt_pixel, "clip_pixel": clip_pixel, "mask_latent": mask_latent}


class FeatureProjector(nn.Module):
    def __init__(self, clip_dim: int, cross_attention_dim: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = nn.Linear(clip_dim, cross_attention_dim * num_tokens)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, clip_embed: torch.Tensor) -> torch.Tensor:
        x = self.proj(clip_embed)
        x = x.view(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


def expand_conv_in(unet: UNet2DConditionModel, extra_channels: int = 1) -> nn.Conv2d:
    old_conv = unet.conv_in
    new_conv = nn.Conv2d(
        old_conv.in_channels + extra_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size, padding=old_conv.padding,
    )
    with torch.no_grad():
        new_conv.weight[:, : old_conv.in_channels] = old_conv.weight
        new_conv.weight[:, old_conv.in_channels:] = 0.0
        new_conv.bias[:] = old_conv.bias
    return new_conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--num_feature_tokens", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)

    paths = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    output_dir = os.path.join(paths["output_dir"], "appearance_conditioned")
    os.makedirs(output_dir, exist_ok=True)

    set_seed(train_cfg.get("seed", 42))
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        mixed_precision=train_cfg.get("mixed_precision", "fp16"),
    )
    device = accelerator.device

    base_model = model_cfg["base_model"]
    print(f"Tải base model: {base_model}")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    print(f"Tải CLIP image encoder: {args.clip_model}")
    clip_vision = CLIPVisionModel.from_pretrained(args.clip_model)
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model)
    clip_dim = clip_vision.config.hidden_size
    cross_attention_dim = unet.config.cross_attention_dim

    vae.requires_grad_(False)
    clip_vision.requires_grad_(False)

    unet.requires_grad_(False)
    new_conv_in = expand_conv_in(unet, extra_channels=1)
    unet.conv_in = new_conv_in
    unet.conv_in.requires_grad_(True)

    lora_config = LoraConfig(
        r=train_cfg["lora_rank"], lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    for n, p in unet.named_parameters():
        if "conv_in" in n:
            p.requires_grad_(True)

    feature_projector = FeatureProjector(clip_dim, cross_attention_dim, args.num_feature_tokens)

    trainable_params = (
        [p for p in unet.parameters() if p.requires_grad] + list(feature_projector.parameters())
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Số tham số trainable: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=float(train_cfg["learning_rate"]))

    dataset = SelfSupervisedOcclusionDataset(paths["train_ready"], model_cfg["resolution"], clip_processor)
    print(f"Số ảnh training (self-supervised occlusion): {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=train_cfg["train_batch_size"], shuffle=True)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / train_cfg["gradient_accumulation_steps"])
    max_train_steps = train_cfg.get("max_train_steps") or (train_cfg["num_train_epochs"] * num_update_steps_per_epoch)

    unet, feature_projector, optimizer, dataloader = accelerator.prepare(
        unet, feature_projector, optimizer, dataloader
    )
    vae.to(device, dtype=torch.float16)
    clip_vision.to(device, dtype=torch.float16)

    print(f"Bắt đầu train: {max_train_steps} steps (mask + feature-vector conditioning)")
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    unet.train()
    feature_projector.train()
    while global_step < max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                gt_pixel = batch["gt_pixel"].to(device, dtype=torch.float16)
                clip_pixel = batch["clip_pixel"].to(device, dtype=torch.float16)
                mask_latent = batch["mask_latent"].to(device, dtype=torch.float16)

                with torch.no_grad():
                    latents = vae.encode(gt_pixel).latent_dist.sample() * vae.config.scaling_factor
                    clip_out = clip_vision(pixel_values=clip_pixel).pooler_output

                encoder_hidden_states = feature_projector(clip_out)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                unet_input = torch.cat([noisy_latents, mask_latent], dim=1)

                model_pred = unet(unet_input, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

                if global_step % train_cfg["checkpointing_steps"] == 0:
                    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    accelerator.unwrap_model(unet).save_pretrained(ckpt_dir)
                    torch.save(accelerator.unwrap_model(feature_projector).state_dict(),
                               os.path.join(ckpt_dir, "feature_projector.pt"))

            if global_step >= max_train_steps:
                break

    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    accelerator.unwrap_model(unet).save_pretrained(final_dir)
    torch.save(accelerator.unwrap_model(feature_projector).state_dict(),
               os.path.join(final_dir, "feature_projector.pt"))

    print(f"\nHoàn tất. Model (mask + feature-vector conditioned) lưu ở: {final_dir}")
    print(f"Loss cuối: {loss.item():.4f}")


if __name__ == "__main__":
    main()
