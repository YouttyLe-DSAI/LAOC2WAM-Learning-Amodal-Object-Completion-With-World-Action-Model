from __future__ import annotations

import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from peft import PeftModel
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPModel, CLIPProcessor

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


def add_synthetic_occluder(image_bgr, amodal_mask, rng):
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


class FeatureProjector(nn.Module):
    def __init__(self, clip_dim, cross_attention_dim, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = nn.Linear(clip_dim, cross_attention_dim * num_tokens)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, clip_embed):
        x = self.proj(clip_embed)
        x = x.view(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


def expand_conv_in(unet, extra_channels=1):
    old_conv = unet.conv_in
    new_conv = nn.Conv2d(old_conv.in_channels + extra_channels, old_conv.out_channels,
                          kernel_size=old_conv.kernel_size, padding=old_conv.padding)
    return new_conv


@torch.no_grad()
def run_inference(unet, feature_projector, vae, clip_vision, clip_processor,
                   noise_scheduler, occluded_rgb, amodal_mask, resolution, device,
                   num_inference_steps=30):
    latent_size = resolution // 8

    clip_inputs = clip_processor(images=Image.fromarray(occluded_rgb), return_tensors="pt")
    clip_pixel = clip_inputs["pixel_values"].to(device, dtype=torch.float16)
    clip_feat = clip_vision(pixel_values=clip_pixel).pooler_output
    encoder_hidden_states = feature_projector(clip_feat)

    mask_small = cv2.resize(amodal_mask.astype(np.float32), (latent_size, latent_size))
    mask_latent = torch.from_numpy(mask_small).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)

    latents = torch.randn((1, 4, latent_size, latent_size), device=device, dtype=torch.float16)
    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    for t in noise_scheduler.timesteps:
        unet_input = torch.cat([latents, mask_latent], dim=1)
        noise_pred = unet(unet_input, t, encoder_hidden_states).sample
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].permute(1, 2, 0).float().cpu().numpy()
    image = (image * 255).round().astype(np.uint8)
    return image


def clip_image_embed(clip_model, pixel_values):
    """Ham thay the get_image_features() -- API do bi doi hanh vi giua cac
    ban transformers, goi truc tiep vision_model + visual_projection cho
    chac chan hoat dong dung."""
    vision_out = clip_model.vision_model(pixel_values=pixel_values)
    return clip_model.visual_projection(vision_out.pooler_output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default="outputs/appearance_conditioned/final")
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--out_dir", default="outputs/appearance_conditioned/eval_samples")
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolution = cfg["model"]["resolution"]
    base_model = cfg["model"]["base_model"]
    test_dir = args.test_dir or cfg["paths"]["train_ready"]

    os.makedirs(args.out_dir, exist_ok=True)

    print("Tải VAE + scheduler...")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device, dtype=torch.float16)
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    print("Tải UNet base + mở rộng conv_in (5 kênh, khớp shape checkpoint)...")
    base_unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    base_unet.conv_in = expand_conv_in(base_unet, extra_channels=1)

    print(f"Nạp checkpoint (LoRA + conv_in đã train): {args.checkpoint}")
    unet = PeftModel.from_pretrained(base_unet, args.checkpoint)
    unet = unet.to(device, dtype=torch.float16)
    unet.eval()

    print(f"Tải CLIP: {args.clip_model}")
    clip_vision = CLIPVisionModel.from_pretrained(args.clip_model).to(device, dtype=torch.float16)
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model)
    clip_dim = clip_vision.config.hidden_size
    cross_attention_dim = base_unet.config.cross_attention_dim

    feature_projector = FeatureProjector(clip_dim, cross_attention_dim).to(device, dtype=torch.float16)
    fp_path = os.path.join(args.checkpoint, "feature_projector.pt")
    feature_projector.load_state_dict(torch.load(fp_path, map_location=device))
    feature_projector.eval()

    eval_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    eval_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    paths = sorted(p for p in glob.glob(os.path.join(test_dir, "*")) if p.lower().endswith(IMG_EXTS))
    rng = random.Random(123)
    sample_paths = rng.sample(paths, min(args.n_samples, len(paths)))

    ssim_scores, clip_scores = [], []
    for i, path in enumerate(sample_paths):
        img_bgr = cv2.resize(cv2.imread(path), (resolution, resolution))
        amodal_mask = estimate_object_mask(img_bgr)
        occluded_bgr, _ = add_synthetic_occluder(img_bgr, amodal_mask, rng)
        occluded_rgb = cv2.cvtColor(occluded_bgr, cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        result = run_inference(unet, feature_projector, vae, clip_vision, clip_processor,
                                noise_scheduler, occluded_rgb, amodal_mask, resolution, device)

        out_path = os.path.join(args.out_dir, f"sample_{i}.png")
        combined = np.concatenate([occluded_rgb, result, gt_rgb], axis=1)
        Image.fromarray(combined).save(out_path)

        s = ssim(cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2GRAY), cv2.cvtColor(result, cv2.COLOR_RGB2GRAY))
        ssim_scores.append(s)

        clip_in = eval_clip_processor(images=[Image.fromarray(result), Image.fromarray(gt_rgb)],
                                       return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_image_embed(eval_clip_model, clip_in["pixel_values"])
        emb = emb / emb.norm(dim=-1, keepdim=True)
        clip_scores.append((emb[0] @ emb[1]).item())

        print(f"[{i+1}/{len(sample_paths)}] {os.path.basename(path)}: SSIM={s:.4f}  CLIP_sim={clip_scores[-1]:.4f}")

    print(f"\n=== KẾT QUẢ TRUNG BÌNH ({len(sample_paths)} mẫu) ===")
    print(f"SSIM trung bình: {np.mean(ssim_scores):.4f}")
    print(f"CLIP similarity trung bình: {np.mean(clip_scores):.4f}")
    print(f"\nẢnh minh hoạ (input | sinh ra | ground truth) lưu ở: {args.out_dir}/sample_*.png")


if __name__ == "__main__":
    main()