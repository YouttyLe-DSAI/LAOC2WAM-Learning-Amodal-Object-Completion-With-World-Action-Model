"""
Kết hợp THẬT 2 nhánh (không phải minh hoạ giả):
  1. Mask model (train trên COCOA)         -> dự đoán amodal mask
  2. Appearance model (personalized, train
     trên vài ảnh của 1 vật cụ thể)         -> nhận mask dự đoán + feature
     vector (trích từ ảnh input) -> sinh ảnh hoàn chỉnh

Usage:
    python scripts/08_combined_pipeline.py --config configs/config.yaml \
        --mask_checkpoint outputs/mask_model/best.pt \
        --appearance_checkpoint outputs/appearance_conditioned/final \
        --test_image data/train_ready_backpack/instance_images/backpack_0.jpg
"""
from __future__ import annotations

import argparse
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
from torchvision import transforms
from transformers import CLIPVisionModel, CLIPImageProcessor

from src.amodal_pipeline.mask_model import AmodalMaskUNet


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
    occ_w = max(int(obj_w * 0.35), 10)
    occ_h = max(int(obj_h * 0.35), 10)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    occluder_mask = np.zeros((h, w), np.uint8)
    out = image_bgr.copy()
    axes_e = (occ_w // 2, occ_h // 2)
    cv2.ellipse(out, (cx, cy), axes_e, 0, 0, 360, (40, 40, 40), -1)
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
    return nn.Conv2d(old_conv.in_channels + extra_channels, old_conv.out_channels,
                      kernel_size=old_conv.kernel_size, padding=old_conv.padding)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--mask_checkpoint", default="outputs/mask_model/best.pt")
    ap.add_argument("--appearance_checkpoint", default="outputs/appearance_conditioned/final")
    ap.add_argument("--test_image", required=True)
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--out", default="outputs/combined_result.png")
    ap.add_argument("--out_dir", default="outputs/combined_stages")
    ap.add_argument("--num_inference_steps", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolution = cfg["model"]["resolution"]
    base_model = cfg["model"]["base_model"]
    latent_size = resolution // 8

    img_bgr = cv2.resize(cv2.imread(args.test_image), (resolution, resolution))
    true_amodal = estimate_object_mask(img_bgr)
    rng = random.Random(7)
    occluded_bgr, occluder_mask = add_synthetic_occluder(img_bgr, true_amodal, rng)
    occluded_rgb = cv2.cvtColor(occluded_bgr, cv2.COLOR_BGR2RGB)
    gt_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    visible_mask = np.logical_and(true_amodal > 0, occluder_mask == 0).astype(np.float32)

    print("Nạp Mask model (COCOA)...")
    mask_model = AmodalMaskUNet(base_ch=32).to(device)
    mask_model.load_state_dict(torch.load(args.mask_checkpoint, map_location=device))
    mask_model.eval()

    img_t = transforms.ToTensor()(Image.fromarray(occluded_rgb)).unsqueeze(0).to(device)
    vis_t = torch.from_numpy(visible_mask).unsqueeze(0).unsqueeze(0).to(device)
    pred_mask_logits = mask_model(img_t, vis_t)
    pred_mask = (torch.sigmoid(pred_mask_logits)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
    print(f"Mask dự đoán: {pred_mask.sum()} pixel (so với ground truth {true_amodal.sum()} pixel)")

    print("Nạp Appearance model (personalized)...")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device, dtype=torch.float16)
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    base_unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    base_unet.conv_in = expand_conv_in(base_unet, extra_channels=1)
    unet = PeftModel.from_pretrained(base_unet, args.appearance_checkpoint).to(device, dtype=torch.float16)
    unet.eval()

    clip_vision = CLIPVisionModel.from_pretrained(args.clip_model).to(device, dtype=torch.float16)
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model)
    clip_dim = clip_vision.config.hidden_size
    cross_attention_dim = base_unet.config.cross_attention_dim

    feature_projector = FeatureProjector(clip_dim, cross_attention_dim).to(device, dtype=torch.float16)
    feature_projector.load_state_dict(
        torch.load(os.path.join(args.appearance_checkpoint, "feature_projector.pt"), map_location=device)
    )
    feature_projector.eval()

    clip_inputs = clip_processor(images=Image.fromarray(occluded_rgb), return_tensors="pt")
    clip_pixel = clip_inputs["pixel_values"].to(device, dtype=torch.float16)
    clip_feat = clip_vision(pixel_values=clip_pixel).pooler_output
    encoder_hidden_states = feature_projector(clip_feat)

    # Mask dùng để sinh: vùng model được TỰ DO vẽ (occluder_mask thật, không
    # phải amodal mask đầy đủ -- vì vùng visible KHÔNG cần sinh, chỉ cần giữ
    # nguyên từ ảnh input).
    gen_region = occluder_mask.astype(np.float32)  # 1 = model tự do sinh, 0 = giữ nguyên ảnh gốc
    mask_small = cv2.resize(pred_mask.astype(np.float32), (latent_size, latent_size))
    mask_latent = torch.from_numpy(mask_small).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)
    gen_region_small = cv2.resize(gen_region, (latent_size, latent_size))
    gen_mask_latent = torch.from_numpy(gen_region_small).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)
    gen_mask_latent = (gen_mask_latent > 0.5).float()

    # Encode ẢNH INPUT THẬT (bị che) -- vùng ngoài occluder trong latent này
    # là "đáp án đúng" cần giữ nguyên xuyên suốt quá trình khử nhiễu.
    known_pixel = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])(
        Image.fromarray(occluded_rgb)
    ).unsqueeze(0).to(device, dtype=torch.float16)
    known_latents = vae.encode(known_pixel).latent_dist.sample() * vae.config.scaling_factor

    latents = torch.randn((1, 4, latent_size, latent_size), device=device, dtype=torch.float16)
    noise_scheduler.set_timesteps(args.num_inference_steps, device=device)
    for t in noise_scheduler.timesteps:
        # RÀNG BUỘC (RePaint-style): vùng ngoài occluder = ép về đúng ảnh
        # gốc (đã thêm nhiễu tương ứng mức t); vùng occluder = giữ nguyên
        # kết quả model đang khử nhiễu (được tự do sinh).
        step_noise = torch.randn_like(known_latents)
        noisy_known = noise_scheduler.add_noise(known_latents, step_noise, t.unsqueeze(0))
        latents = gen_mask_latent * latents + (1 - gen_mask_latent) * noisy_known

        unet_input = torch.cat([latents, mask_latent], dim=1)
        noise_pred = unet(unet_input, t, encoder_hidden_states).sample
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    # Ép buộc cuối cùng (t=0, không nhiễu): đảm bảo vùng ngoài occluder
    # khớp CHÍNH XÁC pixel gốc, không chỉ gần đúng.
    latents = gen_mask_latent * latents + (1 - gen_mask_latent) * known_latents

    latents = latents / vae.config.scaling_factor
    decoded = vae.decode(latents).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    result = (decoded[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)

    # ---------- Xuất riêng từng giai đoạn (rõ ràng cho báo cáo) ----------
    stage_dir = args.out_dir
    os.makedirs(stage_dir, exist_ok=True)

    # 1. Ảnh input bị che
    Image.fromarray(occluded_rgb).save(os.path.join(stage_dir, "01_input_occluded.png"))

    # 2. Mask trắng đen (mask model dự đoán)
    mask_bw = (pred_mask * 255).astype(np.uint8)
    Image.fromarray(mask_bw, mode="L").save(os.path.join(stage_dir, "02_mask_bw.png"))

    # 3. Vật thể được tách riêng (cutout): áp mask lên ảnh gốc, nền = trắng
    cutout = np.full_like(gt_rgb, 255)
    full_mask_bool = pred_mask.astype(bool)
    cutout[full_mask_bool] = gt_rgb[full_mask_bool]
    Image.fromarray(cutout).save(os.path.join(stage_dir, "03_object_cutout.png"))

    # 4. Kết quả sau khi vá (đã ràng buộc giữ nguyên vùng thấy)
    Image.fromarray(result).save(os.path.join(stage_dir, "04_repainted_result.png"))

    # 5. Ground truth (ảnh gốc thật, không bị che, để đối chiếu)
    Image.fromarray(gt_rgb).save(os.path.join(stage_dir, "05_ground_truth.png"))

    # Ảnh ghép tổng hợp (xem nhanh 1 lần)
    pred_mask_vis = np.stack([mask_bw] * 3, axis=-1)
    combined = np.concatenate([occluded_rgb, pred_mask_vis, cutout, result, gt_rgb], axis=1)
    Image.fromarray(combined).save(args.out)

    print(f"\nHoàn tất pipeline (RePaint-constrained).")
    print(f"Các ảnh giai đoạn riêng biệt lưu tại: {stage_dir}/")
    print(f"  01_input_occluded.png    -- ảnh bị che (input)")
    print(f"  02_mask_bw.png           -- mask trắng đen (Mask model dự đoán)")
    print(f"  03_object_cutout.png     -- vật thể tách riêng (nền trắng)")
    print(f"  04_repainted_result.png  -- kết quả sau khi vá")
    print(f"  05_ground_truth.png      -- ảnh gốc thật để đối chiếu")
    print(f"Ảnh ghép tổng hợp: {args.out}")


if __name__ == "__main__":
    main()
