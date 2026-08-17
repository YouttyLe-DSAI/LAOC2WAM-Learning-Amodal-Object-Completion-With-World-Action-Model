"""
Đánh giá mô hình (base vs sau fine-tune LoRA) trên vùng nhìn thấy được của
object, theo đúng bộ metric mà paper Multi-Agent-Amodal dùng: CLIP score,
LPIPS, SSIM — so sánh vùng visible gốc với vùng tương ứng trong ảnh completion.

Usage:
    python scripts/05_evaluate.py --config configs/config.yaml \
        --lora_path outputs/lora/final   # bỏ qua để eval base model
"""
from __future__ import annotations

import argparse
import glob
import os

import lpips
import numpy as np
import torch
import yaml
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel
from transformers import CLIPModel, CLIPProcessor


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def clip_score(clip_model, clip_processor, image: Image.Image, text: str, device) -> float:
    inputs = clip_processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = clip_model(**inputs)
    img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((img_emb @ txt_emb.T).item()) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--lora_path", default=None, help="Nếu bỏ trống -> eval base model")
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cocoa_dir = cfg["paths"]["cocoa_dir"]
    image_paths = sorted(glob.glob(os.path.join(cocoa_dir, "images", "*.jpg")))[
        : cfg["eval"]["n_test_images"]
    ]
    if not image_paths:
        raise SystemExit(f"Không có ảnh test trong {cocoa_dir}/images. Chạy script 01 trước.")

    print(f"Tải pipeline inpainting: {cfg['model']['base_model']}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        cfg["model"]["base_model"], torch_dtype=torch.float16
    ).to(device)

    if args.lora_path:
        print(f"Nạp LoRA weights từ {args.lora_path}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora_path)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    clip_scores, lpips_scores, ssim_scores = [], [], []

    for img_path in image_paths:
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(cocoa_dir, "amodal_masks", f"{image_id}.png")
        if not os.path.exists(mask_path):
            continue

        image = Image.open(img_path).convert("RGB").resize((512, 512))
        mask = Image.open(mask_path).convert("L").resize((512, 512))

        prompt = "a complete object, high quality photo"
        result = pipe(prompt=prompt, image=image, mask_image=mask).images[0]

        clip_scores.append(clip_score(clip_model, clip_processor, result, prompt, device))

        img_np = np.array(image.convert("L"))
        res_np = np.array(result.convert("L"))
        ssim_scores.append(ssim(img_np, res_np))

        t1 = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1
        t2 = torch.tensor(np.array(result)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1
        lpips_scores.append(lpips_fn(t1, t2).item())

    print("\n=== Kết quả đánh giá ===")
    print(f"Ảnh test: {len(clip_scores)}")
    print(f"CLIP score  (cao hơn tốt hơn): {np.mean(clip_scores):.3f}")
    print(f"LPIPS       (thấp hơn tốt hơn): {np.mean(lpips_scores):.3f}")
    print(f"SSIM        (cao hơn tốt hơn): {np.mean(ssim_scores):.3f}")


if __name__ == "__main__":
    main()
