"""
Đánh giá Mask branch: so sánh amodal mask dự đoán vs ground truth COCOA
bằng IoU (Intersection over Union) — metric chuẩn cho segmentation.

Usage:
    python scripts/06_evaluate_mask.py --config configs/config.yaml \
        --checkpoint outputs/mask_model/best.pt
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.amodal_pipeline.mask_model import AmodalMaskUNet


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / union if union > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default="outputs/mask_model/best.pt")
    ap.add_argument("--resolution", type=int, default=256)
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AmodalMaskUNet(base_ch=32).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    cocoa_dir = cfg["paths"]["cocoa_dir"]
    image_paths = sorted(glob.glob(os.path.join(cocoa_dir, "images", "*.jpg")))
    img_tf = transforms.Compose(
        [transforms.Resize((args.resolution, args.resolution)), transforms.ToTensor()]
    )

    ious_amodal_vs_pred = []
    ious_visible_baseline = []  # so sánh với baseline "không làm gì" (visible == amodal)

    with torch.no_grad():
        for img_path in image_paths:
            image_id = os.path.splitext(os.path.basename(img_path))[0]
            vis_path = os.path.join(cocoa_dir, "visible_masks", f"{image_id}.png")
            amo_path = os.path.join(cocoa_dir, "amodal_masks", f"{image_id}.png")
            if not (os.path.exists(vis_path) and os.path.exists(amo_path)):
                continue

            image = img_tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            vis_mask = Image.open(vis_path).convert("L").resize((args.resolution, args.resolution))
            vis_mask_t = torch.from_numpy((np.array(vis_mask) > 127).astype(np.float32))
            vis_mask_t = vis_mask_t.unsqueeze(0).unsqueeze(0).to(device)

            amo_mask = Image.open(amo_path).convert("L").resize((args.resolution, args.resolution))
            amo_np = np.array(amo_mask) > 127

            pred_logits = model(image, vis_mask_t)
            pred_np = (torch.sigmoid(pred_logits)[0, 0].cpu().numpy() > 0.5)

            ious_amodal_vs_pred.append(iou(pred_np, amo_np))
            ious_visible_baseline.append(iou(vis_mask_t[0, 0].cpu().numpy() > 0.5, amo_np))

    print(f"Số ảnh eval: {len(ious_amodal_vs_pred)}")
    print(f"IoU sau fine-tune         : {np.mean(ious_amodal_vs_pred):.4f}")
    print(f"IoU baseline (visible only, chưa fine-tune gì): {np.mean(ious_visible_baseline):.4f}")
    print("-> Nếu IoU sau fine-tune cao hơn rõ rệt baseline, model đã học được cách "
          "'đoán thêm' phần bị che chứ không chỉ copy lại visible mask.")


if __name__ == "__main__":
    main()
