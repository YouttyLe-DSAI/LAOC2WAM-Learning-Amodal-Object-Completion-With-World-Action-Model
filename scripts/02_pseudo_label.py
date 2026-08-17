"""
Sinh pseudo-label cho ảnh trong data/raw (ảnh của bạn, chưa có mask amodal):
  - Occlusion Identification Agent -> tên occluder
  - Boundary Analysis Agent        -> object có bị cắt biên không
  - Description Agent              -> P_text (caption chi tiết, dùng để train)

Kết quả ghi ra data/pseudo_labels/<image_id>.json, có thể dùng ngay ở bước 03.

Yêu cầu: biến môi trường OPENAI_API_KEY (hoặc Kaggle Secret cùng tên).

Usage:
    python scripts/02_pseudo_label.py --images data/raw --out data/pseudo_labels \
        --query "the main object"
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob

from tqdm import tqdm

from src.amodal_pipeline.agents import (
    identify_occlusion,
    analyze_boundary,
    generate_description,
)


IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/raw")
    ap.add_argument("--out", default="data/pseudo_labels")
    ap.add_argument("--query", default="the main object in the image",
                     help="Mô tả ngắn để agent biết cần tìm object nào")
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    image_paths = sorted(
        p for p in glob(os.path.join(args.images, "*")) if p.lower().endswith(IMG_EXTS)
    )
    if not image_paths:
        raise SystemExit(f"Không tìm thấy ảnh nào trong {args.images}. Bỏ ảnh vào đó trước.")

    print(f"Tìm thấy {len(image_paths)} ảnh. Bắt đầu sinh pseudo-label...")

    for img_path in tqdm(image_paths):
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.out, f"{image_id}.json")
        if os.path.exists(out_path):
            continue  # đã xử lý, bỏ qua (resume-friendly)

        try:
            occ = identify_occlusion(img_path, args.query, model=args.model)
            boundary = analyze_boundary(img_path, edge_touch_prior=[], model=args.model)
            caption = generate_description(img_path, occ.target_object, model=args.model)
        except Exception as e:
            print(f"[WARN] Lỗi ở {img_path}: {e}")
            continue

        record = {
            "image_path": img_path,
            "target_object": occ.target_object,
            "is_occluded": occ.is_occluded,
            "occluders": occ.occluders,
            "boundary": boundary,
            "caption": caption,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"Xong. Pseudo-label lưu ở {args.out}/*.json")
    print("Lưu ý: script này CHƯA sinh M_visible/M_occ dạng pixel mask thật "
          "(cần Grounded-SAM, xem TODO trong src/amodal_pipeline/agents.py). "
          "Với fine-tune LoRA kiểu DreamBooth ở bước 04, caption (P_text) là "
          "thứ quan trọng nhất — mask pixel chỉ cần nếu bạn train luôn cả "
          "inpainting mask-conditioned model.")


if __name__ == "__main__":
    main()
