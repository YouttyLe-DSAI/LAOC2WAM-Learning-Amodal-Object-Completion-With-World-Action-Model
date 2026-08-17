"""
Gộp ảnh + pseudo-label (bước 02) thành format chuẩn để script 04 train:

data/train_ready/
├── instance_images/          # ảnh gốc, resize về resolution config
├── metadata.jsonl            # {"file_name": ..., "text": caption}
└── class_images/             # (rỗng, script 04 tự sinh nếu prior_preservation=True)

Format metadata.jsonl tương thích trực tiếp với script train_dreambooth_lora.py
chuẩn của thư viện diffusers (HF).

Usage:
    python scripts/03_prepare_training_data.py \
        --pseudo data/pseudo_labels --out data/train_ready \
        --instance_prompt "a photo of sks object"
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo", default="data/pseudo_labels")
    ap.add_argument("--out", default="data/train_ready")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--instance_prompt", default="a photo of sks object",
                     help="Token đặc trưng kiểu DreamBooth, sẽ được nối vào caption")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "instance_images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out, "class_images"), exist_ok=True)

    records = sorted(glob(os.path.join(args.pseudo, "*.json")))
    if not records:
        raise SystemExit(f"Không có file nào trong {args.pseudo}. Chạy script 02 trước.")

    meta_path = os.path.join(args.out, "metadata.jsonl")
    n_written = 0
    with open(meta_path, "w", encoding="utf-8") as meta_f:
        for rec_path in records:
            with open(rec_path, "r", encoding="utf-8") as f:
                rec = json.load(f)

            src_img = rec["image_path"]
            image_id = os.path.splitext(os.path.basename(rec_path))[0]
            dst_name = f"{image_id}.jpg"
            dst_path = os.path.join(img_dir, dst_name)

            im = Image.open(src_img).convert("RGB")
            im = im.resize((args.resolution, args.resolution), Image.LANCZOS)
            im.save(dst_path, quality=95)

            # Ghép token đặc trưng (kiểu DreamBooth) + caption chi tiết từ Description Agent.
            # Ví dụ: "a photo of sks object, a full ginger tabby cat with green eyes..."
            caption = f"{args.instance_prompt}, {rec['caption']}"

            meta_f.write(json.dumps({"file_name": dst_name, "text": caption},
                                     ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Đã ghi {n_written} ảnh + caption vào {args.out}")
    print(f"metadata.jsonl: {meta_path}")


if __name__ == "__main__":
    main()
