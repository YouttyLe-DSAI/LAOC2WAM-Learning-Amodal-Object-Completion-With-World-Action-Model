"""
Chuẩn bị dữ liệu train cho Appearance branch TỪ DATASET DREAMBOOTH CHÍNH THỨC
(https://github.com/google/dreambooth) — KHÔNG cần GPT-4o/API key, vì dataset
đã có sẵn class name chuẩn cho từng subject.

Trước khi chạy:
    git clone https://github.com/google/dreambooth.git third_party/dreambooth-dataset

Cấu trúc dataset gốc:
    third_party/dreambooth-dataset/dataset/<subject_name>/*.jpg
    third_party/dreambooth-dataset/dataset/prompts_and_classes.txt
        (mỗi dòng: "<subject_name> <class_name>", theo đúng định dạng gốc)

Usage:
    python scripts/02b_prepare_dreambooth_data.py \
        --dreambooth_dir third_party/dreambooth-dataset \
        --subject backpack \
        --out data/train_ready \
        --instance_token sks
"""
from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob

from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png")


def parse_subject_classes(dreambooth_dir: str) -> dict:
    """Đọc dataset/prompts_and_classes.txt -> {subject_name: class_name}."""
    path = os.path.join(dreambooth_dir, "dataset", "prompts_and_classes.txt")
    mapping = {}
    if not os.path.exists(path):
        print(f"[WARN] Không tìm thấy {path}, sẽ cần truyền --class_name thủ công.")
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Định dạng file gốc có thể là "subject_name class_name" hoặc có dấu ':'
            parts = re.split(r"[,:\t]| {2,}", line)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                subject, cls = parts[0], parts[-1]
                mapping[subject] = cls
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreambooth_dir", default="third_party/dreambooth-dataset")
    ap.add_argument("--subject", required=True,
                     help="Tên thư mục subject trong dataset/, VD: backpack, dog6, clock")
    ap.add_argument("--class_name", default=None,
                     help="Ghi đè class name nếu không parse được từ prompts_and_classes.txt")
    ap.add_argument("--out", default="data/train_ready")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--instance_token", default="sks",
                     help="Token đặc trưng kiểu DreamBooth (mặc định 'sks' như paper gốc)")
    args = ap.parse_args()

    subject_dir = os.path.join(args.dreambooth_dir, "dataset", args.subject)
    if not os.path.isdir(subject_dir):
        raise SystemExit(
            f"Không tìm thấy {subject_dir}. Chạy trước:\n"
            f"  git clone https://github.com/google/dreambooth.git {args.dreambooth_dir}\n"
            f"Và kiểm tra tên subject đúng chưa (xem trong {args.dreambooth_dir}/dataset/)."
        )

    class_map = parse_subject_classes(args.dreambooth_dir)
    class_name = args.class_name or class_map.get(args.subject)
    if not class_name:
        raise SystemExit(
            f"Không xác định được class name cho subject '{args.subject}'. "
            "Truyền thủ công qua --class_name, VD: --class_name backpack"
        )

    img_dir = os.path.join(args.out, "instance_images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out, "class_images"), exist_ok=True)

    src_paths = sorted(p for p in glob(os.path.join(subject_dir, "*")) if p.lower().endswith(IMG_EXTS))
    if not src_paths:
        raise SystemExit(f"Không có ảnh nào trong {subject_dir}")

    # QUAN TRỌNG: metadata.jsonl phải nằm CÙNG thư mục với ảnh (img_dir),
    # không phải ở args.out -- vì datasets.load_dataset("imagefolder", ...)
    # tìm cả 2 thứ trong cùng 1 thư mục.
    meta_path = os.path.join(img_dir, "metadata.jsonl")
    instance_prompt = f"a photo of {args.instance_token} {class_name}"

    n_written = 0
    with open(meta_path, "w", encoding="utf-8") as meta_f:
        for i, src_path in enumerate(src_paths):
            im = Image.open(src_path).convert("RGB")
            im = im.resize((args.resolution, args.resolution), Image.LANCZOS)
            dst_name = f"{args.subject}_{i}.jpg"
            im.save(os.path.join(img_dir, dst_name), quality=95)

            meta_f.write(json.dumps({"file_name": dst_name, "text": instance_prompt},
                                     ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Subject: {args.subject}  |  Class: {class_name}")
    print(f"Instance prompt: \"{instance_prompt}\"")
    print(f"Class prompt (cho prior preservation): \"a photo of {class_name}\"")
    print(f"Đã ghi {n_written} ảnh vào {img_dir}")
    print("\nCập nhật configs/config.yaml trước khi train:")
    print(f'  paths.train_ready: "{img_dir}"')
    print(f'  instance_prompt: "{instance_prompt}"')
    print(f'  class_prompt: "a photo of {class_name}"')
    print("\nSau đó chạy: python scripts/04_train_lora.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
