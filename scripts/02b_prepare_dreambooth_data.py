"""
Chuẩn bị dữ liệu train cho Appearance branch TỪ DATASET DREAMBOOTH CHÍNH THỨC
(https://github.com/google/dreambooth) — KHÔNG cần GPT-4o/API key, vì dataset
đã có sẵn class name chuẩn cho từng subject.
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
            parts = re.split(r"[,:\t]| {2,}", line)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                subject, cls = parts[0], parts[-1]
                mapping[subject] = cls
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreambooth_dir", default="third_party/dreambooth-dataset")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--class_name", default=None)
    ap.add_argument("--out", default="data/train_ready")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--instance_token", default="sks")
    args = ap.parse_args()

    subject_dir = os.path.join(args.dreambooth_dir, "dataset", args.subject)
    if not os.path.isdir(subject_dir):
        raise SystemExit(f"Không tìm thấy {subject_dir}.")

    class_map = parse_subject_classes(args.dreambooth_dir)
    class_name = args.class_name or class_map.get(args.subject)
    if not class_name:
        raise SystemExit(
            f"Không xác định được class name cho subject '{args.subject}'. "
            "Truyền thủ công qua --class_name."
        )

    img_dir = os.path.join(args.out, "instance_images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out, "class_images"), exist_ok=True)

    src_paths = sorted(p for p in glob(os.path.join(subject_dir, "*")) if p.lower().endswith(IMG_EXTS))
    if not src_paths:
        raise SystemExit(f"Không có ảnh nào trong {subject_dir}")

    meta_path = os.path.join(img_dir, "metadata.jsonl")
    instance_prompt = f"a photo of {args.instance_token} {class_name}"

    n_written = 0
    with open(meta_path, "w", encoding="utf-8") as meta_f:
        for i, src_path in enumerate(src_paths):
            im = Image.open(src_path).convert("RGB")
            im = im.resize((args.resolution, args.resolution), Image.LANCZOS)
            dst_name = f"{args.subject}_{i}.jpg"
            im.save(os.path.join(img_dir, dst_name), quality=95)
            meta_f.write(json.dumps({"file_name": dst_name, "text": instance_prompt}, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Subject: {args.subject}  |  Class: {class_name}")
    print(f"Instance prompt: \"{instance_prompt}\"")
    print(f"Class prompt (cho prior preservation): \"a photo of {class_name}\"")
    print(f"Đã ghi {n_written} ảnh vào {args.out}")
    print("\nCập nhật configs/config.yaml trước khi train:")
    print(f'  instance_prompt: "{instance_prompt}"')
    print(f'  class_prompt: "a photo of {class_name}"')


if __name__ == "__main__":
    main()
