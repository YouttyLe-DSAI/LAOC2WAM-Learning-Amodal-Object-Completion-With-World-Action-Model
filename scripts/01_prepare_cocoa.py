"""
Chuẩn bị dữ liệu ground-truth từ COCOA (Semantic Amodal Segmentation),
dùng qua loader của shunk031/huggingface-datasets_COCOA.

QUAN TRỌNG — bước thủ công bắt buộc trước khi chạy script này:
  1. Xin quyền tải annotation COCOA (tar.gz) tại repo gốc:
     https://github.com/Wakeupbuddy/amodalAPI#setup  (Google Drive, tác giả
     duyệt truy cập thủ công).
  2. Tải file `cocoa_annotation.tar.gz` về máy/Kaggle, không commit vào git
     (file khá nặng và có điều khoản sử dụng riêng).
  3. Truyền đường dẫn file đó qua --annotation_targz bên dưới.

Loader HF dataset của shunk031 sẽ tự ghép với ảnh gốc COCO train2014/val2014
(tải qua internet khi chạy lần đầu).

Usage:
    python scripts/01_prepare_cocoa.py \
        --annotation_targz /path/to/cocoa_annotation.tar.gz \
        --out data/cocoa --n_samples 200
"""
from __future__ import annotations

import argparse
import os

from PIL import Image
import numpy as np

import datasets as ds

from src.amodal_pipeline.mask_utils import save_mask_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation_targz", required=True,
                     help="Đường dẫn tới cocoa_annotation.tar.gz đã tải thủ công")
    ap.add_argument("--split", default="COCO", help="config name của loader (mặc định 'COCO')")
    ap.add_argument("--out", default="data/cocoa")
    ap.add_argument("--n_samples", type=int, default=200,
                     help="Số ảnh lấy ra làm ground-truth eval (không cần load hết)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "amodal_masks"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "visible_masks"), exist_ok=True)

    print("Đang load COCOA qua huggingface datasets (lần đầu sẽ tải ảnh COCO gốc)...")
    dataset = ds.load_dataset(
        path="shunk031/COCOA",
        name=args.split,
        data_dir=args.annotation_targz,
        decode_rle=True,
        trust_remote_code=True,
    )

    train_split = dataset["train"] if "train" in dataset else list(dataset.values())[0]

    n = min(args.n_samples, len(train_split))
    print(f"Xuất {n} ảnh + mask ra {args.out} ...")

    for i in range(n):
        item = train_split[i]
        image: Image.Image = item["image"]
        image_id = item.get("image_id", i)
        annotations = item.get("annotations", [])

        img_path = os.path.join(args.out, "images", f"{image_id}.jpg")
        image.convert("RGB").save(img_path, quality=95)

        # Mỗi annotation có 'segmentation' (amodal, đã decode nếu decode_rle=True)
        # và 'visible_mask' nếu loader cung cấp. Gộp tất cả instance lại thành
        # 1 mask nhị phân đơn giản cho demo — với dữ liệu thật bạn có thể giữ
        # riêng từng instance nếu cần multi-object.
        h, w = np.array(image).shape[:2]
        amodal_union = np.zeros((h, w), dtype=np.uint8)
        visible_union = np.zeros((h, w), dtype=np.uint8)

        for ann in annotations:
            seg = ann.get("segmentation")
            if seg is not None:
                seg_arr = np.array(seg)
                if seg_arr.shape[:2] == (h, w):
                    amodal_union |= (seg_arr > 0).astype(np.uint8)
            vis = ann.get("visible_mask")
            if vis is not None:
                vis_arr = np.array(vis)
                if vis_arr.shape[:2] == (h, w):
                    visible_union |= (vis_arr > 0).astype(np.uint8)

        save_mask_png(amodal_union, os.path.join(args.out, "amodal_masks", f"{image_id}.png"))
        save_mask_png(visible_union, os.path.join(args.out, "visible_masks", f"{image_id}.png"))

    print("Xong. Cấu trúc:")
    print(f"  {args.out}/images/<id>.jpg")
    print(f"  {args.out}/amodal_masks/<id>.png   (ground truth cho eval)")
    print(f"  {args.out}/visible_masks/<id>.png")


if __name__ == "__main__":
    main()
