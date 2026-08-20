"""
Parse dataset COCOA THẬT (Amodal COCO annotations, format gốc Zhu et al. 2017)
-- thay thế hoàn toàn cho synthetic occlusion (01b_prepare_synthetic.py).

QUAN TRỌNG: COCOA không lưu sẵn visible mask -- phải tự tính bằng cách lấy
amodal mask của 1 region, trừ đi HỢP của amodal mask các region có `order`
NHỎ HƠN (tức nằm "phía trước", che mất nó).

Ảnh gốc COCO val2014 tải trực tiếp từ bucket public MS COCO, KHÔNG cần xin
quyền riêng (chỉ annotation amodal mới cần xin quyền).

Usage:
    python scripts/01c_prepare_cocoa_real.py \
        --annotation_json /path/to/COCO_amodal_val2014.json \
        --out data/cocoa_real \
        --n_images 100 \
        --only_occluded
"""
from __future__ import annotations

import argparse
import json
import os
import re

import cv2
import numpy as np
import requests
from PIL import Image

COCO_IMG_BASE = "http://images.cocodataset.org"


def get_image_url(file_name: str) -> str:
    """Tự nhận diện train2014/val2014 từ tên file, tránh lỗi 404 khi
    annotation dùng file train2014 nhưng URL cứng val2014."""
    m = re.search(r"(train2014|val2014)", file_name)
    split = m.group(1) if m else "val2014"
    return f"{COCO_IMG_BASE}/{split}/{file_name}"


def polygon_to_mask(seg_flat, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = seg_flat if (seg_flat and isinstance(seg_flat[0], list)) else [seg_flat]
    for poly in polygons:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def download_image(file_name: str, dest_dir: str):
    dest_path = os.path.join(dest_dir, file_name)
    if os.path.exists(dest_path):
        return dest_path
    url = get_image_url(file_name)
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        return dest_path
    except Exception as e:
        print(f"[WARN] Không tải được {url}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation_json", required=True)
    ap.add_argument("--out", default="data/cocoa_real")
    ap.add_argument("--n_images", type=int, default=100)
    ap.add_argument("--only_occluded", action="store_true")
    args = ap.parse_args()

    raw_dir = os.path.join(args.out, "_raw_coco_images")
    for sub in ("images", "visible_masks", "amodal_masks"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Đọc annotation: {args.annotation_json}")
    with open(args.annotation_json, "r") as f:
        data = json.load(f)

    entries = data["annotations"][: args.n_images]
    print(f"Xử lý {len(entries)} ảnh...")

    n_instances = 0
    for img_idx, entry in enumerate(entries):
        file_name = entry["url"].split("/")[-1]
        regions = entry.get("regions", [])
        if not regions:
            continue

        img_path = download_image(file_name, raw_dir)
        if img_path is None:
            continue

        try:
            with Image.open(img_path) as im:
                width, height = im.size
                im.convert("RGB").save(img_path)
        except Exception as e:
            print(f"[WARN] Ảnh lỗi {img_path}: {e}")
            continue

        regions_sorted = sorted(regions, key=lambda r: r.get("order", 999))
        amodal_masks = [polygon_to_mask(r["segmentation"], height, width) for r in regions_sorted]

        cumulative_front = np.zeros((height, width), dtype=np.uint8)
        for i, region in enumerate(regions_sorted):
            if args.only_occluded and region.get("occlude_rate", 0) <= 0:
                cumulative_front = np.logical_or(cumulative_front, amodal_masks[i]).astype(np.uint8)
                continue

            amodal_mask = amodal_masks[i]
            visible_mask = np.logical_and(amodal_mask > 0, cumulative_front == 0).astype(np.uint8)

            if visible_mask.sum() < 50:
                cumulative_front = np.logical_or(cumulative_front, amodal_mask).astype(np.uint8)
                continue

            instance_id = f"{os.path.splitext(file_name)[0]}_{i}"
            img_out = cv2.imread(img_path)
            cv2.imwrite(os.path.join(args.out, "images", f"{instance_id}.jpg"), img_out)
            cv2.imwrite(os.path.join(args.out, "visible_masks", f"{instance_id}.png"), visible_mask * 255)
            cv2.imwrite(os.path.join(args.out, "amodal_masks", f"{instance_id}.png"), amodal_mask * 255)
            n_instances += 1

            cumulative_front = np.logical_or(cumulative_front, amodal_mask).astype(np.uint8)

        if (img_idx + 1) % 20 == 0:
            print(f"  Đã xử lý {img_idx + 1}/{len(entries)} ảnh, {n_instances} instance...")

    print(f"\nHoàn tất. Tổng {n_instances} instance (vật thể bị che) từ {len(entries)} ảnh.")
    print(f"Cấu trúc: {args.out}/images, visible_masks, amodal_masks")
    print("Đổi paths.cocoa_dir trong config.yaml thành thư mục này rồi chạy:")
    print("  python scripts/04a_train_mask_model.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
