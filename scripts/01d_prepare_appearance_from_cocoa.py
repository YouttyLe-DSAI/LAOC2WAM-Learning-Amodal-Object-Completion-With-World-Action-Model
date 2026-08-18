"""
Trích xuất các vật thể KHÔNG bị che (occlude_rate == 0) từ COCOA để làm dữ
liệu "appearance ground truth" -- THAY THẾ HOÀN TOÀN cho dataset DreamBooth.

Ý tưởng: COCOA cho biết vật nào trong ảnh đang HOÀN TOÀN không bị che (thấy
đủ 100%). Ta cắt riêng từng vật đó ra làm "ảnh đáp án đúng", ghép với tên
class thật (vd "backpack", "clock") làm caption -- không cần token đặc biệt
kiểu "sks"/"zwx" vì đây là học QUY LUẬT CHUNG của cả class, không phải học
thuộc 1 vật cụ thể (personalization).

Khác biệt so với DreamBooth-style (02b_prepare_dreambooth_data.py):
  - DreamBooth: 4-6 ảnh CÙNG 1 vật cụ thể -> model "nhớ" đúng vật đó
  - Script này: hàng trăm/nghìn ảnh KHÁC NHAU cùng 1 class -> model học
    "quy luật chung" của class đó (đúng tinh thần Amodal Appearance
    Completion tổng quát, không phải personalization)

Usage:
    python scripts/01d_prepare_appearance_from_cocoa.py \
        --annotation_json /path/to/COCO_amodal_train2014.json \
        --out data/appearance_from_cocoa \
        --n_images 300 \
        --target_class backpack   # (tuỳ chọn) chỉ lấy 1 class cụ thể
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import requests
from PIL import Image


def _get_image_url(file_name: str) -> str:
    import re
    m = re.search(r"(train2014|val2014)", file_name)
    split = m.group(1) if m else "val2014"
    return f"http://images.cocodataset.org/{split}/{file_name}"


def _polygon_to_mask(seg_flat, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = seg_flat if (seg_flat and isinstance(seg_flat[0], list)) else [seg_flat]
    for poly in polygons:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def _download_image(file_name: str, dest_dir: str):
    dest_path = os.path.join(dest_dir, file_name)
    if os.path.exists(dest_path):
        return dest_path
    try:
        r = requests.get(_get_image_url(file_name), timeout=20)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        return dest_path
    except Exception as e:
        print(f"[WARN] {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation_json", required=True)
    ap.add_argument("--out", default="data/appearance_from_cocoa")
    ap.add_argument("--n_images", type=int, default=300)
    ap.add_argument("--target_class", default=None,
                     help="Chỉ lấy 1 class cụ thể (vd 'backpack'). Bỏ trống = lấy mọi class.")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--min_crop_size", type=int, default=80,
                     help="Bỏ qua vật quá nhỏ (px), tránh crop mờ/vô nghĩa")
    ap.add_argument("--margin_ratio", type=float, default=0.15,
                     help="Mở rộng thêm biên quanh bbox khi crop")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "instance_images")
    raw_dir = os.path.join(args.out, "_raw_coco_images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Đọc annotation: {args.annotation_json}")
    with open(args.annotation_json, "r") as f:
        data = json.load(f)

    entries = data["annotations"][: args.n_images]
    print(f"Quét {len(entries)} ảnh, tìm vật KHÔNG bị che...")

    meta_path = os.path.join(img_dir, "metadata.jsonl")
    n_written = 0
    class_counts = {}

    with open(meta_path, "w", encoding="utf-8") as meta_f:
        for img_idx, entry in enumerate(entries):
            file_name = entry["url"].split("/")[-1]
            regions = entry.get("regions", [])
            if not regions:
                continue

            unoccluded = [r for r in regions if r.get("occlude_rate", 0) == 0]
            if args.target_class:
                unoccluded = [r for r in unoccluded if r.get("name", "").lower() == args.target_class.lower()]
            if not unoccluded:
                continue

            img_path = _download_image(file_name, raw_dir)
            if img_path is None:
                continue
            try:
                with Image.open(img_path) as im:
                    width, height = im.size
                    im.convert("RGB").save(img_path)
            except Exception as e:
                print(f"[WARN] Ảnh lỗi {img_path}: {e}")
                continue

            img_full = cv2.imread(img_path)

            for i, region in enumerate(unoccluded):
                mask = _polygon_to_mask(region["segmentation"], height, width)
                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue
                x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
                w_box, h_box = x1 - x0, y1 - y0
                if w_box < args.min_crop_size or h_box < args.min_crop_size:
                    continue

                mx, my = int(w_box * args.margin_ratio), int(h_box * args.margin_ratio)
                cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
                cx1, cy1 = min(width, x1 + mx), min(height, y1 + my)

                crop = img_full[cy0:cy1, cx0:cx1]
                crop = cv2.resize(crop, (args.resolution, args.resolution))

                class_name = region.get("name", "object")
                instance_id = f"{os.path.splitext(file_name)[0]}_{i}"
                out_path = os.path.join(img_dir, f"{instance_id}.jpg")
                cv2.imwrite(out_path, crop)

                caption = f"a photo of {class_name}"
                meta_f.write(json.dumps({"file_name": f"{instance_id}.jpg", "text": caption},
                                         ensure_ascii=False) + "\n")
                n_written += 1
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            if (img_idx + 1) % 50 == 0:
                print(f"  {img_idx + 1}/{len(entries)} ảnh, {n_written} crop...")

    print(f"\nHoàn tất. Tổng {n_written} ảnh vật thể KHÔNG bị che, {len(class_counts)} class khác nhau.")
    top5 = sorted(class_counts.items(), key=lambda x: -x[1])[:5]
    print(f"5 class nhiều nhất: {top5}")
    print(f"\nDữ liệu: {img_dir}/ (kèm metadata.jsonl)")
    print("Cập nhật configs/config.yaml:")
    print(f'  paths.train_ready: "{img_dir}"')
    print("Rồi chạy: python scripts/04b_train_appearance_general.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
