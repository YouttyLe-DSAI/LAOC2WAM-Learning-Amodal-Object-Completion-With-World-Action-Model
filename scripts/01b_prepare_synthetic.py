"""
Tạo dữ liệu "amodal giả" (synthetic occlusion) từ ảnh THƯỜNG (không cần
annotation COCOA, không cần chờ duyệt gì cả).

Ý tưởng (giống cách SeGAN/PCNet tạo data khi thiếu ground truth thật):
  1. Lấy 1 ảnh gốc, KHÔNG bị che -> coi phần vật thể trong đó là "đáp án
     đúng" (amodal ground truth). Ta tự động ước lượng vùng vật thể bằng
     GrabCut (không cần gán nhãn tay).
  2. Tự vẽ 1 hình che giả (chữ nhật/oval, màu ngẫu nhiên) đè lên ảnh đó
     -> tạo ra ảnh "bị che" (input cho model).
  3. visible_mask = amodal_mask trừ đi vùng bị che giả.
     amodal_mask = mask gốc từ bước 1 (KHÔNG đổi, vì đó là "đáp án đúng").

Kết quả ghi ra đúng cấu trúc mà scripts/04a_train_mask_model.py cần:
  <out>/images/<id>.jpg          <- ảnh ĐÃ bị che giả (input)
  <out>/visible_masks/<id>.png   <- vùng còn nhìn thấy sau khi che
  <out>/amodal_masks/<id>.png    <- ground truth (vật thể đầy đủ)

Usage:
    # Dùng ảnh của bạn (bỏ vài chục ảnh vào data/raw/ trước, mỗi ảnh có
    # 1 vật thể rõ ràng, nền tương đối đơn giản để GrabCut hoạt động tốt)
    python scripts/01b_prepare_synthetic.py --images data/raw --out data/cocoa

    # Hoặc tự tải vài ảnh mẫu public domain nếu data/raw đang trống
    python scripts/01b_prepare_synthetic.py --images data/raw --out data/cocoa --download_samples
"""
from __future__ import annotations

import argparse
import os
import random
from glob import glob

import cv2
import numpy as np
import requests

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# Vài ảnh public-domain/free-license đơn giản (1 vật thể, nền tương đối rõ)
# để test nhanh khi bạn chưa có sẵn ảnh riêng. Có thể thay bằng ảnh của bạn.
SAMPLE_URLS = [
    "https://images.pexels.com/photos/47261/pexels-photo-47261.jpeg?w=640",   # phone
    "https://images.pexels.com/photos/2072181/pexels-photo-2072181.jpeg?w=640",  # cup
    "https://images.pexels.com/photos/1279813/pexels-photo-1279813.jpeg?w=640",  # shoe
    "https://images.pexels.com/photos/374912/pexels-photo-374912.jpeg?w=640",   # apple
    "https://images.pexels.com/photos/1667088/pexels-photo-1667088.jpeg?w=640", # bag
]


def download_samples(dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for i, url in enumerate(SAMPLE_URLS):
        out_path = os.path.join(dest_dir, f"sample_{i}.jpg")
        if os.path.exists(out_path):
            continue
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"Đã tải: {out_path}")
        except Exception as e:
            print(f"[WARN] Không tải được {url}: {e}")


def estimate_object_mask(image_bgr: np.ndarray, border_ratio: float = 0.06) -> np.ndarray:
    """Ước lượng amodal mask (vùng vật thể đầy đủ, CHƯA bị che) bằng GrabCut.
    Giả định vật thể nằm gần giữa ảnh, nền đơn giản hơn vật thể."""
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    bx, by = int(w * border_ratio), int(h * border_ratio)
    rect = (bx, by, w - 2 * bx, h - 2 * by)

    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # ảnh quá nhỏ / GrabCut fail -> fallback: coi cả rect là object
        fallback = np.zeros((h, w), np.uint8)
        fallback[by : h - by, bx : w - bx] = 1
        return fallback

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    if fg_mask.sum() < 0.02 * h * w:  # GrabCut fail, mask quá nhỏ -> fallback
        fallback = np.zeros((h, w), np.uint8)
        fallback[by : h - by, bx : w - bx] = 1
        return fallback
    return fg_mask


def add_synthetic_occluder(image_bgr: np.ndarray, amodal_mask: np.ndarray, rng: random.Random):
    """Vẽ 1 hình che giả (chữ nhật hoặc oval) đè lên một phần amodal_mask.
    Trả về (ảnh đã bị che, occluder_mask)."""
    h, w = amodal_mask.shape
    ys, xs = np.where(amodal_mask > 0)
    if len(xs) == 0:
        return image_bgr.copy(), np.zeros((h, w), np.uint8)

    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    obj_w, obj_h = x1 - x0, y1 - y0

    # Kích thước occluder: 25-45% kích thước vật thể
    occ_w = int(obj_w * rng.uniform(0.25, 0.45))
    occ_h = int(obj_h * rng.uniform(0.25, 0.45))
    occ_w, occ_h = max(occ_w, 10), max(occ_h, 10)

    # Vị trí: chồng lấn lên rìa vật thể (mô phỏng occlusion thật tự nhiên hơn
    # là che ngay giữa)
    cx = rng.randint(x0, x1)
    cy = rng.randint(y0, y1)

    occluder_mask = np.zeros((h, w), np.uint8)
    color = tuple(rng.randint(0, 255) for _ in range(3))

    out_img = image_bgr.copy()
    if rng.random() < 0.5:
        x_start, y_start = max(0, cx - occ_w // 2), max(0, cy - occ_h // 2)
        x_end, y_end = min(w, x_start + occ_w), min(h, y_start + occ_h)
        cv2.rectangle(out_img, (x_start, y_start), (x_end, y_end), color, -1)
        cv2.rectangle(occluder_mask, (x_start, y_start), (x_end, y_end), 1, -1)
    else:
        axes = (occ_w // 2, occ_h // 2)
        cv2.ellipse(out_img, (cx, cy), axes, 0, 0, 360, color, -1)
        cv2.ellipse(occluder_mask, (cx, cy), axes, 0, 0, 360, 1, -1)

    return out_img, occluder_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/raw")
    ap.add_argument("--out", default="data/cocoa")
    ap.add_argument("--n_occluders_per_image", type=int, default=2,
                     help="Sinh bao nhiêu phiên bản bị che khác nhau cho mỗi ảnh gốc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--download_samples", action="store_true",
                     help="Tải vài ảnh mẫu public-domain nếu --images đang trống")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if args.download_samples:
        download_samples(args.images)

    src_paths = sorted(p for p in glob(os.path.join(args.images, "*")) if p.lower().endswith(IMG_EXTS))
    if not src_paths:
        raise SystemExit(
            f"Không có ảnh nào trong {args.images}. Bỏ ảnh vào đó, hoặc chạy lại với --download_samples."
        )

    for sub in ("images", "visible_masks", "amodal_masks"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    n_written = 0
    for src_path in src_paths:
        image = cv2.imread(src_path)
        if image is None:
            print(f"[WARN] Không đọc được {src_path}, bỏ qua")
            continue

        # Resize về kích thước vừa phải để GrabCut nhanh hơn
        h, w = image.shape[:2]
        scale = 512 / max(h, w)
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        amodal_mask = estimate_object_mask(image)
        base_id = os.path.splitext(os.path.basename(src_path))[0]

        for k in range(args.n_occluders_per_image):
            occluded_img, occluder_mask = add_synthetic_occluder(image, amodal_mask, rng)
            visible_mask = np.logical_and(amodal_mask > 0, occluder_mask == 0).astype(np.uint8)

            image_id = f"{base_id}_{k}"
            cv2.imwrite(os.path.join(args.out, "images", f"{image_id}.jpg"), occluded_img)
            cv2.imwrite(os.path.join(args.out, "visible_masks", f"{image_id}.png"), visible_mask * 255)
            cv2.imwrite(os.path.join(args.out, "amodal_masks", f"{image_id}.png"), amodal_mask * 255)
            n_written += 1

    print(f"Đã tạo {n_written} mẫu synthetic occlusion trong {args.out}")
    print("Cấu trúc:")
    print(f"  {args.out}/images/<id>.jpg          <- ảnh input (đã bị che giả)")
    print(f"  {args.out}/visible_masks/<id>.png    <- vùng còn nhìn thấy")
    print(f"  {args.out}/amodal_masks/<id>.png     <- ground truth (đầy đủ vật thể)")
    print("\nDùng ngay được với: python scripts/04a_train_mask_model.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
