"""
Tiện ích xử lý mask cho pipeline amodal.

Các mask trong pipeline này (xem README):
  - M_visible : vùng nhìn thấy được của object (từ segmentation agent)
  - M_occ_i   : mask từng occluder
  - M_bdy     : vùng mở rộng biên ảnh (boundary truncation)
  - M_inpaint : union(dilate(M_occ_i)) U M_bdy  -> vùng cần sinh ảnh
  - M_alpha   : mask cuối cùng của object hoàn chỉnh (visible + synthesized)
"""
from __future__ import annotations

import numpy as np
import cv2
from pycocotools import mask as coco_mask


def rle_to_mask(rle, height: int, width: int) -> np.ndarray:
    """Giải mã RLE (định dạng COCO/COCOA) thành mask nhị phân HxW."""
    if isinstance(rle, list):
        rles = coco_mask.frPyObjects(rle, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(rle["counts"], list):
        rle = coco_mask.frPyObjects(rle, height, width)
    m = coco_mask.decode(rle)
    return m.astype(np.uint8)


def dilate_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Giãn nở mask (dùng cho occluder mask trước khi union vào M_inpaint,
    để tránh viền hở khi inpaint)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def union_masks(*masks: np.ndarray) -> np.ndarray:
    """Hợp nhiều mask nhị phân lại (M_inpaint = union occluder masks + boundary mask)."""
    out = np.zeros_like(masks[0])
    for m in masks:
        out = np.logical_or(out, m > 0).astype(np.uint8)
    return out


def bbox_from_mask(mask: np.ndarray):
    """Trả về (x_min, y_min, x_max, y_max) của mask, hoặc None nếu mask rỗng."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def edge_touch(mask: np.ndarray, tol: int = 2):
    """Kiểm tra bbox của mask có chạm biên ảnh không -> geometric prior cho
    Boundary Analysis Agent. Trả về list các cạnh chạm: subset của
    ['left','right','top','bottom']."""
    h, w = mask.shape[:2]
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return []
    x0, y0, x1, y1 = bbox
    touched = []
    if x0 <= tol:
        touched.append("left")
    if x1 >= w - 1 - tol:
        touched.append("right")
    if y0 <= tol:
        touched.append("top")
    if y1 >= h - 1 - tol:
        touched.append("bottom")
    return touched


def save_mask_png(mask: np.ndarray, path: str) -> None:
    cv2.imwrite(path, (mask > 0).astype(np.uint8) * 255)


def load_mask_png(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (m > 127).astype(np.uint8)
