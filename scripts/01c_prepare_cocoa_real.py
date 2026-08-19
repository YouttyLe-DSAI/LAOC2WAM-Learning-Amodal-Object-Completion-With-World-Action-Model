from __future__ import annotations
import argparse, json, os, re
import cv2, numpy as np, requests
from PIL import Image

def get_image_url(file_name):
    m = re.search(r"(train2014|val2014)", file_name)
    split = m.group(1) if m else "val2014"
    return f"http://images.cocodataset.org/{split}/{file_name}"

def polygon_to_mask(seg_flat, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = seg_flat if (seg_flat and isinstance(seg_flat[0], list)) else [seg_flat]
    for poly in polygons:
        if len(poly) < 6: continue
        pts = np.array(poly, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask

def download_image(file_name, dest_dir):
    dest_path = os.path.join(dest_dir, file_name)
    if os.path.exists(dest_path): return dest_path
    try:
        r = requests.get(get_image_url(file_name), timeout=20)
        r.raise_for_status()
        with open(dest_path, "wb") as f: f.write(r.content)
        return dest_path
    except Exception as e:
        print(f"[WARN] {e}"); return None

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

    with open(args.annotation_json) as f: data = json.load(f)
    entries = data["annotations"][: args.n_images]
    print(f"Xu ly {len(entries)} anh...")

    n_instances = 0
    for img_idx, entry in enumerate(entries):
        file_name = entry["url"].split("/")[-1]
        regions = entry.get("regions", [])
        if not regions: continue
        img_path = download_image(file_name, raw_dir)
        if img_path is None: continue
        try:
            with Image.open(img_path) as im:
                width, height = im.size
                im.convert("RGB").save(img_path)
        except Exception as e:
            print(f"[WARN] {e}"); continue

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
            print(f"  {img_idx+1}/{len(entries)} anh, {n_instances} instance")

    print(f"\nHoan tat. Tong {n_instances} instance.")

if __name__ == "__main__":
    main()
