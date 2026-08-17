"""
Fine-tune Mask branch: dự đoán amodal mask từ (ảnh, visible mask).

Dữ liệu: data/cocoa/{images, visible_masks, amodal_masks}/<id>.{jpg,png}
(sinh ra từ scripts/01_prepare_cocoa.py). Nếu bạn tự có thêm ảnh với mask
riêng, bỏ vào cùng cấu trúc thư mục là dùng chung được luôn.

Đây là bài toán supervised nhỏ, KHÔNG cần SAM/FLUX/GPT-4o — chỉ cần
COCOA (ground truth) để học cách "đoán phần bị che" dựa trên hình dạng/
ngữ cảnh của phần nhìn thấy.

Usage:
    python scripts/04a_train_mask_model.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.amodal_pipeline.mask_model import AmodalMaskUNet, dice_loss


class CocoaMaskDataset(Dataset):
    def __init__(self, cocoa_dir: str, resolution: int = 256):
        self.image_paths = sorted(glob.glob(os.path.join(cocoa_dir, "images", "*.jpg")))
        self.cocoa_dir = cocoa_dir
        self.resolution = resolution
        self.img_tf = transforms.Compose(
            [transforms.Resize((resolution, resolution)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.image_paths)

    def _load_mask(self, path: str) -> torch.Tensor:
        m = Image.open(path).convert("L").resize((self.resolution, self.resolution), Image.NEAREST)
        return torch.from_numpy((np.array(m) > 127).astype(np.float32)).unsqueeze(0)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        vis_path = os.path.join(self.cocoa_dir, "visible_masks", f"{image_id}.png")
        amo_path = os.path.join(self.cocoa_dir, "amodal_masks", f"{image_id}.png")

        image = self.img_tf(Image.open(img_path).convert("RGB"))
        visible_mask = self._load_mask(vis_path)
        amodal_mask = self._load_mask(amo_path)
        return image, visible_mask, amodal_mask


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị: {device}")

    dataset = CocoaMaskDataset(cfg["paths"]["cocoa_dir"], resolution=args.resolution)
    if len(dataset) == 0:
        raise SystemExit(
            f"Không có dữ liệu trong {cfg['paths']['cocoa_dir']}. "
            "Chạy scripts/01_prepare_cocoa.py trước."
        )
    print(f"Số ảnh training: {len(dataset)}")

    n_val = max(1, int(0.1 * len(dataset)))
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = AmodalMaskUNet(base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = os.path.join(cfg["paths"]["output_dir"], "mask_model")
    os.makedirs(out_dir, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for image, vis_mask, amo_mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            image, vis_mask, amo_mask = image.to(device), vis_mask.to(device), amo_mask.to(device)

            logits = model(image, vis_mask)
            loss = F.binary_cross_entropy_with_logits(logits, amo_mask) + dice_loss(logits, amo_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, vis_mask, amo_mask in val_loader:
                image, vis_mask, amo_mask = image.to(device), vis_mask.to(device), amo_mask.to(device)
                logits = model(image, vis_mask)
                loss = F.binary_cross_entropy_with_logits(logits, amo_mask) + dice_loss(logits, amo_mask)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    torch.save(model.state_dict(), os.path.join(out_dir, "final.pt"))
    print(f"Xong. Checkpoint: {out_dir}/best.pt (val_loss thấp nhất) và final.pt")


if __name__ == "__main__":
    main()
