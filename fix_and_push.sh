#!/bin/bash
set -e
echo '=== Ghi de 5 file da fix ==='

cat > scripts/01c_prepare_cocoa_real.py << 'PYEOF_INNER'
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
PYEOF_INNER
echo 'Da ghi: scripts/01c_prepare_cocoa_real.py'

cat > scripts/02b_prepare_dreambooth_data.py << 'PYEOF_INNER'
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
PYEOF_INNER
echo 'Da ghi: scripts/02b_prepare_dreambooth_data.py'

cat > scripts/04c_train_appearance_conditioned.py << 'PYEOF_INNER'
"""
Fine-tune Stable Diffusion UNet nhận ĐÚNG 2 input theo slide:
  - Object's mask (amodal mask, nối thêm kênh vào input UNet)
  - Object's appearance feature vector (CLIP image embedding, tiêm qua
    cross-attention THAY cho text embedding)
Output: amodal RGB.

Training kiểu SELF-SUPERVISED (giống PCNet/SeGAN/SynergyAmodal): lấy ảnh
vật thể KHÔNG bị che từ COCOA làm "đáp án đúng", TỰ che 1 phần đi, trích
feature vector từ phần còn thấy, dạy model tái tạo lại đúng ảnh gốc.

Usage:
    python scripts/04c_train_appearance_conditioned.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor

IMG_EXTS = (".jpg", ".jpeg", ".png")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def estimate_object_mask(image_bgr: np.ndarray, border_ratio: float = 0.06) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    bx, by = int(w * border_ratio), int(h * border_ratio)
    rect = (bx, by, w - 2 * bx, h - 2 * by)
    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        fb = np.zeros((h, w), np.uint8)
        fb[by:h - by, bx:w - bx] = 1
        return fb
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    if fg.sum() < 0.02 * h * w:
        fb = np.zeros((h, w), np.uint8)
        fb[by:h - by, bx:w - bx] = 1
        return fb
    return fg


def add_synthetic_occluder(image_bgr: np.ndarray, amodal_mask: np.ndarray, rng: random.Random):
    h, w = amodal_mask.shape
    ys, xs = np.where(amodal_mask > 0)
    if len(xs) == 0:
        return image_bgr.copy(), np.zeros((h, w), np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    obj_w, obj_h = x1 - x0, y1 - y0
    occ_w = max(int(obj_w * rng.uniform(0.25, 0.45)), 10)
    occ_h = max(int(obj_h * rng.uniform(0.25, 0.45)), 10)
    cx, cy = rng.randint(x0, x1), rng.randint(y0, y1)
    occluder_mask = np.zeros((h, w), np.uint8)
    color = tuple(rng.randint(0, 255) for _ in range(3))
    out = image_bgr.copy()
    axes_e = (occ_w // 2, occ_h // 2)
    cv2.ellipse(out, (cx, cy), axes_e, 0, 0, 360, color, -1)
    cv2.ellipse(occluder_mask, (cx, cy), axes_e, 0, 0, 360, 1, -1)
    return out, occluder_mask


class SelfSupervisedOcclusionDataset(Dataset):
    def __init__(self, image_dir: str, resolution: int, clip_processor, seed: int = 42):
        self.paths = sorted(
            p for p in glob.glob(os.path.join(image_dir, "*")) if p.lower().endswith(IMG_EXTS)
        )
        if not self.paths:
            raise SystemExit(f"Không có ảnh trong {image_dir}")
        self.resolution = resolution
        self.latent_size = resolution // 8
        self.clip_processor = clip_processor
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.paths[idx])
        img_bgr = cv2.resize(img_bgr, (self.resolution, self.resolution))

        amodal_mask = estimate_object_mask(img_bgr)
        occluded_bgr, _ = add_synthetic_occluder(img_bgr, amodal_mask, self.rng)

        gt_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        occluded_rgb = cv2.cvtColor(occluded_bgr, cv2.COLOR_BGR2RGB)

        gt_pixel = self.tf(Image.fromarray(gt_rgb))

        clip_inputs = self.clip_processor(images=Image.fromarray(occluded_rgb), return_tensors="pt")
        clip_pixel = clip_inputs["pixel_values"][0]

        mask_small = cv2.resize(amodal_mask.astype(np.float32), (self.latent_size, self.latent_size))
        mask_latent = torch.from_numpy(mask_small).unsqueeze(0)

        return {"gt_pixel": gt_pixel, "clip_pixel": clip_pixel, "mask_latent": mask_latent}


class FeatureProjector(nn.Module):
    def __init__(self, clip_dim: int, cross_attention_dim: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = nn.Linear(clip_dim, cross_attention_dim * num_tokens)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, clip_embed: torch.Tensor) -> torch.Tensor:
        x = self.proj(clip_embed)
        x = x.view(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


def expand_conv_in(unet: UNet2DConditionModel, extra_channels: int = 1) -> nn.Conv2d:
    old_conv = unet.conv_in
    new_conv = nn.Conv2d(
        old_conv.in_channels + extra_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size, padding=old_conv.padding,
    )
    with torch.no_grad():
        new_conv.weight[:, : old_conv.in_channels] = old_conv.weight
        new_conv.weight[:, old_conv.in_channels:] = 0.0
        new_conv.bias[:] = old_conv.bias
    return new_conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--num_feature_tokens", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)

    paths = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    output_dir = os.path.join(paths["output_dir"], "appearance_conditioned")
    os.makedirs(output_dir, exist_ok=True)

    set_seed(train_cfg.get("seed", 42))
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        mixed_precision=train_cfg.get("mixed_precision", "fp16"),
    )
    device = accelerator.device

    base_model = model_cfg["base_model"]
    print(f"Tải base model: {base_model}")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    print(f"Tải CLIP image encoder: {args.clip_model}")
    clip_vision = CLIPVisionModel.from_pretrained(args.clip_model)
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model)
    clip_dim = clip_vision.config.hidden_size
    cross_attention_dim = unet.config.cross_attention_dim

    vae.requires_grad_(False)
    clip_vision.requires_grad_(False)

    unet.requires_grad_(False)
    new_conv_in = expand_conv_in(unet, extra_channels=1)
    unet.conv_in = new_conv_in
    unet.conv_in.requires_grad_(True)

    lora_config = LoraConfig(
        r=train_cfg["lora_rank"], lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        modules_to_save=["conv_in"],  # QUAN TRỌNG: đảm bảo conv_in (5 kênh, có train)
                                       # được lưu/load đầy đủ cùng checkpoint, không chỉ LoRA delta
    )
    unet = get_peft_model(unet, lora_config)

    feature_projector = FeatureProjector(clip_dim, cross_attention_dim, args.num_feature_tokens)

    trainable_params = (
        [p for p in unet.parameters() if p.requires_grad] + list(feature_projector.parameters())
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Số tham số trainable: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=float(train_cfg["learning_rate"]))

    dataset = SelfSupervisedOcclusionDataset(paths["train_ready"], model_cfg["resolution"], clip_processor)
    print(f"Số ảnh training (self-supervised occlusion): {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=train_cfg["train_batch_size"], shuffle=True)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / train_cfg["gradient_accumulation_steps"])
    max_train_steps = train_cfg.get("max_train_steps") or (train_cfg["num_train_epochs"] * num_update_steps_per_epoch)

    unet, feature_projector, optimizer, dataloader = accelerator.prepare(
        unet, feature_projector, optimizer, dataloader
    )
    vae.to(device, dtype=torch.float16)
    clip_vision.to(device, dtype=torch.float16)

    print(f"Bắt đầu train: {max_train_steps} steps (mask + feature-vector conditioning)")
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    unet.train()
    feature_projector.train()
    while global_step < max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                gt_pixel = batch["gt_pixel"].to(device, dtype=torch.float16)
                clip_pixel = batch["clip_pixel"].to(device, dtype=torch.float16)
                mask_latent = batch["mask_latent"].to(device, dtype=torch.float16)

                with torch.no_grad():
                    latents = vae.encode(gt_pixel).latent_dist.sample() * vae.config.scaling_factor
                    clip_out = clip_vision(pixel_values=clip_pixel).pooler_output

                encoder_hidden_states = feature_projector(clip_out)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                unet_input = torch.cat([noisy_latents, mask_latent], dim=1)

                model_pred = unet(unet_input, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

                if global_step % train_cfg["checkpointing_steps"] == 0:
                    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    accelerator.unwrap_model(unet).save_pretrained(ckpt_dir)
                    torch.save(accelerator.unwrap_model(feature_projector).state_dict(),
                               os.path.join(ckpt_dir, "feature_projector.pt"))

            if global_step >= max_train_steps:
                break

    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    accelerator.unwrap_model(unet).save_pretrained(final_dir)
    torch.save(accelerator.unwrap_model(feature_projector).state_dict(),
               os.path.join(final_dir, "feature_projector.pt"))

    print(f"\nHoàn tất. Model (mask + feature-vector conditioned) lưu ở: {final_dir}")
    print(f"Loss cuối: {loss.item():.4f}")


if __name__ == "__main__":
    main()
PYEOF_INNER
echo 'Da ghi: scripts/04c_train_appearance_conditioned.py'

cat > scripts/07_infer_appearance_conditioned.py << 'PYEOF_INNER'
"""
Inference cho model Appearance Conditioned (mask + feature vector).

Usage:
    python scripts/07_infer_appearance_conditioned.py --config configs/config.yaml \
        --checkpoint outputs/appearance_conditioned/final \
        --n_samples 5
"""
from __future__ import annotations

import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from peft import PeftModel
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPModel, CLIPProcessor

IMG_EXTS = (".jpg", ".jpeg", ".png")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def estimate_object_mask(image_bgr: np.ndarray, border_ratio: float = 0.06) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    bx, by = int(w * border_ratio), int(h * border_ratio)
    rect = (bx, by, w - 2 * bx, h - 2 * by)
    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        fb = np.zeros((h, w), np.uint8)
        fb[by:h - by, bx:w - bx] = 1
        return fb
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    if fg.sum() < 0.02 * h * w:
        fb = np.zeros((h, w), np.uint8)
        fb[by:h - by, bx:w - bx] = 1
        return fb
    return fg


def add_synthetic_occluder(image_bgr, amodal_mask, rng):
    h, w = amodal_mask.shape
    ys, xs = np.where(amodal_mask > 0)
    if len(xs) == 0:
        return image_bgr.copy(), np.zeros((h, w), np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    obj_w, obj_h = x1 - x0, y1 - y0
    occ_w = max(int(obj_w * rng.uniform(0.25, 0.45)), 10)
    occ_h = max(int(obj_h * rng.uniform(0.25, 0.45)), 10)
    cx, cy = rng.randint(x0, x1), rng.randint(y0, y1)
    occluder_mask = np.zeros((h, w), np.uint8)
    color = tuple(rng.randint(0, 255) for _ in range(3))
    out = image_bgr.copy()
    axes_e = (occ_w // 2, occ_h // 2)
    cv2.ellipse(out, (cx, cy), axes_e, 0, 0, 360, color, -1)
    cv2.ellipse(occluder_mask, (cx, cy), axes_e, 0, 0, 360, 1, -1)
    return out, occluder_mask


class FeatureProjector(nn.Module):
    def __init__(self, clip_dim, cross_attention_dim, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = nn.Linear(clip_dim, cross_attention_dim * num_tokens)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, clip_embed):
        x = self.proj(clip_embed)
        x = x.view(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


def expand_conv_in(unet, extra_channels=1):
    old_conv = unet.conv_in
    new_conv = nn.Conv2d(old_conv.in_channels + extra_channels, old_conv.out_channels,
                          kernel_size=old_conv.kernel_size, padding=old_conv.padding)
    return new_conv


@torch.no_grad()
def run_inference(unet, feature_projector, vae, clip_vision, clip_processor,
                   noise_scheduler, occluded_rgb, amodal_mask, resolution, device,
                   num_inference_steps=30):
    latent_size = resolution // 8

    clip_inputs = clip_processor(images=Image.fromarray(occluded_rgb), return_tensors="pt")
    clip_pixel = clip_inputs["pixel_values"].to(device, dtype=torch.float16)
    clip_feat = clip_vision(pixel_values=clip_pixel).pooler_output
    encoder_hidden_states = feature_projector(clip_feat)

    mask_small = cv2.resize(amodal_mask.astype(np.float32), (latent_size, latent_size))
    mask_latent = torch.from_numpy(mask_small).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)

    latents = torch.randn((1, 4, latent_size, latent_size), device=device, dtype=torch.float16)
    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    for t in noise_scheduler.timesteps:
        unet_input = torch.cat([latents, mask_latent], dim=1)
        noise_pred = unet(unet_input, t, encoder_hidden_states).sample
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].permute(1, 2, 0).float().cpu().numpy()
    image = (image * 255).round().astype(np.uint8)
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default="outputs/appearance_conditioned/final")
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--out_dir", default="outputs/appearance_conditioned/eval_samples")
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolution = cfg["model"]["resolution"]
    base_model = cfg["model"]["base_model"]
    test_dir = args.test_dir or cfg["paths"]["train_ready"]

    os.makedirs(args.out_dir, exist_ok=True)

    print("Tải VAE + scheduler...")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device, dtype=torch.float16)
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    print("Tải UNet base + mở rộng conv_in (5 kênh, khớp shape checkpoint)...")
    base_unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    base_unet.conv_in = expand_conv_in(base_unet, extra_channels=1)

    print(f"Nạp checkpoint (LoRA + conv_in đã train): {args.checkpoint}")
    unet = PeftModel.from_pretrained(base_unet, args.checkpoint)
    unet = unet.to(device, dtype=torch.float16)
    unet.eval()

    print(f"Tải CLIP: {args.clip_model}")
    clip_vision = CLIPVisionModel.from_pretrained(args.clip_model).to(device, dtype=torch.float16)
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model)
    clip_dim = clip_vision.config.hidden_size
    cross_attention_dim = base_unet.config.cross_attention_dim

    feature_projector = FeatureProjector(clip_dim, cross_attention_dim).to(device, dtype=torch.float16)
    fp_path = os.path.join(args.checkpoint, "feature_projector.pt")
    feature_projector.load_state_dict(torch.load(fp_path, map_location=device))
    feature_projector.eval()

    eval_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    eval_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    paths = sorted(p for p in glob.glob(os.path.join(test_dir, "*")) if p.lower().endswith(IMG_EXTS))
    rng = random.Random(123)
    sample_paths = rng.sample(paths, min(args.n_samples, len(paths)))

    ssim_scores, clip_scores = [], []
    for i, path in enumerate(sample_paths):
        img_bgr = cv2.resize(cv2.imread(path), (resolution, resolution))
        amodal_mask = estimate_object_mask(img_bgr)
        occluded_bgr, _ = add_synthetic_occluder(img_bgr, amodal_mask, rng)
        occluded_rgb = cv2.cvtColor(occluded_bgr, cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        result = run_inference(unet, feature_projector, vae, clip_vision, clip_processor,
                                noise_scheduler, occluded_rgb, amodal_mask, resolution, device)

        out_path = os.path.join(args.out_dir, f"sample_{i}.png")
        combined = np.concatenate([occluded_rgb, result, gt_rgb], axis=1)
        Image.fromarray(combined).save(out_path)

        s = ssim(cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2GRAY), cv2.cvtColor(result, cv2.COLOR_RGB2GRAY))
        ssim_scores.append(s)

        clip_in = eval_clip_processor(images=[Image.fromarray(result), Image.fromarray(gt_rgb)],
                                       return_tensors="pt").to(device)
        with torch.no_grad():
            emb = eval_clip_model.get_image_features(**clip_in)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        clip_scores.append((emb[0] @ emb[1]).item())

        print(f"[{i+1}/{len(sample_paths)}] {os.path.basename(path)}: SSIM={s:.4f}  CLIP_sim={clip_scores[-1]:.4f}")

    print(f"\n=== KẾT QUẢ TRUNG BÌNH ({len(sample_paths)} mẫu) ===")
    print(f"SSIM trung bình: {np.mean(ssim_scores):.4f}")
    print(f"CLIP similarity trung bình: {np.mean(clip_scores):.4f}")
    print(f"\nẢnh minh hoạ (input | sinh ra | ground truth) lưu ở: {args.out_dir}/sample_*.png")


if __name__ == "__main__":
    main()
PYEOF_INNER
echo 'Da ghi: scripts/07_infer_appearance_conditioned.py'

cat > scripts/08_combined_pipeline.py << 'PYEOF_INNER'
"""
Kết hợp THẬT 2 nhánh (không phải minh hoạ giả):
  1. Mask model (train trên COCOA)         -> dự đoán amodal mask
  2. Appearance model (personalized, train
     trên vài ảnh của 1 vật cụ thể)         -> nhận mask dự đoán + feature
     vector (trích từ ảnh input) -> sinh ảnh hoàn chỉnh

Usage:
    python scripts/08_combined_pipeline.py --config configs/config.yaml \
        --mask_checkpoint outputs/mask_model/best.pt \
        --appearance_checkpoint outputs/appearance_conditioned/final \
        --test_image data/train_ready_backpack/instance_images/backpack_0.jpg
"""
from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from peft import PeftModel
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModel, CLIPImageProcessor

from src.amodal_pipeline.mask_model import AmodalMaskUNet


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def estimate_object_mask(image_bgr: np.ndarray, border_ratio: float = 0.06) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    bx, by = int(w * border_ratio), int(h * border_ratio)
    rect = (bx, by, w - 2 * bx, h - 2 * by)
    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        fb = np.zeros((h, w), np.uint8)
        fb[by:h - by, bx:w - bx] = 1
        return fb
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    if fg.sum() < 0.02 * h * w:
        fb = np.zeros((h, w), np.uint8)
        fb[by:h - by, bx:w - bx] = 1
        return fb
    return fg


def add_synthetic_occluder(image_bgr, amodal_mask, rng):
    h, w = amodal_mask.shape
    ys, xs = np.where(amodal_mask > 0)
    if len(xs) == 0:
        return image_bgr.copy(), np.zeros((h, w), np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    obj_w, obj_h = x1 - x0, y1 - y0
    occ_w = max(int(obj_w * 0.35), 10)
    occ_h = max(int(obj_h * 0.35), 10)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    occluder_mask = np.zeros((h, w), np.uint8)
    out = image_bgr.copy()
    axes_e = (occ_w // 2, occ_h // 2)
    cv2.ellipse(out, (cx, cy), axes_e, 0, 0, 360, (40, 40, 40), -1)
    cv2.ellipse(occluder_mask, (cx, cy), axes_e, 0, 0, 360, 1, -1)
    return out, occluder_mask


class FeatureProjector(nn.Module):
    def __init__(self, clip_dim, cross_attention_dim, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = nn.Linear(clip_dim, cross_attention_dim * num_tokens)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, clip_embed):
        x = self.proj(clip_embed)
        x = x.view(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


def expand_conv_in(unet, extra_channels=1):
    old_conv = unet.conv_in
    return nn.Conv2d(old_conv.in_channels + extra_channels, old_conv.out_channels,
                      kernel_size=old_conv.kernel_size, padding=old_conv.padding)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--mask_checkpoint", default="outputs/mask_model/best.pt")
    ap.add_argument("--appearance_checkpoint", default="outputs/appearance_conditioned/final")
    ap.add_argument("--test_image", required=True)
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--out", default="outputs/combined_result.png")
    ap.add_argument("--out_dir", default="outputs/combined_stages")
    ap.add_argument("--num_inference_steps", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolution = cfg["model"]["resolution"]
    base_model = cfg["model"]["base_model"]
    latent_size = resolution // 8

    img_bgr = cv2.resize(cv2.imread(args.test_image), (resolution, resolution))
    true_amodal = estimate_object_mask(img_bgr)
    rng = random.Random(7)
    occluded_bgr, occluder_mask = add_synthetic_occluder(img_bgr, true_amodal, rng)
    occluded_rgb = cv2.cvtColor(occluded_bgr, cv2.COLOR_BGR2RGB)
    gt_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    visible_mask = np.logical_and(true_amodal > 0, occluder_mask == 0).astype(np.float32)

    print("Nạp Mask model (COCOA)...")
    mask_model = AmodalMaskUNet(base_ch=32).to(device)
    mask_model.load_state_dict(torch.load(args.mask_checkpoint, map_location=device))
    mask_model.eval()

    img_t = transforms.ToTensor()(Image.fromarray(occluded_rgb)).unsqueeze(0).to(device)
    vis_t = torch.from_numpy(visible_mask).unsqueeze(0).unsqueeze(0).to(device)
    pred_mask_logits = mask_model(img_t, vis_t)
    pred_mask = (torch.sigmoid(pred_mask_logits)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
    print(f"Mask dự đoán: {pred_mask.sum()} pixel (so với ground truth {true_amodal.sum()} pixel)")

    print("Nạp Appearance model (personalized)...")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device, dtype=torch.float16)
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    base_unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    base_unet.conv_in = expand_conv_in(base_unet, extra_channels=1)
    unet = PeftModel.from_pretrained(base_unet, args.appearance_checkpoint).to(device, dtype=torch.float16)
    unet.eval()

    clip_vision = CLIPVisionModel.from_pretrained(args.clip_model).to(device, dtype=torch.float16)
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model)
    clip_dim = clip_vision.config.hidden_size
    cross_attention_dim = base_unet.config.cross_attention_dim

    feature_projector = FeatureProjector(clip_dim, cross_attention_dim).to(device, dtype=torch.float16)
    feature_projector.load_state_dict(
        torch.load(os.path.join(args.appearance_checkpoint, "feature_projector.pt"), map_location=device)
    )
    feature_projector.eval()

    clip_inputs = clip_processor(images=Image.fromarray(occluded_rgb), return_tensors="pt")
    clip_pixel = clip_inputs["pixel_values"].to(device, dtype=torch.float16)
    clip_feat = clip_vision(pixel_values=clip_pixel).pooler_output
    encoder_hidden_states = feature_projector(clip_feat)

    # Mask dùng để sinh: vùng model được TỰ DO vẽ (occluder_mask thật, không
    # phải amodal mask đầy đủ -- vì vùng visible KHÔNG cần sinh, chỉ cần giữ
    # nguyên từ ảnh input).
    gen_region = occluder_mask.astype(np.float32)  # 1 = model tự do sinh, 0 = giữ nguyên ảnh gốc
    mask_small = cv2.resize(pred_mask.astype(np.float32), (latent_size, latent_size))
    mask_latent = torch.from_numpy(mask_small).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)
    gen_region_small = cv2.resize(gen_region, (latent_size, latent_size))
    gen_mask_latent = torch.from_numpy(gen_region_small).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)
    gen_mask_latent = (gen_mask_latent > 0.5).float()

    # Encode ẢNH INPUT THẬT (bị che) -- vùng ngoài occluder trong latent này
    # là "đáp án đúng" cần giữ nguyên xuyên suốt quá trình khử nhiễu.
    known_pixel = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])(
        Image.fromarray(occluded_rgb)
    ).unsqueeze(0).to(device, dtype=torch.float16)
    known_latents = vae.encode(known_pixel).latent_dist.sample() * vae.config.scaling_factor

    latents = torch.randn((1, 4, latent_size, latent_size), device=device, dtype=torch.float16)
    noise_scheduler.set_timesteps(args.num_inference_steps, device=device)
    for t in noise_scheduler.timesteps:
        # RÀNG BUỘC (RePaint-style): vùng ngoài occluder = ép về đúng ảnh
        # gốc (đã thêm nhiễu tương ứng mức t); vùng occluder = giữ nguyên
        # kết quả model đang khử nhiễu (được tự do sinh).
        step_noise = torch.randn_like(known_latents)
        noisy_known = noise_scheduler.add_noise(known_latents, step_noise, t.unsqueeze(0))
        latents = gen_mask_latent * latents + (1 - gen_mask_latent) * noisy_known

        unet_input = torch.cat([latents, mask_latent], dim=1)
        noise_pred = unet(unet_input, t, encoder_hidden_states).sample
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    # Ép buộc cuối cùng (t=0, không nhiễu): đảm bảo vùng ngoài occluder
    # khớp CHÍNH XÁC pixel gốc, không chỉ gần đúng.
    latents = gen_mask_latent * latents + (1 - gen_mask_latent) * known_latents

    latents = latents / vae.config.scaling_factor
    decoded = vae.decode(latents).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    result = (decoded[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)

    # ---------- Xuất riêng từng giai đoạn (rõ ràng cho báo cáo) ----------
    stage_dir = args.out_dir
    os.makedirs(stage_dir, exist_ok=True)

    # 1. Ảnh input bị che
    Image.fromarray(occluded_rgb).save(os.path.join(stage_dir, "01_input_occluded.png"))

    # 2. Mask trắng đen (mask model dự đoán)
    mask_bw = (pred_mask * 255).astype(np.uint8)
    Image.fromarray(mask_bw, mode="L").save(os.path.join(stage_dir, "02_mask_bw.png"))

    # 3. Vật thể được tách riêng (cutout): áp mask lên ảnh gốc, nền = trắng
    cutout = np.full_like(gt_rgb, 255)
    full_mask_bool = pred_mask.astype(bool)
    cutout[full_mask_bool] = gt_rgb[full_mask_bool]
    Image.fromarray(cutout).save(os.path.join(stage_dir, "03_object_cutout.png"))

    # 4. Kết quả sau khi vá (đã ràng buộc giữ nguyên vùng thấy)
    Image.fromarray(result).save(os.path.join(stage_dir, "04_repainted_result.png"))

    # 5. Ground truth (ảnh gốc thật, không bị che, để đối chiếu)
    Image.fromarray(gt_rgb).save(os.path.join(stage_dir, "05_ground_truth.png"))

    # Ảnh ghép tổng hợp (xem nhanh 1 lần)
    pred_mask_vis = np.stack([mask_bw] * 3, axis=-1)
    combined = np.concatenate([occluded_rgb, pred_mask_vis, cutout, result, gt_rgb], axis=1)
    Image.fromarray(combined).save(args.out)

    print(f"\nHoàn tất pipeline (RePaint-constrained).")
    print(f"Các ảnh giai đoạn riêng biệt lưu tại: {stage_dir}/")
    print(f"  01_input_occluded.png    -- ảnh bị che (input)")
    print(f"  02_mask_bw.png           -- mask trắng đen (Mask model dự đoán)")
    print(f"  03_object_cutout.png     -- vật thể tách riêng (nền trắng)")
    print(f"  04_repainted_result.png  -- kết quả sau khi vá")
    print(f"  05_ground_truth.png      -- ảnh gốc thật để đối chiếu")
    print(f"Ảnh ghép tổng hợp: {args.out}")


if __name__ == "__main__":
    main()
PYEOF_INNER
echo 'Da ghi: scripts/08_combined_pipeline.py'

echo '=== Kiem tra syntax ==='
python3 -m py_compile scripts/01c_prepare_cocoa_real.py scripts/02b_prepare_dreambooth_data.py scripts/04c_train_appearance_conditioned.py scripts/07_infer_appearance_conditioned.py scripts/08_combined_pipeline.py && echo 'Tat ca syntax OK'

echo '=== Kiem tra cac fix quan trong ==='
grep -q 'train2014|val2014' scripts/01c_prepare_cocoa_real.py && echo 'OK: 01c fix train2014/val2014'
grep -q 're.split(r"\[,:' scripts/02b_prepare_dreambooth_data.py && echo 'OK: 02b fix parser dau phay'
grep -q 'modules_to_save' scripts/04c_train_appearance_conditioned.py && echo 'OK: 04c fix modules_to_save'
grep -q 'gen_mask_latent' scripts/08_combined_pipeline.py && echo 'OK: 08 fix RePaint'

echo '=== Git add + commit + push ==='
git add -A
git commit -m 'fix: V2 - all known bugs fixed (URL, parser, modules_to_save, RePaint)'
git push origin main
echo '=== XONG - kiem tra ket qua o tren ==='
