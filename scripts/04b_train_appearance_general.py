"""
Train LoRA "tổng quát" cho Amodal Appearance Completion -- KHÔNG phải
personalization kiểu DreamBooth, mà học QUY LUẬT CHUNG từ nhiều ảnh COCOA
khác nhau, mỗi ảnh có caption là tên class thật (đọc từ metadata.jsonl do
01d_prepare_appearance_from_cocoa.py tạo ra).

Usage:
    python scripts/04b_train_appearance_general.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
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
from transformers import CLIPTextModel, CLIPTokenizer


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class MetadataDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer, resolution: int):
        meta_path = os.path.join(data_dir, "metadata.jsonl")
        self.records = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        if not self.records:
            raise SystemExit(f"Không có record nào trong {meta_path}")

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.tf = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = self.tf(Image.open(os.path.join(self.data_dir, rec["file_name"])).convert("RGB"))
        ids = self.tokenizer(
            rec["text"], padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": img, "input_ids": ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    paths = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    output_dir = os.path.join(paths["output_dir"], "lora_general")
    os.makedirs(output_dir, exist_ok=True)

    set_seed(train_cfg.get("seed", 42))
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        mixed_precision=train_cfg.get("mixed_precision", "fp16"),
    )

    base_model = model_cfg["base_model"]
    print(f"Tải base model: {base_model}")
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    print(f"LoRA rank={train_cfg['lora_rank']}")
    lora_config = LoraConfig(
        r=train_cfg["lora_rank"], lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    trainable_params = [p for p in unet.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=float(train_cfg["learning_rate"]))

    train_dataset = MetadataDataset(paths["train_ready"], tokenizer, model_cfg["resolution"])
    print(f"Số ảnh training: {len(train_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg["train_batch_size"], shuffle=True)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_cfg["gradient_accumulation_steps"])
    max_train_steps = train_cfg.get("max_train_steps") or (train_cfg["num_train_epochs"] * num_update_steps_per_epoch)

    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
    vae.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    print(f"Bắt đầu train: {max_train_steps} steps")
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    unet.train()
    while global_step < max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
                input_ids = batch["input_ids"].to(accelerator.device)

                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(input_ids)[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

                if global_step % train_cfg["checkpointing_steps"] == 0:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.unwrap_model(unet).save_pretrained(save_path)

            if global_step >= max_train_steps:
                break

    final_path = os.path.join(output_dir, "final")
    accelerator.unwrap_model(unet).save_pretrained(final_path)
    print(f"Hoàn tất. LoRA (tổng quát, đa class) lưu ở: {final_path}")
    print(f"Loss cuối: {loss.item():.4f}")


if __name__ == "__main__":
    main()
