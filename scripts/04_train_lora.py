"""
Fine-tune LoRA tren UNet cua Stable Diffusion theo DUNG cong thuc DreamBooth
goc, gom du 2 nhanh loss:
  1. Reconstruction Loss (instance)
  2. Class-Specific Prior Preservation Loss (class)

L_total = L_instance + prior_loss_weight * L_class
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

IMG_EXTS = (".jpg", ".jpeg", ".png")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_class_images(base_model, class_prompt, class_dir, num_needed, resolution, device):
    os.makedirs(class_dir, exist_ok=True)
    existing = [p for p in glob.glob(os.path.join(class_dir, "*")) if p.lower().endswith(IMG_EXTS)]
    n_missing = num_needed - len(existing)
    if n_missing <= 0:
        print(f"Da co du {len(existing)} class images trong {class_dir}, bo qua buoc sinh.")
        return

    print(f"Sinh them {n_missing} class images (\"{class_prompt}\")...")
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16).to(device)
    pipe.set_progress_bar_config(disable=True)

    start_idx = len(existing)
    batch_size = 4
    n_done = 0
    with torch.no_grad():
        while n_done < n_missing:
            n_this_batch = min(batch_size, n_missing - n_done)
            images = pipe(
                [class_prompt] * n_this_batch,
                height=resolution, width=resolution, num_inference_steps=25,
            ).images
            for i, img in enumerate(images):
                img.save(os.path.join(class_dir, f"class_{start_idx + n_done + i}.jpg"), quality=95)
            n_done += n_this_batch

    del pipe
    torch.cuda.empty_cache()
    print(f"Da sinh xong. Tong so class images: {num_needed}")


class DreamBoothDataset(Dataset):
    def __init__(self, instance_dir, instance_prompt, class_dir, class_prompt, tokenizer, resolution):
        self.instance_paths = sorted(
            p for p in glob.glob(os.path.join(instance_dir, "*")) if p.lower().endswith(IMG_EXTS)
        )
        self.class_paths = sorted(
            p for p in glob.glob(os.path.join(class_dir, "*")) if p.lower().endswith(IMG_EXTS)
        )
        if not self.instance_paths:
            raise SystemExit(f"Khong co anh instance trong {instance_dir}")
        if not self.class_paths:
            raise SystemExit(f"Khong co anh class trong {class_dir}")

        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.tokenizer = tokenizer
        self.tf = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.instance_paths)

    def _load(self, path, prompt):
        img = self.tf(Image.open(path).convert("RGB"))
        ids = self.tokenizer(
            prompt, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        ).input_ids[0]
        return img, ids

    def __getitem__(self, idx):
        inst_img, inst_ids = self._load(self.instance_paths[idx], self.instance_prompt)
        class_idx = idx % len(self.class_paths)
        cls_img, cls_ids = self._load(self.class_paths[class_idx], self.class_prompt)
        return {
            "instance_pixel": inst_img, "instance_ids": inst_ids,
            "class_pixel": cls_img, "class_ids": cls_ids,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    paths = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    pl_cfg = cfg.get("pseudo_label", {})

    output_dir = os.path.join(paths["output_dir"], "lora")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(train_cfg.get("seed", 42))

    base_model = model_cfg["base_model"]
    instance_prompt = pl_cfg.get("instance_prompt", "a photo of sks object")
    class_prompt = pl_cfg.get("class_prompt", "a photo of an object")
    instance_dir = paths["train_ready"]
    class_dir = os.path.join(os.path.dirname(instance_dir.rstrip("/")), "class_images")
    num_class_images = train_cfg.get("num_class_images", 12)
    prior_loss_weight = train_cfg.get("prior_loss_weight", 1.0)
    use_prior = train_cfg.get("prior_preservation", True)

    print(f"Instance prompt: \"{instance_prompt}\"  ({instance_dir})")
    print(f"Class prompt:    \"{class_prompt}\"  ({class_dir})")
    print(f"Prior preservation: {use_prior}  (weight={prior_loss_weight}, num_class_images={num_class_images})")

    if use_prior:
        generate_class_images(base_model, class_prompt, class_dir, num_class_images, model_cfg["resolution"], device)

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        mixed_precision=train_cfg.get("mixed_precision", "fp16"),
    )

    print(f"Tai base model de train: {base_model}")
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if train_cfg["mode"] == "lora":
        print(f"Che do LoRA (rank={train_cfg['lora_rank']})")
        unet.requires_grad_(False)
        lora_config = LoraConfig(
            r=train_cfg["lora_rank"], lora_alpha=train_cfg["lora_alpha"],
            lora_dropout=train_cfg["lora_dropout"],
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        unet = get_peft_model(unet, lora_config)
        trainable_params = [p for p in unet.parameters() if p.requires_grad]
    else:
        print("Che do full_dreambooth")
        unet.requires_grad_(True)
        trainable_params = list(unet.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=float(train_cfg["learning_rate"]))

    if use_prior:
        train_dataset = DreamBoothDataset(instance_dir, instance_prompt, class_dir, class_prompt, tokenizer, model_cfg["resolution"])
    else:
        train_dataset = DreamBoothDataset(instance_dir, instance_prompt, instance_dir, instance_prompt, tokenizer, model_cfg["resolution"])

    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg["train_batch_size"], shuffle=True)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_cfg["gradient_accumulation_steps"])
    max_train_steps = train_cfg.get("max_train_steps") or (train_cfg["num_train_epochs"] * num_update_steps_per_epoch)

    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
    vae.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    print(f"Bat dau train: {max_train_steps} steps")

    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    unet.train()
    while global_step < max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = torch.cat([batch["instance_pixel"], batch["class_pixel"]], dim=0)
                input_ids = torch.cat([batch["instance_ids"], batch["class_ids"]], dim=0)
                pixel_values = pixel_values.to(accelerator.device, dtype=torch.float16)
                input_ids = input_ids.to(accelerator.device)

                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(input_ids)[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                half = bsz // 2
                pred_instance, pred_class = model_pred[:half], model_pred[half:]
                noise_instance, noise_class = noise[:half], noise[half:]

                loss_instance = F.mse_loss(pred_instance.float(), noise_instance.float(), reduction="mean")

                if use_prior:
                    loss_class = F.mse_loss(pred_class.float(), noise_class.float(), reduction="mean")
                    loss = loss_instance + prior_loss_weight * loss_class
                else:
                    loss = loss_instance
                    loss_class = torch.tensor(0.0)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item(), inst=loss_instance.item(), cls=loss_class.item())

                if global_step % train_cfg["checkpointing_steps"] == 0:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.unwrap_model(unet).save_pretrained(save_path)
                    print(f"Da luu checkpoint: {save_path}")

            if global_step >= max_train_steps:
                break

    final_path = os.path.join(output_dir, "final")
    accelerator.unwrap_model(unet).save_pretrained(final_path)
    print(f"Hoan tat. LoRA weights luu o: {final_path}")
    print(f"Loss cuoi: total={loss.item():.4f}  instance={loss_instance.item():.4f}  class={loss_class.item():.4f}")


if __name__ == "__main__":
    main()
