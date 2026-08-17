# Amodal Fine-tuning Pipeline (small-data)

**Scope hiện tại (theo yêu cầu của thầy hướng dẫn): chỉ fine-tune 2 nhánh
độc lập — Object Mask và Object Appearance.** Không cần chạy baseline
3-service của Multi-Agent-Amodal (SAM API + FLUX API + GPT-4o orchestration)
— đó là hệ training-free, không fine-tune được. Ở đây train 2 model nhỏ
trực tiếp trên COCOA, KHÔNG cần GPU kép, KHÔNG cần OpenRouter/Azure key.

```
                    COCOA (images, visible_masks, amodal_masks)
                                     |
                +--------------------+--------------------+
                |                                         |
        [04a] Mask branch                        [02]+[03]+[04] Appearance branch
        U-Net nhỏ: (ảnh, visible_mask)            LoRA fine-tune Stable Diffusion:
              -> amodal_mask                       ảnh + caption -> nội dung vùng che
                |                                         |
        [06] Evaluate (IoU)                    [05] Evaluate (CLIP/LPIPS/SSIM)
```

Hai nhánh **độc lập với nhau**, có thể chạy song song, dùng chung mỗi
COCOA làm dữ liệu gốc. Xem chi tiết từng nhánh bên dưới.

## Cấu trúc thư mục

```
amodal-finetune/
├── configs/config.yaml          # toàn bộ tham số (paths, model, LoRA, training)
├── data/
│   ├── raw/                     # ảnh gốc của bạn (ít ảnh) -> để vào đây
│   └── cocoa/                   # annotation COCOA tải về
├── scripts/
│   ├── 01_prepare_cocoa.py      # tải + parse COCOA -> mask PNG + ảnh
│   ├── 02_pseudo_label.py       # sinh M_visible, M_occ, P_text cho data/raw
│   ├── 03_prepare_training_data.py  # gom thành format instance_images/captions
│   ├── 04_train_lora.py         # fine-tune LoRA trên UNet (kiểu DreamBooth)
│   └── 05_evaluate.py           # CLIP score, LPIPS, SSIM trước/sau fine-tune
├── src/amodal_pipeline/
│   ├── mask_utils.py            # union mask, dilation, RLE<->mask, v.v.
│   └── agents.py                # gọi GPT-4o/VLM cho occlusion + description agent
├── notebooks/kaggle_train.ipynb # notebook chạy trực tiếp trên Kaggle GPU
└── requirements.txt
```

## Chạy local (test nhanh, CPU/GPU nhỏ)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Bước chung: tải COCOA làm ground truth cho cả 2 nhánh
python scripts/01_prepare_cocoa.py --annotation_targz /path/to/cocoa_annotation.tar.gz \
    --out data/cocoa --n_samples 200
```

### Nhánh A — Mask (fine-tune model dự đoán amodal mask)

```bash
python scripts/04a_train_mask_model.py --config configs/config.yaml --epochs 30
python scripts/06_evaluate_mask.py --config configs/config.yaml \
    --checkpoint outputs/mask_model/best.pt
```
Không cần API key, không cần SAM/GroundingDINO thật — model tự học từ cặp
(visible_mask, amodal_mask) có sẵn trong COCOA.

### Nhánh B — Appearance (fine-tune LoRA sinh nội dung vùng bị che)

```bash
# Bỏ ảnh của bạn vào data/raw/, rồi sinh caption tự động (cần OpenAI-compatible key)
export OPENAI_API_KEY=sk-...
python scripts/02_pseudo_label.py --images data/raw --out data/pseudo_labels

python scripts/03_prepare_training_data.py \
    --pseudo data/pseudo_labels --out data/train_ready

python scripts/04_train_lora.py --config configs/config.yaml
python scripts/05_evaluate.py --config configs/config.yaml \
    --lora_path outputs/lora/final
```
Nhánh này vẫn cần một VLM (GPT-4o hoặc tương đương) để sinh caption chi
tiết ($P_{text}$) — không cần SAM/FLUX thật, chỉ cần chính Stable Diffusion
mà bạn đang fine-tune.

## Đẩy lên GitHub

```bash
cd amodal-finetune
git init
git add .
git commit -m "init: amodal fine-tuning pipeline"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

`.gitignore` đã loại `data/`, `outputs/`, checkpoint, cache — chỉ push code + config,
không push ảnh/dữ liệu nặng hoặc trọng số mô hình. Ảnh dataset ít của bạn nên
để trên Kaggle Dataset riêng (private) thay vì commit vào git.

## Chuyển sang Kaggle để train

1. Trên Kaggle: **New Notebook** → tab **Add-ons → Add Secret** để lưu
   `OPENAI_API_KEY` (nếu bước 02 chạy trên Kaggle) và `HF_TOKEN` nếu cần tải
   model gated (VD: SD 2.1, SDXL).
2. Bật GPU: Settings → Accelerator → GPU T4 x2 hoặc P100.
3. Upload ảnh gốc (`data/raw`) thành một **Kaggle Dataset** riêng (Add Data),
   không cần commit vào GitHub.
4. Trong notebook, cell đầu tiên clone repo GitHub của bạn:

```python
!git clone https://github.com/<user>/<repo>.git
%cd <repo>
!pip install -r requirements.txt -q
```

5. Copy dataset đã upload vào `data/raw`, rồi chạy tuần tự script 02 → 03 → 04 → 05
   (xem chi tiết trong `notebooks/kaggle_train.ipynb`, đã viết sẵn từng cell).
6. Checkpoint LoRA lưu ở `outputs/lora/` — cuối notebook có cell `!zip` để
   nén và tải về, vì Kaggle không giữ session lâu dài.

## Ghi chú "fine-tuning nhỏ nhỏ"

- Mặc định `04_train_lora.py` chỉ fine-tune **LoRA trên UNet** (không full
  fine-tune), rank mặc định 4-8, phù hợp vài chục ảnh — đúng tinh thần
  DreamBooth nhưng nhẹ hơn, chạy được trên GPU free tier của Kaggle (T4, 16GB).
- Nếu muốn đúng DreamBooth gốc hơn (full UNet fine-tune + prior preservation
  loss), đổi `train.mode: full_dreambooth` trong `config.yaml` — sẽ tốn VRAM
  hơn nhiều, nên test trên P100/A100 nếu có.
