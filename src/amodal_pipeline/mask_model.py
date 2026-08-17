"""
Mask model nhỏ cho bài toán: dự đoán amodal mask (M_amodal) từ
(ảnh RGB, visible mask M_visible).

Kiến trúc: U-Net nhỏ, input 4 kênh (RGB + visible mask), output 1 kênh
(xác suất amodal mask, sigmoid). Đủ nhẹ để fine-tune trên vài chục/vài trăm
ảnh COCOA + ảnh của bạn, chạy tốt trên 1 GPU T4 hoặc CPU (chậm hơn) để test.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AmodalMaskUNet(nn.Module):
    """U-Net nhỏ: input (B,4,H,W) = concat(RGB, visible_mask) -> output (B,1,H,W)."""

    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.enc1 = conv_block(4, base_ch)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, image: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, visible_mask], dim=1)  # (B,4,H,W)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)  # logits, dùng BCEWithLogitsLoss


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()
