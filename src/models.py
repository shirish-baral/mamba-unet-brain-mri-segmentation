# src/models.py
"""
Model definitions for the Mamba U-Net repository.

Includes:
- UNet (classic)
- AttentionUNet
- ASPP (lightweight)
- ResUNetASPP
- MambaUNet (uses MambaBlock from src.mamba_block)
- helpers: count_parameters, load_model

This file assumes `src/mamba_block.py` provides a class `MambaBlock`.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try relative import for MambaBlock (works when used as package)
try:
    from .mamba_block import MambaBlock
except Exception:
    # fallback to absolute import (works when run as script with PYTHONPATH including repo root)
    try:
        from src.mamba_block import MambaBlock
    except Exception:
        # If import fails, define a simple fallback here to avoid runtime errors.
        class MambaBlock(nn.Module):
            def __init__(self, channels, kernel=31):
                super().__init__()
                mid = max(16, channels // 2)
                self.pre = nn.Sequential(
                    nn.Conv2d(channels, mid, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mid),
                    nn.ReLU(inplace=True)
                )
                self.dw = nn.Conv2d(mid, mid, kernel_size=kernel, padding=kernel//2, groups=mid, bias=False)
                self.post = nn.Sequential(
                    nn.BatchNorm2d(mid),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels)
                )
                self.act = nn.ReLU(inplace=True)

            def forward(self, x):
                y = self.pre(x)
                y = self.dw(y)
                y = self.post(y)
                return self.act(x + y)


# -------------------------
# Basic building blocks
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch: Optional[int] = None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            # when using transposed conv, in_ch should be adjusted accordingly
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if necessary (odd sizes)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# -------------------------
# UNet
# -------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features: List[int] = [64, 128, 256, 512], bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        self.in_conv = DoubleConv(in_channels, features[0])
        self.downs = nn.ModuleList()
        for idx in range(len(features) - 1):
            self.downs.append(Down(features[idx], features[idx + 1]))
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        rev_features = list(reversed(features))
        up_in_channels = [features[-1] * 2] + [f * 2 for f in rev_features[:-1]]
        up_out_channels = rev_features
        self.ups = nn.ModuleList()
        for in_ch, out_ch in zip(up_in_channels, up_out_channels):
            self.ups.append(Up(in_ch, out_ch, bilinear=bilinear))
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        encs = [x1]
        for d in self.downs:
            encs.append(d(encs[-1]))
        bott = self.bottleneck(encs[-1])
        x = bott
        for up, skip in zip(self.ups, reversed(encs[:-1])):
            x = up(x, skip)
        return self.out_conv(x)


# -------------------------
# Attention U-Net
# -------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(UNet):
    def __init__(self, in_channels=1, out_channels=1, features: List[int] = [64, 128, 256, 512], bilinear=True):
        super().__init__(in_channels, out_channels, features, bilinear)
        rev_features = list(reversed(features))
        att_in = [f * 2 for f in rev_features]
        att_skip = rev_features
        self.attentions = nn.ModuleList()
        for g, x in zip(att_in, att_skip):
            self.attentions.append(AttentionGate(F_g=g, F_l=x, F_int=max(16, x // 2)))

    def forward(self, x):
        x1 = self.in_conv(x)
        encs = [x1]
        for d in self.downs:
            encs.append(d(encs[-1]))
        bott = self.bottleneck(encs[-1])
        x = bott
        for up, att, skip in zip(self.ups, self.attentions, reversed(encs[:-1])):
            gated = att(x, skip)
            x = up(x, gated)
        return self.out_conv(x)


# -------------------------
# ASPP
# -------------------------
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))
        self.project = nn.Sequential(
            nn.Conv2d(len(rates) * out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        x = torch.cat(outs, dim=1)
        return self.project(x)


# -------------------------
# ResUNet + ASPP
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        skip = self.skip(x)
        return self.relu(out + skip)


class ResUNetASPP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features: List[int] = [64, 128, 256, 512]):
        super().__init__()
        self.encs = nn.ModuleList()
        for idx, f in enumerate(features):
            in_ch = in_channels if idx == 0 else features[idx - 1]
            self.encs.append(ResidualBlock(in_ch, f))
        self.pool = nn.MaxPool2d(2)
        self.aspp = ASPP(features[-1], features[-1])
        rev_features = list(reversed(features))
        self.up_convs = nn.ModuleList()
        for i in range(len(rev_features) - 1):
            self.up_convs.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ResidualBlock(rev_features[i] + rev_features[i + 1], rev_features[i + 1])
            ))
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc_feats = []
        cur = x
        for enc in self.encs:
            cur = enc(cur)
            enc_feats.append(cur)
            cur = self.pool(cur)
        cur = self.aspp(cur)
        for i, up in enumerate(self.up_convs):
            skip = enc_feats[-(i + 1)]
            cur = up(torch.cat([cur, skip], dim=1))
        return self.out_conv(cur)


# -------------------------
# MambaUNet (uses MambaBlock)
# -------------------------
class MambaUNet(UNet):
    def __init__(self, in_channels=1, out_channels=1, features: List[int] = [64, 128, 256, 512], bilinear=True):
        super().__init__(in_channels, out_channels, features, bilinear)
        bott_ch = features[-1] * 2
        # replace bottleneck with DoubleConv + MambaBlock
        self.bottleneck = nn.Sequential(
            DoubleConv(features[-1], bott_ch),
            MambaBlock(bott_ch, kernel=31)
        )
        # optionally replace deepest encoder stage with MambaBlock appended
        if len(self.downs) >= 1:
            try:
                self.downs[-1] = nn.Sequential(
                    nn.MaxPool2d(2),
                    DoubleConv(features[-2], features[-1]),
                    MambaBlock(features[-1], kernel=31)
                )
            except Exception:
                pass


# -------------------------
# Helpers
# -------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(checkpoint_path: str, model_name: str = "unet", device: str = "cpu"):
    """
    Instantiate a model and load a checkpoint. Accepts checkpoints that are:
    - a state_dict
    - a dict with key "state_dict" or "model_state_dict"
    - a full checkpoint dict saved by training script
    """
    device = torch.device(device)
    mn = model_name.lower()
    if mn in ("mamba", "mamba_unet"):
        model = MambaUNet(in_channels=1, out_channels=1)
    elif mn in ("resunet_aspp", "resunet-aspp"):
        model = ResUNetASPP(in_channels=1, out_channels=1)
    elif mn in ("attention", "attention_unet"):
        model = AttentionUNet(in_channels=1, out_channels=1)
    else:
        model = UNet(in_channels=1, out_channels=1)

    model.to(device)
    loaded = torch.load(checkpoint_path, map_location=device)

    # common checkpoint key patterns
    if isinstance(loaded, dict) and "state_dict" in loaded:
        state = loaded["state_dict"]
    elif isinstance(loaded, dict) and "model_state_dict" in loaded:
        state = loaded["model_state_dict"]
    elif isinstance(loaded, dict) and any(k.startswith("module.") for k in loaded.keys()):
        state = loaded
    else:
        state = loaded

    if isinstance(state, dict):
        # strip "module." prefix if present (DataParallel)
        new_state = {}
        for k, v in state.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_key] = v
        try:
            model.load_state_dict(new_state)
        except Exception as e:
            print("Warning: strict load failed:", e)
            model.load_state_dict(new_state, strict=False)
    else:
        print("Warning: checkpoint format not recognized; returning uninitialized model.")

    model.eval()
    return model
