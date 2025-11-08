# src/models.py
"""
Model implementations for the Mamba-U-Net repo.

Contains:
- UNet (classic)
- AttentionUNet
- ASPP module
- ResUNetASPP
- MambaUNet (placeholder Mamba block integrated)
- helpers: count_parameters, load_model (compatible with common checkpoint formats)

The Mamba block is implemented as a lightweight, replaceable module (MambaBlock).
Replace MambaBlock with your full state-space implementation as needed.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
# Basic building blocks
# -----------------------------------------------------------
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
        # in_ch = channels of the input being upsampled
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            # using transposed conv; assumes in_ch is doubled from concat
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1: decoder feature to be upsampled
        # x2: skip connection from encoder
        x1 = self.up(x1)
        # handle odd size differences by padding
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# -----------------------------------------------------------
# Classic U-Net
# -----------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features: List[int] = [64, 128, 256, 512], bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        self.in_conv = DoubleConv(in_channels, features[0])
        self.downs = nn.ModuleList()
        for idx in range(len(features) - 1):
            self.downs.append(Down(features[idx], features[idx + 1]))
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        # construct ups
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


# -----------------------------------------------------------
# Attention U-Net
# -----------------------------------------------------------
class AttentionGate(nn.Module):
    """
    Attention gate used in Attention U-Net.
    Implementation adapted to be lightweight and compatible with UNet skips.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (decoder feature), x: skip connection (encoder feature)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(UNet):
    """
    Attention U-Net wrapper built on the UNet backbone.
    This implementation adds attention gates before concatenation in upsampling.
    """
    def __init__(self, in_channels=1, out_channels=1, features: List[int] = [64, 128, 256, 512], bilinear=True):
        super().__init__(in_channels, out_channels, features, bilinear)
        # build attention gates corresponding to upsampling stages
        # note: reversed features correspond to skip channels
        rev_features = list(reversed(features))
        att_in = [f * 2 for f in rev_features]  # gating features (decoder)
        att_skip = rev_features  # encoder skip features
        self.attentions = nn.ModuleList()
        for g, x in zip(att_in, att_skip):
            # intermediate channels
            self.attentions.append(AttentionGate(F_g=g, F_l=x, F_int=max(16, x // 2)))

    def forward(self, x):
        x1 = self.in_conv(x)
        encs = [x1]
        for d in self.downs:
            encs.append(d(encs[-1]))
        bott = self.bottleneck(encs[-1])
        x = bott
        # use attentions before concatenating skip
        for up, att, skip in zip(self.ups, self.attentions, reversed(encs[:-1])):
            gated = att(x, skip)
            x = up(x, gated)
        return self.out_conv(x)


# -----------------------------------------------------------
# ASPP module (lightweight)
# -----------------------------------------------------------
class ASPP(nn.Module):
    """
    Lightweight ASPP module. Produces multi-scale context features.
    """
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


# -----------------------------------------------------------
# ResUNet + ASPP
# -----------------------------------------------------------
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
        # encoder residual blocks
        self.encs = nn.ModuleList()
        for idx, f in enumerate(features):
            in_ch = in_channels if idx == 0 else features[idx - 1]
            self.encs.append(ResidualBlock(in_ch, f))
        self.pool = nn.MaxPool2d(2)
        self.aspp = ASPP(features[-1], features[-1])
        # decoder (simple upsample + residual conv)
        rev_features = list(reversed(features))
        self.up_convs = nn.ModuleList()
        for i in range(len(rev_features) - 1):
            self.up_convs.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ResidualBlock(rev_features[i] + rev_features[i+1], rev_features[i+1])
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
        # decoder: go backwards
        for i, up in enumerate(self.up_convs):
            skip = enc_feats[-(i+1)]
            cur = up(torch.cat([cur, skip], dim=1))
        return self.out_conv(cur)


# -----------------------------------------------------------
# Mamba block placeholder
# -----------------------------------------------------------
class MambaBlock(nn.Module):
    """
    Placeholder for a Mamba (state-space or other) block.
    This lightweight example implements a small gated conv + depthwise conv
    to mimic longer receptive field behavior while remaining simple.

    Replace or extend this class with your full Mamba state-space block.
    """
    def __init__(self, channels, kernel=31):
        super().__init__()
        # small gated mechanism
        mid = max(16, channels // 2)
        self.pre = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )
        # depthwise conv to increase receptive field (simulates long-range)
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


class MambaUNet(UNet):
    """
    Mamba U-Net: inherit UNet but replace some DoubleConv blocks with MambaBlock.
    This is a minimal, drop-in approach: we replace the bottleneck and the last encoder stage.
    Adjust replacements as you prefer.
    """
    def __init__(self, in_channels=1, out_channels=1, features: List[int] = [64, 128, 256, 512], bilinear=True):
        super().__init__(in_channels, out_channels, features, bilinear)
        # replace bottleneck with MambaBlock wrapper (project to channels)
        bott_ch = features[-1] * 2
        self.bottleneck = nn.Sequential(
            DoubleConv(features[-1], features[-1]*2),
            MambaBlock(bott_ch, kernel=31)
        )
        # replace the last encoder output (deepest encoder feature) with MambaBlock
        # locate last Down (deepest)
        if len(self.downs) >= 1:
            deep_down = self.downs[-1]
            # replace its DoubleConv (pool_conv[1]) with a sequential including MambaBlock
            try:
                base_out_ch = features[-1]
                self.downs[-1] = nn.Sequential(
                    nn.MaxPool2d(2),
                    DoubleConv(features[-2], features[-1]),
                    MambaBlock(features[-1], kernel=31)
                )
            except Exception:
                pass


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(checkpoint_path: str, model_name: str = "unet", device: str = "cpu"):
    """
    Instantiate a model by name and load checkpoint. Accepts state_dict or full checkpoint dict.
    """
    device = torch.device(device)
    model_name = model_name.lower()
    if model_name == "mamba" or model_name == "mamba_unet":
        model = MambaUNet(in_channels=1, out_channels=1)
    elif model_name == "resunet_aspp" or model_name == "resunet-aspp":
        model = ResUNetASPP(in_channels=1, out_channels=1)
    elif model_name == "attention" or model_name == "attention_unet":
        model = AttentionUNet(in_channels=1, out_channels=1)
    else:
        model = UNet(in_channels=1, out_channels=1)

    model.to(device)
    # load checkpoint
    loaded = torch.load(checkpoint_path, map_location=device)
    # common checkpoint conventions
    if isinstance(loaded, dict) and "state_dict" in loaded:
        state = loaded["state_dict"]
    elif isinstance(loaded, dict) and any(k.startswith("module.") for k in loaded.keys()):
        state = loaded
    elif isinstance(loaded, dict) and any(k.startswith("model") for k in loaded.keys()):
        # try to find key 'model' or 'model_state_dict'
        if "model_state_dict" in loaded:
            state = loaded["model_state_dict"]
        elif "model" in loaded:
            state = loaded["model"]
        else:
            state = loaded
    else:
        state = loaded

    # strip DataParallel 'module.' if present
    if isinstance(state, dict):
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
