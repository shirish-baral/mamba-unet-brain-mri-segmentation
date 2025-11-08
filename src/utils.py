# src/utils.py
"""
Utilities: seed, metrics, checkpoint save/load.
"""

import os
import random
import torch
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    # pred & target expected as binary tensors (0/1) or same shapes
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    dice = (2. * intersection + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)
    return float(dice.mean().item())

def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return float(iou.mean().item())

def save_checkpoint(state: dict, filename: str):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(filename: str, map_location=None):
    return torch.load(filename, map_location=map_location)
def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)