# src/utils.py
"""
Utility functions: metrics, seeding, checkpoint management.
"""

import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_coeff(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    dice = (2 * inter + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)
    return float(dice.mean())


def iou_score(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return float(iou.mean())


def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    torch.save(state, filename)
