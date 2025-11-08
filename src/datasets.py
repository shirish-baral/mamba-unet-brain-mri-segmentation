# src/datasets.py
"""
Dataset utilities.

Provides:
- BrainMRIDataset: simple dataset that loads image/mask pairs (grayscale).
- get_image_mask_pairs: helper to list files given directories.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALB = True
except Exception:
    _HAS_ALB = False

def get_image_mask_pairs(image_dir: str, mask_dir: str, exts: Tuple[str]=(".tif", ".png", ".jpg")) -> List[Tuple[str,str]]:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    imgs = []
    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() in exts:
            mask_path = mask_dir / p.name
            if mask_path.exists():
                imgs.append((str(p), str(mask_path)))
    return imgs

class BrainMRIDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, transform=None, preload: bool=False):
        self.pairs = get_image_mask_pairs(image_dir, mask_dir)
        self.transform = transform
        self.preload = preload
        if preload:
            self.data = []
            for img_p, m_p in self.pairs:
                img = np.array(Image.open(img_p).convert("L"))
                mask = np.array(Image.open(m_p).convert("L"))
                self.data.append((img, mask))
        else:
            self.data = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.preload:
            img, mask = self.data[idx]
        else:
            img_p, mask_p = self.pairs[idx]
            img = np.array(Image.open(img_p).convert("L"))
            mask = np.array(Image.open(mask_p).convert("L"))

        # normalize to [0,1]
        img = img.astype("float32") / 255.0
        mask = (mask > 127).astype("float32")  # binary

        if self.transform is not None:
            if _HAS_ALB:
                augmented = self.transform(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']  # expected to be tensors
            else:
                # basic numpy transforms: to CHW torch tensor later in collate
                pass

        # ensure channel dimension (C,H,W)
        if isinstance(img, np.ndarray):
            img = np.expand_dims(img, axis=0)
        if isinstance(mask, np.ndarray):
            mask = np.expand_dims(mask, axis=0)

        return img.astype("float32"), mask.astype("float32")
