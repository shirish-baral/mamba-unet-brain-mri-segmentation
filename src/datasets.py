# src/datasets.py
"""
Dataset utilities for Brain MRI segmentation.
Handles grayscale images and binary masks.
"""

from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def get_image_mask_pairs(image_dir: str, mask_dir: str):
    """Get all matching image–mask pairs."""
    img_dir, m_dir = Path(image_dir), Path(mask_dir)
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() in exts:
            mask_path = m_dir / img_path.name
            if mask_path.exists():
                pairs.append((str(img_path), str(mask_path)))
    return pairs


class BrainMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, preload=False):
        self.pairs = get_image_mask_pairs(image_dir, mask_dir)
        self.transform = transform
        self.preload = preload
        if preload:
            self.data = [
                (
                    np.array(Image.open(i).convert("L")),
                    np.array(Image.open(m).convert("L")),
                )
                for i, m in self.pairs
            ]
        else:
            self.data = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.preload:
            img, mask = self.data[idx]
        else:
            img_path, mask_path = self.pairs[idx]
            img = np.array(Image.open(img_path).convert("L"))
            mask = np.array(Image.open(mask_path).convert("L"))

        img = img.astype("float32") / 255.0
        mask = (mask > 127).astype("float32")

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)
        return img, mask
