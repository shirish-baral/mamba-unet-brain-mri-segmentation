# src/transforms.py
"""
Albumentations-based augmentations for medical segmentation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.0,), std=(1.0,)),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.0,), std=(1.0,)),
        ToTensorV2(),
    ])
