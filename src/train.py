# src/train.py
"""
Training script.

Example:
python src/train.py --model unet --data-dir data/brain_mri --epochs 50 --batch-size 8
"""

import argparse
import os
import time
import yaml
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn

from src.datasets import BrainMRIDataset, get_image_mask_pairs
from src.utils import set_seed, dice_coeff, iou_score, save_checkpoint
from src.models import UNet, AttentionUNet, ResUNetASPP, MambaUNet, count_parameters

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2. * intersection + self.eps) / (probs.sum(dim=1) + targets.sum(dim=1) + self.eps)
        return 1. - dice.mean()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unet", choices=["unet", "attention_unet", "resunet_aspp", "mamba_unet"])
    p.add_argument("--data-dir", default="data/brain_mri")
    p.add_argument("--images-dir", default=None)
    p.add_argument("--masks-dir", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-dir", default="experiments")
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def build_model(name: str):
    mn = name.lower()
    if mn == "attention_unet":
        return AttentionUNet(in_channels=1, out_channels=1)
    elif mn == "resunet_aspp":
        return ResUNetASPP(in_channels=1, out_channels=1)
    elif mn == "mamba_unet":
        return MambaUNet(in_channels=1, out_channels=1)
    else:
        return UNet(in_channels=1, out_channels=1)

def prepare_datasets(data_dir: str, images_dir: str = None, masks_dir: str = None, val_ratio: float = 0.2, seed: int = 42):
    images_dir = images_dir or os.path.join(data_dir, "images")
    masks_dir = masks_dir or os.path.join(data_dir, "masks")
    pairs = get_image_mask_pairs(images_dir, masks_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"No image/mask pairs found in {images_dir} and {masks_dir}")
    ds = BrainMRIDataset(images_dir, masks_dir, transform=None, preload=False)
    n_val = max(1, int(len(ds) * val_ratio))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    return train_ds, val_ds

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs = torch.tensor(imgs).float().to(device)
        masks = torch.tensor(masks).float().to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def eval_epoch(model, loader, device):
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = torch.tensor(imgs).float().to(device)
            masks = torch.tensor(masks).float().to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            dices.append(dice_coeff(preds, masks))
            ious.append(iou_score(preds, masks))
    return float(sum(dices) / len(dices)), float(sum(ious) / len(ious))

def save_run_config(save_dir: str, args):
    os.makedirs(save_dir, exist_ok=True)
    cfg = vars(args).copy()
    cfg_path = os.path.join(save_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    run_name = args.run_name or f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.save_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Preparing data...")
    train_ds, val_ds = prepare_datasets(args.data_dir, args.images_dir, args.masks_dir, val_ratio=0.2, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=True)

    print("Building model:", args.model)
    model = build_model(args.model)
    print("Trainable parameters:", count_parameters(model))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = DiceLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    save_run_config(run_dir, args)
    best_score = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_dice, val_iou = eval_epoch(model, val_loader, device)
        scheduler.step(val_dice)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} - train_loss: {train_loss:.4f} - val_dice: {val_dice:.4f} - val_iou: {val_iou:.4f} - time: {elapsed:.1f}s")

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
        state = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "val_dice": val_dice}
        save_checkpoint(state, ckpt_path)

        if val_dice > best_score:
            best_score = val_dice
            best_path = os.path.join(ckpt_dir, "best.pth")
            save_checkpoint(state, best_path)
            print("Saved new best model to:", best_path)

    print("Training complete. Best val_dice:", best_score)
    print("Checkpoints saved in:", ckpt_dir)

if __name__ == "__main__":
    main()
