# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse, time, os
from src.datasets import BrainMRIDataset
from src.transforms import get_train_transforms, get_val_transforms
from src.models import UNet, AttentionUNet, ResUNetASPP, MambaUNet, count_parameters
from src.utils import dice_coeff, iou_score, set_seed, save_checkpoint

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1 - (2. * inter + 1e-6) / (union + 1e-6)
        return dice.mean()

def build_model(name):
    name = name.lower()
    if name == "attention_unet": return AttentionUNet()
    if name == "resunet_aspp": return ResUNetASPP()
    if name == "mamba_unet": return MambaUNet()
    return UNet()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/brain_mri")
    p.add_argument("--model", default="unet")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", default="experiments")
    args = p.parse_args()

    set_seed(42)
    train_ds = BrainMRIDataset(f"{args.data_dir}/images", f"{args.data_dir}/masks", transform=get_train_transforms())
    val_ds = BrainMRIDataset(f"{args.data_dir}/images", f"{args.data_dir}/masks", transform=get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = build_model(args.model).to(args.device)
    print(f"Training {args.model} | Parameters: {count_parameters(model):,}")

    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = DiceLoss()
    best = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = torch.tensor(imgs).float().to(args.device), torch.tensor(masks).float().to(args.device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        dices, ious = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = torch.tensor(imgs).float().to(args.device), torch.tensor(masks).float().to(args.device)
                preds = (torch.sigmoid(model(imgs)) > 0.5).float()
                dices.append(dice_coeff(preds, masks))
                ious.append(iou_score(preds, masks))
        dice_val = sum(dices) / len(dices)
        iou_val = sum(ious) / len(ious)
        print(f"[{epoch:03d}] Loss={train_loss:.4f} | Dice={dice_val:.4f} | IoU={iou_val:.4f}")

        if dice_val > best:
            best = dice_val
            os.makedirs(f"{args.save-dir}/checkpoints", exist_ok=True)
            save_checkpoint({"state_dict": model.state_dict()}, f"{args.save_dir}/best.pth")

if __name__ == "__main__":
    main()
