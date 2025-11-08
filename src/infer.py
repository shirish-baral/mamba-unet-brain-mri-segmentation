# src/infer.py
"""
Simple inference CLI for saved checkpoints.

Example:
python src/infer.py --weights experiments/<run>/checkpoints/best.pth --input_dir data/images --output_dir out_preds
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from src.models import load_model
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--model", default="unet", choices=["unet", "attention_unet", "resunet_aspp", "mamba_unet"])
    return p.parse_args()

def preprocess_img(img_path):
    img = Image.open(img_path).convert("L")
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(np.expand_dims(arr, 0), 0)  # 1x1xHxW
    return torch.from_numpy(arr)

def save_mask(mask_np, out_path):
    Image.fromarray(mask_np.astype("uint8")).save(out_path)

def main():
    args = parse_args()
    device = args.device
    model = load_model(args.weights, model_name=args.model, device=device)
    model.eval()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]])
    if len(img_paths) == 0:
        print("No images found in", input_dir)
        return
    with torch.no_grad():
        for p in img_paths:
            inp = preprocess_img(p).to(device)
            pred = model(inp)
            if isinstance(pred, torch.Tensor):
                pred = torch.sigmoid(pred).cpu().numpy()[0,0]
            else:
                pred = np.array(pred)
            mask = (pred > args.threshold).astype("uint8") * 255
            out_name = output_dir / (p.stem + "_pred.png")
            save_mask(mask, out_name)
            print("Saved:", out_name)
    print("Done. Predictions saved to", output_dir)

if __name__ == "__main__":
    main()
