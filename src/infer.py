# src/infer.py
"""
Inference script for trained models.
"""

import torch
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from src.models import load_model

def load_image(path):
    img = Image.open(path).convert("L")
    x = np.expand_dims(np.expand_dims(np.array(img) / 255.0, 0), 0)
    return torch.from_numpy(x).float()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="unet")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = load_model(args.weights, model_name=args.model, device=args.device)
    model.eval()

    for img_path in Path(args.input_dir).iterdir():
        x = load_image(img_path).to(args.device)
        with torch.no_grad():
            pred = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        mask = (pred > 0.5).astype("uint8") * 255
        Image.fromarray(mask).save(Path(args.output_dir) / f"{img_path.stem}_mask.png")

if __name__ == "__main__":
    main()
