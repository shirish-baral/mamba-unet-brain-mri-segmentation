# 🧠 Mamba U-Net for Brain MRI Segmentation

This repository contains the implementation, experiments, and supporting materials for the paper  
**“Mamba U-Net: Integrating State Space Models for Efficient Medical Image Segmentation.”**

It includes comparative experiments between **U-Net**, **Attention U-Net**, **ASPP-enhanced U-Net variants**, and the proposed **Mamba U-Net** architecture on brain MRI datasets.

---

## 📁 Repository Structure
```bash
mamba-unet-brain-mri-segmentation/
├── README.md
├── requirements.txt
├── environment.yml
├── paper/
│ └── Mamba_Architecture_model.pdf
├── notebooks/
│ ├── 01_unet_baseline.ipynb
│ ├── 02_attention_unet.ipynb
│ ├── 03_unet_aspp_resunet_aspp.ipynb
│ └── 04_implement_mamba.ipynb
├── src/
│ ├── models/
│ ├── datasets.py
│ ├── transforms.py
│ ├── train.py
│ └── infer.py
├── data/
├── experiments/
├── figures/
└── docs/
```

---

## 🚀 Models Included
```bash
| Model | Description |
|-------|--------------|
| **U-Net** | Classic encoder–decoder CNN for segmentation. |
| **Attention U-Net** | Adds attention gates to focus on relevant spatial regions. |
| **U-Net + ASPP** | Integrates Atrous Spatial Pyramid Pooling for multi-scale feature extraction. |
| **ResUNet + ASPP** | Residual backbone with ASPP for deeper contextual learning. |
| **Mamba U-Net (Proposed)** | Incorporates *Mamba* (Selective State Space) blocks for efficient long-range dependency modeling. |
```
---

## 🧩 Dataset
```bash
Experiments use the **Brain MRI Segmentation Dataset** (Kaggle), containing 7,860 `.tif` images with corresponding binary masks.

data/
├── images/
│ ├── image_001.tif
│ ├── image_002.tif
└── masks/
├── mask_001.tif
├── mask_002.tif


```
All notebooks automatically handle grayscale conversion, normalization, and an **80 : 20 train/validation split**.

---

## ⚙️ Environment Setup


```bash
git clone https://github.com/<your-username>/mamba-unet-brain-mri-segmentation.git
cd mamba-unet-brain-mri-segmentation
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## 🧩 Requirements (main libraries)
```bash
torch
torchvision
numpy
matplotlib
opencv-python
scikit-image
pandas
albumentations
tqdm
jupyterlab
tensorboard
```

## 📘 Notebooks Overview
```bash
Notebook	Description
01_unet_baseline.ipynb	Baseline U-Net training and evaluation.
02_attention_unet.ipynb	Implements Attention U-Net with attention gates.
03_unet_aspp_resunet_aspp.ipynb	Combines ASPP with U-Net and ResUNet architectures.
04_implement_mamba.ipynb	Implements and evaluates the proposed Mamba U-Net.

Each notebook visualizes predictions and computes Dice and IoU metrics.
```
## 🧠 Proposed Architecture – Mamba U-Net

Mamba U-Net integrates Selective State Space (Mamba) blocks into the U-Net encoder, enabling:

Long-range dependency modeling

Efficient memory utilization

Competitive segmentation accuracy with reduced complexity

## 📊 Experimental Results (Summary)
```bash
Model	Params (M)	Dice	IoU	Inference (ms)
U-Net	7.85	0.842	0.728	36.7
Attention U-Net	8.12	0.847	0.732	35.8
U-Net + ASPP	9.21	0.854	0.741	28.5
ResUNet + ASPP	10.4	0.852	0.739	29.6
Mamba U-Net (Proposed)	9.65	0.849	0.736	31.2

(Refer to the paper for complete metrics and discussion.)
```

## 🧪 Reproducibility
```bash

Fixed random seed: torch.manual_seed(42)

Deterministic DataLoader behavior

Split: 80 % train / 20 % validation

Hardware: NVIDIA RTX GPU (CUDA 12.1)

Logs and checkpoints → experiments/

Full configuration details in docs/REPRODUCIBILITY.md.
```

## 🧰 Scripts (under src/)
Script	Purpose
train.py	Trains selected model via command-line arguments.
infer.py	Runs inference on unseen images.
datasets.py	Dataset loader and preprocessing utilities.
transforms.py	Data augmentation definitions.
models/	All model architectures (U-Net variants + Mamba blocks).

Example usage

python src/train.py --model mamba_unet --epochs 50 --batch_size 8 --lr 1e-4

## 📈 Visualization

Qualitative predictions are saved under:

figures/qualitative_results/

## 🧾 Citation

If you use this repository, please cite:

@article{baral2025mambaunet,
  title={Mamba U-Net: Integrating State Space Models for Efficient Medical Image Segmentation},
  author={Baral, Shirish and et al.},
  year={2025},
  journal={Under Review}
}

## 🪪 License

Released under the MIT License — see LICENSE
 for details.

## 🙌 Acknowledgments

Vision Mamba (2024) – for state-space architecture inspiration

Kaggle Brain MRI Dataset – public dataset source

Ronneberger et al., 2015 – original U-Net architecture

## 🤝 Contributing

Contributions are welcome!

Fork the repo

Create your feature branch (feature/<name>)

Commit your changes

Push and open a Pull Request

## 📬 Contact

Author: Shirish Baral
Email: baral.shirish8@gmail.com

GitHub: https://github.com/shirish-baral