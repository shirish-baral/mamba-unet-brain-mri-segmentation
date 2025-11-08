# 🧠 Mamba U-Net for Brain MRI Segmentation

This repository contains the implementation, experiments, and supporting materials for the paper  
**“Mamba U-Net: Integrating State Space Models for Efficient Medical Image Segmentation.”**

It includes comparative experiments between **U-Net**, **Attention U-Net**, **ASPP-enhanced U-Net variants**, and the proposed **Mamba U-Net** architecture on brain MRI datasets.

---

## 📁 Repository Structure

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


---
