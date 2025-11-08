
---

```markdown
# 🧠 Mamba U-Net for Brain MRI Segmentation

This repository contains the implementation, experiments, and supporting materials for the paper  
**“Mamba U-Net: Integrating State Space Models for Efficient Medical Image Segmentation.”**

It includes comparative experiments between **U-Net**, **Attention U-Net**, **ASPP-enhanced U-Net variants**, and the proposed **Mamba U-Net** architecture on brain MRI datasets.

---

## 📁 Repository Structure

```

mamba-unet-brain-mri-segmentation/
├── README.md
├── requirements.txt
├── environment.yml
├── paper/
│   └── Mamba_Architecture_model.pdf
├── notebooks/
│   ├── 01_unet_baseline.ipynb
│   ├── 02_attention_unet.ipynb
│   ├── 03_unet_aspp_resunet_aspp.ipynb
│   └── 04_implement_mamba.ipynb
├── src/
│   ├── models/
│   ├── datasets.py
│   ├── transforms.py
│   ├── train.py
│   └── infer.py
├── data/
├── experiments/
├── figures/
└── docs/

```

---

## 🚀 Models Included

| Model | Description |
|-------|--------------|
| **U-Net** | Classic encoder–decoder CNN for segmentation. |
| **Attention U-Net** | Adds attention gates to focus on relevant regions. |
| **U-Net + ASPP** | Integrates Atrous Spatial Pyramid Pooling for multi-scale context. |
| **ResUNet + ASPP** | Residual backbone with ASPP for deeper features. |
| **Mamba U-Net (Proposed)** | Replaces convolutional blocks with *Mamba* (Selective State Space) layers for long-range efficiency. |

---

## 🧩 Dataset

Experiments use the **Brain MRI Segmentation Dataset** (Kaggle) containing 7,860 `.tif` images and corresponding binary masks.

```

data/
├── images/
│   ├── image_001.tif
│   ├── image_002.tif
└── masks/
├── mask_001.tif
├── mask_002.tif

````

All notebooks automatically handle grayscale conversion, normalization, and 80:20 train/validation split.

---

## ⚙️ Environment Setup

### 🔹 Option 1 – pip
```bash
git clone https://github.com/<your-username>/mamba-unet-brain-mri-segmentation.git
cd mamba-unet-brain-mri-segmentation
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
pip install -r requirements.txt
````

### 🔹 Option 2 – conda

```bash
conda env create -f environment.yml
conda activate mamba-unet
```

---

## 🧩 Requirements (main libraries)

```
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

---

## 📘 Notebooks Overview

| Notebook                          | Description                                     |
| --------------------------------- | ----------------------------------------------- |
| `01_unet_baseline.ipynb`          | Trains baseline U-Net model.                    |
| `02_attention_unet.ipynb`         | Experiments with Attention U-Net.               |
| `03_unet_aspp_resunet_aspp.ipynb` | Evaluates ASPP and ResUNet+ASPP architectures.  |
| `04_implement_mamba.ipynb`        | Implements and evaluates the Mamba U-Net model. |

Each notebook visualizes sample predictions and computes Dice & IoU metrics.

---

## 🧠 Proposed Architecture: Mamba U-Net

**Mamba U-Net** integrates *Selective State Space (Mamba)* blocks into U-Net’s encoder, enabling:

* Long-range dependency modeling
* Efficient memory usage
* Strong segmentation accuracy with reduced complexity

---

## 📊 Experimental Results (Summary)

| Model                      | Params (M) | Dice  | IoU   | Inference (ms) |
| -------------------------- | ---------- | ----- | ----- | -------------- |
| U-Net                      | 7.85       | 0.842 | 0.728 | 36.7           |
| Attention U-Net            | 8.12       | 0.847 | 0.732 | 35.8           |
| U-Net + ASPP               | 9.21       | 0.854 | 0.741 | 28.5           |
| ResUNet + ASPP             | 10.4       | 0.852 | 0.739 | 29.6           |
| **Mamba U-Net (Proposed)** | 9.65       | 0.849 | 0.736 | 31.2           |

*(Refer to the paper for complete metrics and analysis.)*

---

## 🧪 Reproducibility

* Fixed random seed: `torch.manual_seed(42)`
* Deterministic DataLoader
* Split: 80% train / 20% validation
* Hardware: NVIDIA RTX GPU (CUDA 12.1)
* Logs and checkpoints: `experiments/`

More details in `docs/REPRODUCIBILITY.md`.

---

## 🧰 Scripts (under `src/`)

| Script          | Purpose                                 |
| --------------- | --------------------------------------- |
| `train.py`      | Train any model with command-line args. |
| `infer.py`      | Run inference on new images.            |
| `datasets.py`   | Handles dataset reading and transforms. |
| `transforms.py` | Defines augmentations.                  |
| `models/`       | Contains all architecture definitions.  |

Example usage:

```bash
python src/train.py --model mamba_unet --epochs 50 --batch_size 8 --lr 1e-4
```

---

## 📈 Visualization

Qualitative predictions are stored in:

```
figures/qualitative_results/
```

---

## 🧾 Citation

If you use this repository, please cite:

```
@article{baral2025mambaunet,
  title={Mamba U-Net: Integrating State Space Models for Efficient Medical Image Segmentation},
  author={Baral, Shirish and et al.},
  year={2025},
  journal={Under Review}
}
```

---

## 🪪 License

Released under the **MIT License** — see `LICENSE` for details.

---

## 🙌 Acknowledgments

* **Vision Mamba (2024)** – inspiration for state-space modeling
* **Kaggle Brain MRI Dataset** – public data source
* **Ronneberger et al., 2015** – original U-Net architecture

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create your feature branch (`feature/xyz`)
3. Commit your changes
4. Push and open a Pull Request

---

## 📬 Contact

**Author:** Shirish Baral
**Email:** [baral.shirish8@gmail.com](mailto:baral.shirish8@gmail.com)
**GitHub:** [https://github.com/shirish-baral](https://github.com/shirish-baral)

---

**⭐ If you find this repository helpful, please give it a star on GitHub!**

```

---

✅ **Instructions:**
1. Copy all text above (including the first and last triple backticks).  
2. Paste into a new file named `README.md` in your repository root.  
3. Save and push — it will render perfectly on GitHub with all bold, tables, and code blocks correctly formatted.  

---

```
