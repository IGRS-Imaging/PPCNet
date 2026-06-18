
<div align="center">

# PPCNet
### Projection-Conditioned Point Cloud Reconstruction of the Lumbar Spine from Biplanar Radiographs

[![Dataset](https://img.shields.io/badge/Dataset-1%2C037_patients-green)](https://drive.google.com/drive/folders/1pHZ8N0acK-EdG-VJ46D2npwnw4q4k75Y?usp=sharing)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-brightgreen)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)](https://pytorch.org)

**DATASET MADE OPEN-SOURCE ➡️ [Dataset](https://drive.google.com/drive/folders/1pHZ8N0acK-EdG-VJ46D2npwnw4q4k75Y?usp=sharing)**

</div>

---

## About

**PPCNet** (Projection-conditioned Point Cloud Network) reconstructs a **dense 8,192-point cloud** of the complete lumbar spine (L1–L5) from just **two orthogonal DRRs and their calibrated 3×4 projection matrices**. Unlike existing volumetric methods that fuse views through learned attention or concatenation, PPCNet uses the **projection geometry as the core 2D-to-3D lifting mechanism** — each query point is explicitly projected into both views, features are sampled, and bounded displacements nudge it toward the bone surface. With **gap-preserving and curvature losses**, PPCNet achieves **1.981 mm Chamfer distance** on 105 test patients, **97% VCL Grade A** clinical accuracy across 525 vertebrae, and a morphometric MAE of **2.80 mm**.

---

## Architecture

<!-- Replace with your architecture diagram -->
<div align="center">
<img width="9360" height="10140" alt="arch_diag" src="https://github.com/user-attachments/assets/11855dd6-0128-4a2e-9cd1-c50d672d169e" />
</div>

<br>

PPCNet follows a six-stage pipeline:

> **Two DRRs + Projection Matrices** → Dual ResNet-34 Encoders → Feature Lift to 3D → Biplanar Fusion → Coarse 3D U-Net → Occupancy-Gated Query Init (8,192 points) → 3× Projection-Conditioned Refinement → **Dense 3D Point Cloud (world-mm)**

At each refinement stage, every query point is projected into both image planes via the known 3×4 matrices, 2D and 3D features are sampled at the projected locations, and a bounded displacement nudges the point toward the bone surface.

---

## Results

### Qualitative

<div align="center">
<img width="2147" height="1118" alt="implementation_figure" src="https://github.com/user-attachments/assets/31275ebd-10c3-4033-8f78-802b3fc8912d" />
</div>

<p align="center"><i>(A) Input AP & Lateral DRRs → (B) Predicted 8,192-point cloud → (C) Per-vertebra labels via ICP → (D) Axial view of L1 → (E) Distance-to-GT heatmap</i></p>

### Quantitative (105 test patients)

<div align="center">

| Metric | Value |
|:------:|:-----:|
| Chamfer Distance | **1.981 ± 1.060 mm** |
| F-score @ 1 mm | 0.155 ± 0.070 |
| F-score @ 2 mm | 0.646 ± 0.160 |
| F-score @ 5 mm | **0.973 ± 0.074** |
| Hausdorff-95 | 4.525 ± 5.382 mm |

</div>

### Clinical Validation (525 vertebrae)

<div align="center">
<img width="10435" height="4020" alt="Paper_Clinical_Figure" src="https://github.com/user-attachments/assets/01f5760c-79ad-4d82-96ad-350620af2dc2" />
</div>

<div align="center">

| Parameter | MAE | Grade |
|:---------:|:---:|:-----:|
| Disc Height | 1.26 mm | Excellent |
| VBH / aVBH / pVBH | 3.07 / 2.77 / 2.86 mm | Good |
| VBW / VBD | 3.82 / 2.99 mm | Good |
| Lumbar Lordosis | 1.62° | Excellent |
| Coronal Cobb | 0.83° | Excellent |
| **Overall Morphometric** | **2.80 mm** | **Good** |
| VCL Grade A (VBH) | **509/525 (97%)** | — |

</div>

### Ablation Study

<div align="center">
<img width="2148" height="1380" alt="ablation_visual" src="https://github.com/user-attachments/assets/d37ff776-3b07-4c0d-b2b1-925b466ad146" />
</div>

<div align="center">

| Version | Params | Points | Key Addition | CD (mm) ↓ |
|:-------:|:------:|:------:|:------------|:---------:|
| PPCNet-v6 | 21.8M | 5,120 | Baseline + projection refinement | 2.672 |
| PPCNet-v7 | 21.8M | 5,120 | + Gap-occupancy, gated queries | 2.484 |
| PPCNet-v8 | 21.8M | 5,120 | + Unified pipeline | 2.452 |
| PPCNet-v9 | 21.8M | 5,120 | + Warm-start, Chamfer-ramp | 2.450 |
| PPCNet-v10 | 21.8M | 5,120 | + Stability-focused, phased losses | 1.999 |
| **PPCNet-v11** | **38.6M** | **8,192** | **+ Curvature, extent (ResNet-34)** | **1.981** |

</div>

---

## Dataset

The dataset is derived from **VerSe'19 & VerSe'20** and **CTSpine1K**, containing **1,037 patients** with complete L1–L5 lumbar segmentation.

**[⬇️ Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1pHZ8N0acK-EdG-VJ46D2npwnw4q4k75Y?usp=sharing)** (64.4 GB)

<details>
<summary><b>Dataset Structure</b></summary>

```
Lumbar_Filtered_1037/
├── dataset_split.json              # Fixed 829/103/105 split (seed=42)
├── lumbar_0001/
│   ├── ct.nii.gz                   # CT volume (LPS orientation)
│   ├── seg.nii.gz                  # Segmentation labels (L1=20 ... L5=24)
│   ├── gt_ppc.vtk                  # Ground-truth point cloud (5,120 pts)
│   ├── AP_0/
│   │   ├── drr_AP_0.png           # AP DRR (512×512)
│   │   ├── P_AP_0.txt             # 3×4 projection matrix
│   └── LP_90/
│       ├── drr_LP_90.png          # Lateral DRR (512×512)
│       ├── P_LP_90.txt            # 3×4 projection matrix
├── lumbar_0002/
│   └── ...
└── ...  (1,037 patients)
```

</details>

<details>
<summary><b>DRR Generation Parameters</b></summary>

| Parameter | Value |
|-----------|-------|
| Algorithm | Plastimatch ray-casting |
| SAD | 1,000 mm |
| SID | 1,500 mm |
| Detector | 500 × 500 mm |
| Resolution | 512 × 512 px |
| Views | AP (0°) + Lateral (90°) |
| Bone Enhancement | 2.5× for HU > 300 |
| Post-processing | CLAHE contrast normalisation |

</details>

---

## Installation

```bash
# Tested on NVIDIA A100-SXM4-80GB | CUDA 12.2 | Python 3.10

# Create environment
conda create -n ppcnet python=3.10 -y
conda activate ppcnet

# Install PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install numpy scipy scikit-image nibabel open3d vtk pillow tqdm matplotlib
```
---

## Repository Structure

```
PPCNet/
├── README.md                           # This file
├── LICENSE
├── assets/                             # Figures for README
│   ├── arch_diag.png
│   ├── implementation_figure.png
│   ├── Paper_Clinical_Figure.png
│   └── ablation_visual.png
├── PPCNet-v6/
│   ├── README.md                       # Version-specific details
│   └── ppc_training_v6_fixed.ipynb
├── PPCNet-v7/
│   ├── README.md
│   ├── ppc_training_v7_gap_perfect.ipynb
│   └── ppc_v7_gan_refiner.ipynb
├── PPCNet-v8/
│   ├── README.md
│   └── ppc_training_v8_unified.ipynb
├── PPCNet-v9/
│   ├── README.md
│   └── ppc_v9_hybrid_copy.ipynb
├── PPCNet-v10/
│   ├── README.md
│   └── ppc_v10_stable.ipynb
└── PPCNet-v11/                         # ⭐ Final model (paper)
    ├── README.md
    └── ppc_v11_spine_aware.ipynb
```

---

## Quick Start

```python
# 1. Update paths in the notebook config cell
DATA_ROOT   = Path("/path/to/Lumbar_Filtered_1037")
PROJECT_DIR = Path("/path/to/output")

# 2. Run the final model (PPCNet-v11)
# Open PPCNet-v11/ppc_v11_spine_aware.ipynb and run all cells

# 3. Outputs
# ├── checkpoints/best_checkpoint.pth   # Trained model
# ├── results/test_results_v11_tta.csv  # Per-patient metrics
# └── results/<patient_id>_pred.vtk     # Predicted point clouds
```

Each notebook is self-contained with:
- ⚙️ **Config** — all hyperparameters in one cell
- 📦 **Data** — dataset class with augmentation
- 🏗️ **Model** — full architecture definition
- 📉 **Loss** — all loss functions
- 🏋️ **Training** — with checkpoint resume
- 📊 **Evaluation** — test metrics + VTK export

---

## Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|:---------:|
| **CD** | Bidirectional Chamfer Distance (mm) | ↓ |
| **F@1** | F-Score at 1 mm threshold | ↑ |
| **F@2** | F-Score at 2 mm threshold | ↑ |
| **F@5** | F-Score at 5 mm threshold | ↑ |
| **HD95** | 95th percentile Hausdorff Distance (mm) | ↓ |

---

<div align="center">

**⭐ If you find this repository helpful, please consider giving it a star! ⭐**

</div>
