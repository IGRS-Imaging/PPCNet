
<div align="center">

# PPCNet: Projection-Conditioned Point Cloud Reconstruction of Spinal Vertebrae from Biplanar Radiographs
(The final model is **PPCNet-v6** (38.6M parameters, 8,192 points, ResNet-34 backbone), the result of a six-generation ablation study (v1–v6) details are provided.)

[![Dataset](https://img.shields.io/badge/Dataset-1%2C037_patients-green)](https://huggingface.co/datasets/ppcnet-dataset/PPCNet)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-brightgreen)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)](https://pytorch.org)

**WE MADE DATASET OPEN-SOURCE ➡️ [Dataset](https://huggingface.co/datasets/ppcnet-dataset/PPCNet)**

</div>

---

## About

**PPCNet** (Projection-Conditioned Point Cloud Network) reconstructs a **dense 8,192-point cloud** of the complete lumbar spine (L1–L5) from just **two orthogonal DRRs and their calibrated 3×4 projection matrices**. Unlike existing volumetric methods that fuse views through learned attention or concatenation, PPCNet uses the **projection geometry as the core 2D-to-3D lifting mechanism** — each query point is explicitly projected into both views, features are sampled, and bounded displacements nudge it toward the bone surface. With **gap-preserving and curvature losses**, PPCNet achieves **1.981 mm Chamfer distance** on 105 test patients, **97% VCL Grade A** clinical accuracy across 525 vertebrae, a morphometric MAE of **2.80 mm**, and a **3.01 mm mean centroid error** in phantom-based surgical navigation tracking.

---

## Architecture

<!-- Replace with your architecture diagram -->
<div align="center">
<img alt="arch_diag" src="https://github.com/user-attachments/assets/11855dd6-0128-4a2e-9cd1-c50d672d169e" />
</div>

<br>

PPCNet follows a six-stage pipeline:

> **Two DRRs + Projection Matrices** → Dual ResNet-34 Encoders → Feature Lift to 3D → Biplanar Fusion → Coarse 3D U-Net → Occupancy-Gated Query Init (8,192 points) → 3× Projection-Conditioned Refinement → **Dense 3D Point Cloud (world-mm)**

At each refinement stage, every query point is projected into both image planes via the known 3×4 matrices, 2D and 3D features are sampled at the projected locations, and a bounded displacement nudges the point toward the bone surface.

---

## Results

### Qualitative

<div align="center">
<img alt="implementation_figure" src="https://github.com/user-attachments/assets/748b12d3-a5a0-4df7-b9ad-92b546108fb3" />
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
<img alt="Paper_Clinical_Figure" src="https://github.com/user-attachments/assets/9b81a055-292f-4c0b-8cf3-25650b20454a" />
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

## Phantom-Based Surgical Navigation Tracking

To validate clinical applicability beyond computational metrics, we perform phantom-based navigation tracking using a 3D-printed lumbar spine phantom, an optical tracking system, and fiducial-based CT-to-phantom registration.

<div align="center">
<img alt="navigation_figure" src="https://github.com/user-attachments/assets/9568b0a9-4cab-4df5-94f8-7f401280b339" />
</div>

Phantom-based navigation: (i) needle placed at L4, (ii) needle placed at L1. Each row shows (A) physical setup, (B) CT-derived STL navigation, (C) PPCNet point cloud navigation.

### Registration Results

<div align="center">
    
| Metric | Value |
|--------|-------|
| Fiducial Registration Error (FRE) | **0.41 mm** |

</div>

### Vertebra Centroid Localisation Error

<div align="center">

| Vertebra | Error (mm) |
|----------|-----------|
| L1 | 2.84 |
| L2 | 3.85 |
| L3 | 2.56 |
| L4 | 2.31 |
| L5 | 3.50 |
| **Mean** | **3.01 mm** |
| **Max** | **3.85 mm** |

</div>

### Clinical Significance

- **FRE = 0.41 mm** — better than the 0.87 mm reported for clinical CT-navigated instrumentation ([Gubian et al., 2022](https://doi.org/10.3390/jcm11195530))
- **Mean centroid error = 3.01 mm** — well within the clinically accepted Gertzbein-Robbins Grade B threshold (<2 mm pedicle cortical breach) ([Gertzbein & Robbins, 1990](https://doi.org/10.1097/00007632-199001000-00004)) and consistent with ~5 mm screw tip deviations reported in CT-navigated spine surgery ([Virk & Qureshi, 2019](https://doi.org/10.21037/jss.2019.04.23))

### Ablation Study

<div align="center">
<img alt="ablation_visual" src="https://github.com/user-attachments/assets/f769b724-1d73-4669-b6c2-415d078cc9cd" />
</div>

<div align="center">

| Version | Params | Points | Key Addition | CD (mm) ↓ |
|:-------:|:------:|:------:|:------------|:---------:|
| PPCNet-v1 | 21.8M | 5,120 | Baseline + projection refinement | 2.672 |
| PPCNet-v2 | 21.8M | 5,120 | + Gap-occupancy, gated queries | 2.484 |
| PPCNet-v3 | 21.8M | 5,120 | + Unified pipeline | 2.452 |
| PPCNet-v4 | 21.8M | 5,120 | + Chamfer-ramp curriculum | 2.450 |
| PPCNet-v5 | 21.8M | 5,120 | + Stability-focused, phased losses | 1.999 |
| **PPCNet-v6** | **38.6M** | **8,192** | **+ Curvature, extent (ResNet-34)** | **1.981** |

</div>

---

## Dataset

We curate a custom dataset from **[VerSe'19 & VerSe'20](https://github.com/anjany/verse)** and **[CTSpine1K](https://github.com/MIRACLE-Center/CTSpine1K)**, selecting **1,037 patients** with complete L1–L5 lumbar segmentation labels. For each patient, we generate paired biplanar DRRs (AP + Lateral) using **[Plastimatch](https://plastimatch.org/)** ray-casting, along with calibrated 3×4 projection matrices and ground-truth point clouds — all configured for direct use with PPCNet training and evaluation. The complete dataset is open-sourced on Hugging Face.

**[⬇️ Download Dataset (Hugging Face)](https://huggingface.co/datasets/ppcnet-dataset/PPCNet)** (69.2 GB)

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

<summary><b>DRR Generation Parameters</b></summary>

<div align="center">

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

</div>

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
<details>

```
PPCNet/
├── README.md                           # This file
├── LICENSE
├── Media/
│   ├── Images/
│   ├── Videos/
├── Inter-Comparison/
│   ├── X2CTGAN/
│   │   ├── README.md
│   │   ├── X2CT_GAN_Inter_Comparison.ipynb
│   │   └── results/
│   │       ├── test_results_x2ctgan.csv
│   │       ├── X2CTGAN_AP_View (lumbar_0028).png
│   │       ├── X2CTGAN_LP_View (lumbar_0028).png
│   │       └── X2CTGAN_Axial_View (lumbar_0028).png
│   ├── BX2SNET/
│   │   ├── README.md
│   │   ├── BX2S_Net_Inter_Comparison.ipynb
│   │   └── results/
│   │       ├── test_results_bx2snet.csv
│   │       ├── BX2SNET_AP_View (lumbar_0028).png
│   │       ├── BX2SNET_LP_View (lumbar_0028).png
│   │       └── BX2SNET_Axial_View (lumbar_0028).png
│   ├── Swin-X2S/
│   │   ├── README.md
│   │   ├── Swin_X2S_Inter_Comparison.ipynb
│   │   └── results/
│   │       ├── test_results_swinx2s.csv
│   │       ├── SwinX2S_AP_View (lumbar_0028).png
│   │       ├── SwinX2S_LP_View (lumbar_0028).png
│   │       └── SwinX2S_Axial_View (lumbar_0028).png
│   └── 3D-ReVert/
│       ├── README.md
│       ├── 3D_ReVert_Inter_Comparison.ipynb
│       └── results/
│           ├── test_results_3drevert.csv
│           ├── 3DReVert_AP_View (lumbar_0028).png
│           ├── 3DReVert_LP_View (lumbar_0028).png
│           └── 3DReVert_Axial_View (lumbar_0028).png
├── PPCNet-v1/
│   ├── README.md
│   └── ppc_training_v6_fixed.ipynb
├── PPCNet-v2/
│   ├── README.md
│   ├── ppc_training_v7_gap_perfect.ipynb
│   └── ppc_v7_gan_refiner.ipynb
├── PPCNet-v3/
│   ├── README.md
│   └── ppc_training_v8_unified.ipynb
├── PPCNet-v4/
│   ├── README.md
│   └── ppc_v9_hybrid_copy.ipynb
├── PPCNet-v5/
│   ├── README.md
│   └── ppc_v10_stable.ipynb
└── PPCNet-v6/                         # ⭐ Final model
    ├── README.md
    └── ppc_v6_spine_aware.ipynb
```
</details>

---

## Quick Start

```python
# 1. Update paths in the notebook config cell
DATA_ROOT   = Path("/path/to/Lumbar_Filtered_1037")
PROJECT_DIR = Path("/path/to/output")

# 2. Run the final model (PPCNet-v6)
# Open PPCNet-v6/ppc_v6_spine_aware.ipynb and run all cells

# 3. Outputs
# ├── checkpoints/best_checkpoint.pth   # Trained model
# ├── results/test_results_v6_tta.csv  # Per-patient metrics
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

## Evaluation Metrics Used

<div align="center">

| Metric | Description | Direction |
|--------|-------------|:---------:|
| **CD** | Bidirectional Chamfer Distance (mm) | ↓ |
| **F@1** | F-Score at 1 mm threshold | ↑ |
| **F@2** | F-Score at 2 mm threshold | ↑ |
| **F@5** | F-Score at 5 mm threshold | ↑ |
| **HD95** | 95th percentile Hausdorff Distance (mm) | ↓ |

</div>

---

## License

This project is licensed under the [MIT License](LICENSE).
