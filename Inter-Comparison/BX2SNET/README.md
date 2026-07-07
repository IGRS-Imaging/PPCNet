# BX2S-Net — Inter-Comparison Baseline

## Paper Reference

> **BX2S-Net: Learning to Reconstruct 3D Spinal Structures from Bi-planar X-ray Images**
> Chen, Z., Guo, L., Zhang, R., Fang, Z., He, X., Wang, J.
> *Computers in Biology and Medicine, 2023*
> [Paper](https://doi.org/10.1016/j.compbiomed.2023.106615) | [Original Code](https://github.com/NBU-CVMI/bx2s-net/tree/main)

---

## Adaptation Protocol

This is a **re-implementation** of BX2S-Net for fair comparison with PPCNet. The following protocol ensures a controlled evaluation:

| Aspect | What we keep from BX2S-Net | What we keep from PPCNet |
|--------|---------------------------|-------------------------|
| **Architecture** | Dimensionality enhancement, VGG-style 3D encoder, FFAG decoder | — |
| **Loss functions** | Spatially-weighted BCE + Dice loss | — |
| **Optimizer** | Adam (lr=1e-3, wd=1e-4), cosine annealing | — |
| **Dataset** | — | Lumbar_Filtered_1037 (829/103/105 split) |
| **Coordinate system** | — | `pts_to_local` / `local_to_world` / `compute_scale` |
| **GT representation** | 64³ occupancy volume from GT point cloud | GT point cloud for evaluation |
| **Evaluation metrics** | — | Chamfer, F@1/2/5 mm, HD95 (world-mm) |
| **Output points** | — | 8,192 (extracted from predicted volume) |

> **Key difference from PPCNet:** BX2S-Net converts the two 2D DRRs into dimensionally-consistent pseudo-3D volumes by tiling along the depth axis, then processes them with a VGG-style 3D encoder and FFAG attention decoder. It does **not** use calibrated projection matrices.

---

## Architecture

```
Input: AP DRR (64×64) + LP DRR (64×64)
                    ↓
    ┌───────────────┴───────────────┐
    │   Dimensionality Enhancement  │
    │   AP: tile along depth axis   │
    │   LP: tile along height axis  │
    │   → 2-ch pseudo-3D (64³)     │
    └───────────────┬───────────────┘
                    ↓
    ┌───────────────┴───────────────┐
    │   VGG-style 3D Encoder        │
    │   (Conv-BN-ReLU ×2 blocks)    │
    │   32→64→128→256 channels      │
    │   64³→32³→16³→8³              │
    └───────────────┬───────────────┘
                    ↓
    ┌───────────────┴───────────────┐
    │   FFAG Decoder                │
    │   Full-scale Feature Attention│
    │   Guidance at each level      │
    │   Channel + Spatial Attention │
    │   8³→16³→32³→64³              │
    └───────────────┬───────────────┘
                    ↓
    Output: 64³ occupancy volume
                    ↓
    Point cloud extraction (8,192 pts)
```

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| BX2S-Net (Encoder + FFAG Decoder) | **23.2M** |

---

## Loss Functions (BX2S-Net Original)

| Loss | Weight | Description |
|------|--------|-------------|
| Spatially-weighted BCE | λ = 1.0 | Binary cross-entropy with 5× boundary weight |
| Dice | λ = 1.0 | Volumetric Dice loss |

> **No PPCNet losses used.** No Chamfer, no gap penalty, no axial density, no curvature loss.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-3, wd=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=200, η_min=1e-6) |
| Epochs | 200 (early stopped at 135) |
| Batch Size | 4 |
| Input Resolution | 64×64 (DRRs resized from 512) |
| Output Resolution | 64³ voxel grid |
| Boundary Weight | 5.0× on surface voxels |
| Gradient Clip | 1.0 |
| Precision | fp32 |
| GPU | NVIDIA A100-SXM4-80GB |

### Training Progress

```
Ep   1/200: L=1.124  bce=0.656  dice=0.468  Val: Chamfer=8.xxx mm
    ...
Ep  53/200: Val: Chamfer=2.347 mm                                ← Best
    ...
Ep 135/200: Early stop (60 epochs without improvement)
```

**Best checkpoint:** Epoch 53 (val Chamfer = 2.347 mm)

---

## Test Results (105 patients)

<div align="center">

| Metric | Mean ± Std | Median | 95th Percentile |
|:------:|:----------:|:------:|:---------------:|
| **Chamfer (mm) ↓** | 2.564 ± 1.471 | 2.371 | 3.173 |
| **F@1 ↑** | 0.081 ± 0.031 | 0.074 | 0.137 |
| **F@2 ↑** | 0.448 ± 0.110 | 0.451 | 0.603 |
| **F@5 ↑** | 0.927 ± 0.093 | 0.945 | 0.984 |
| **HD95 (mm) ↓** | 6.722 ± 5.829 | 5.767 | 8.940 |

</div>

### Comparison with PPCNet

<div align="center">

| Metric | BX2S-Net | PPCNet-v11 (Ours) | Δ |
|:------:|:--------:|:-----------------:|:-:|
| CD (mm) ↓ | 2.564 | **1.981** | −0.583 |
| F@1 ↑ | 0.081 | **0.155** | +0.074 |
| F@2 ↑ | 0.448 | **0.646** | +0.198 |
| F@5 ↑ | 0.927 | **0.973** | +0.046 |
| HD95 (mm) ↓ | 6.722 | **4.525** | −2.197 |

</div>

> **PPCNet outperforms BX2S-Net across all five metrics**, reducing Chamfer distance by 0.583 mm (22.7%) and HD95 by 2.197 mm (32.7%). Despite BX2S-Net's FFAG attention mechanism being specifically designed for biplanar fusion, it cannot match explicit projection conditioning. The dimensionality enhancement approach (tiling 2D images into pseudo-3D) discards the calibrated projection geometry that PPCNet leverages as its core lifting mechanism.

---

## Files

```
Inter-Comparison/BX2SNET/
├── README.md                              # This file
├── BX2S_NET_Inter_Comparison.ipynb        # Complete training + evaluation notebook
└── results/
    ├── test_results_bx2snet.csv           # Per-patient test metrics
    ├── BX2SNET_AP_View (lumbar_0028).png
    ├── BX2SNET_LP_View (lumbar_0028).png
    └── BX2SNET_Axial_View (lumbar_0028).png
```

---

## Usage

### 1. Update Paths

In Cell 1 of the notebook, update:

```python
DATA_ROOT   = Path("./data/Lumbar_Filtered_1037")
PROJECT_DIR = Path("./inter_comparison_bx2snet")
```

### 2. Run Training

Run all cells sequentially. The notebook handles:
- Data loading with GT volume generation (64³) and boundary weight maps
- Training with spatially-weighted BCE + Dice loss
- Automatic checkpointing (resumes from interruption)
- Validation every epoch with Chamfer distance monitoring
- Early stopping at 60 epochs patience

### 3. Test Evaluation

The final cells:
- Load best checkpoint
- Run inference on 105 test patients
- Extract 8,192-point clouds from predicted volumes
- Compute all five metrics in world-mm space
- Save per-patient CSV to `results/test_results_bx2snet.csv`

---

## Key Observations

1. **Dimensionality enhancement bottleneck** — tiling 2D images to create pseudo-3D volumes (64×64 → 64³) introduces redundant information along the depth axis, as the same 2D features are replicated without geometric guidance
2. **Early stopping at epoch 135** — the model plateaued with val Chamfer oscillating between 2.36–2.46 mm, unable to break below 2.347 mm despite continued loss decrease (bce: 0.223→0.193, dice: 0.171→0.135)
3. **FFAG attention helps locally but not globally** — the Full-scale Feature Attention Guidance improves boundary sharpness but cannot compensate for the lack of projection geometry awareness
4. **23.2M parameters at 2.564 mm** — the most parameter-efficient baseline but with the highest Chamfer, suggesting that architecture efficiency without geometric inductive bias yields diminishing returns

---

## Citation

```bibtex
@article{chen2023bx2s,
  author  = {Chen, Zhiqiang and Guo, Liangjun and Zhang, Rongguang 
             and Fang, Zhihui and He, Xinyi and Wang, Jianwei},
  title   = {{BX2S-Net}: Learning to Reconstruct {3D} Spinal Structures 
             from Bi-planar {X}-ray Images},
  journal = {Computers in Biology and Medicine},
  volume  = {154},
  pages   = {106615},
  year    = {2023},
  doi     = {10.1016/j.compbiomed.2023.106615}
}
```
