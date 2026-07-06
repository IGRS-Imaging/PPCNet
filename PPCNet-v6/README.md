# PPCNet-v6: Spine-Aware (Final Model)

## Overview

PPCNet-v6 is the final and best-performing version, introducing architectural scaling (ResNet-18 → ResNet-34, 5,120 → 8,192 points, QUERY_DIM 256 → 384) alongside spine-aware losses including curvature preservation and extent matching. It uses OneCycleLR scheduling with gradient accumulation for effective batch size 8. This is the model reported in the paper.

## Architecture

| Component | Details |
|-----------|---------|
| Encoder | 2× **ResNet-34** (ImageNet pretrained, 1-ch adapted) |
| Feature Maps | 192-ch at 32×32 |
| Volume | 128-ch at 32³ |
| Auxiliary Volume | 48³ occupancy |
| Query Points | **8,192** (up from 5,120) |
| Query Grid | 20×20×21 = 8,400 → subsampled to 8,192 |
| Refinement Stages | 3 (independent weights) |
| MLP Dimension | **384** (up from 256) |
| Displacement Bound | 0.25·tanh(·) per stage |
| Total Parameters | **38.6M** |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=3e-4, β₁=0.9, β₂=0.999, wd=5e-5) |
| Scheduler | OneCycleLR (5% warmup, cosine decay, div_factor=25) |
| Encoder LR | 0.3× body LR |
| Epochs | 300 |
| Batch Size | 2 (physical) × 4 (accumulation) = **8 effective** |
| Precision | fp32 |
| Gradient Clip | 1.0 |
| Encoder Frozen | Epochs 1–5 |
| Early Stopping | 60 epochs patience |

## Loss Functions (8 total, 2 phases)

**Phase 1 (epoch 1+):**
- Asymmetric Chamfer (truncated at 8 mm, GT→pred weighted 1.5×)
- Gap penalty (w=4.0)
- Axial density KDE (w=0.8)
- Auxiliary occupancy BCE+Dice (w=0.05)
- Extent match (w=1.5)

**Phase 2 (epoch 5+, linearly ramped):**
- Sliced Wasserstein distance (w=0.3)
- 2D projection consistency (w=0.02)
- Curvature loss (w=0.5) — matches per-band XY centroids across 20 axial slices

## Data Augmentation

- Gamma correction (γ ∈ [0.7, 1.5])
- Horizontal flip with projection matrix adjustment
- Gaussian noise (σ ∈ [0.0, 0.03])
- Elastic deformation (α=8, σ=3)
- Cutout (2 holes, 40×40 px)
- Affine transforms (±3°, ±8 px, ±4%)

## Test Results (105 patients, with TTA)

| CD (mm)↓ | F@1↑ | F@2↑ | F@5↑ | HD95 (mm)↓ |
|----------|------|------|------|------------|
| **1.981** | **0.155** | 0.646 | **0.973** | 4.525 |

**Test-Time Augmentation:** Averages predictions from original and horizontally-flipped inputs (Δ = 0.004 mm improvement).

## Clinical Validation (525 vertebrae)

| Metric | MAE | Grade |
|--------|-----|-------|
| Disc Height | 1.26 mm | Excellent |
| VBH | 3.07 mm | Good |
| aVBH | 2.77 mm | Good |
| pVBH | 2.86 mm | Good |
| VBW | 3.82 mm | Good |
| VBD | 2.99 mm | Good |
| Lumbar Lordosis | 1.62° | Excellent |
| Coronal Cobb | 0.83° | Excellent |
| **Overall Morphometric** | **2.80 mm** | **Good** |
| VCL Grade A (VBH) | 509/525 (97%) | — |
| VCL Grade A (AVBH) | 499/525 (95%) | — |

## Files

- `ppc_v11_spine_aware.ipynb` — Complete training, evaluation, and clinical validation notebook

## Usage

1. Update paths in the config cell:
   ```python
   DATA_ROOT   = Path("/path/to/Lumbar_Filtered_1037")
   PROJECT_DIR = Path("/path/to/ppc_network_v11")
   ```
2. Run all cells sequentially
3. Training takes ~300 epochs on NVIDIA A100-SXM4-80GB (~42s/epoch Phase 1, ~70s/epoch Phase 2)
4. Inference: ~5.4s per patient with TTA
5. Test results saved to `PROJECT_DIR/results/test_results_v11_tta.csv`
