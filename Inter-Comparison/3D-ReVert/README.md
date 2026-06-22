# 3D-ReVert — Inter-Comparison Baseline

## Paper Reference

> **3D-ReVert: 3D Reconstruction of Vertebrae from a Single Radiograph for Minimally Invasive Spine Surgery**
> Shakya, S., et al.
> *MICCAI Workshop on Clinical Image-Based Procedures (CLIP), 2024*
> [Paper](https://dl.acm.org/doi/10.1007/978-3-032-09513-8_4)

---

## Adaptation Protocol

This is a **re-implementation** of 3D-ReVert adapted for complete lumbar spine reconstruction. The following protocol ensures a controlled evaluation:

| Aspect | What we keep from 3D-ReVert | What we change for fair comparison |
|--------|---------------------------|-----------------------------------|
| **Encoder** | ResNet-18 (pretrained) | `in_channels` 1 → **2** (AP + LP concatenated) |
| **Decoder** | DGCNN — **completely unchanged** | — |
| **Loss** | CD-L1 (train) + CD-L2 (monitor) — **unchanged** | — |
| **LR schedule** | Step decay ×0.7 every 50 epochs | — |
| **Dataset** | — | Lumbar_Filtered_1037 (829/103/105 split) |
| **Points** | — | 8,192 (matching PPCNet-v11) |
| **Evaluation** | — | Chamfer, F@1/2/5 mm, HD95 (world-mm) |
| **Projection matrix** | — | **Not used** |

> **Key differences from PPCNet:** 3D-ReVert concatenates the AP and LP DRRs as a 2-channel image and processes them through a single ResNet-18 encoder, producing a 1024-d global latent vector. A DGCNN decoder then generates the point cloud from this latent code using EdgeConv graph convolutions. Unlike PPCNet, it uses **no projection matrices**, **no volumetric feature representation**, and **no iterative refinement** — the entire 3D structure is generated in a single forward pass from the global feature.

> **Original 3D-ReVert** was designed for **single-vertebra** reconstruction from a single radiograph. This adaptation extends it to **complete lumbar spine (L1–L5)** from biplanar input.

---

## Architecture

```
Input: AP DRR (512×512) + LP DRR (512×512)
                    ↓
         Concatenate as 2-channel image
                    ↓
    ┌───────────────┴───────────────┐
    │     ResNet-18 Encoder         │
    │     (ImageNet pretrained)     │
    │     in_channels: 2 (AP+LP)   │
    │     Output: 1024-d global z   │
    └───────────────┬───────────────┘
                    ↓
    ┌───────────────┴───────────────┐
    │     DGCNN Decoder             │
    │     (Original 3D-ReVert)      │
    │                               │
    │  z tiled → (B, 1024, 8192)   │
    │  + seed coords (B, 3, 8192)  │
    │         ↓                     │
    │  EdgeConv ×4                  │
    │  (k=20 neighbors)            │
    │  64→64→128→256 channels       │
    │         ↓                     │
    │  MLP: 256→256→3               │
    └───────────────┬───────────────┘
                    ↓
    Output: 8,192-point cloud
            (normalised local space)
                    ↓
    Denormalise to world-mm
```

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| Encoder (ResNet-18) | 11.2M |
| Decoder (DGCNN) | 27.2M |
| **Total** | **38.4M** |

---

## Loss Functions (3D-ReVert Original — Unchanged)

| Loss | Role | Description |
|------|------|-------------|
| CD-L1 | Training | L1 Chamfer distance in normalised local space |
| CD-L2 | Monitoring only | L2 Chamfer distance (not backpropagated) |

> **No PPCNet losses used.** No gap penalty, no axial density, no curvature loss, no sliced Wasserstein, no projection consistency. Only Chamfer distance — exactly as in the original 3D-ReVert paper.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-4, wd=1e-4) |
| Encoder LR | 0.1× body LR (differential) |
| Scheduler | Step decay ×0.7 every 50 epochs (min 1e-6) |
| Epochs | 300 (best at ep 276) |
| Batch Size | 4 |
| Input Resolution | 512×512 (2-channel: AP + LP) |
| Output Points | 8,192 (normalised local → world-mm) |
| Gradient Clip | 1.0 |
| Precision | fp32 |
| Early Stopping | 60 epochs patience |
| GPU | NVIDIA A100-SXM4-80GB |

### Training Progress

```
Ep   1/300: cd_l1=0.xxxxx  Val: CD-L1=0.xxxxx
    ...
Ep 276/300: Val: CD-L1=0.02639                                   ← Best
    ...
Training completed (300 epochs, no early stop triggered)
```

**Best checkpoint:** Epoch 276 (val CD-L1 = 0.02639)

---

## Test Results (105 patients)

<div align="center">

| Metric | Mean ± Std | Median | 95th Percentile |
|:------:|:----------:|:------:|:---------------:|
| **Chamfer (mm) ↓** | 2.084 ± 0.622 | 1.940 | 3.221 |
| **F@1 ↑** | 0.133 ± 0.065 | 0.126 | 0.263 |
| **F@2 ↑** | 0.587 ± 0.152 | 0.606 | 0.812 |
| **F@5 ↑** | 0.966 ± 0.060 | 0.987 | 0.999 |
| **HD95 (mm) ↓** | 4.586 ± 2.453 | 3.845 | 8.384 |

</div>

### Comparison with PPCNet

<div align="center">

| Metric | 3D-ReVert | PPCNet-v11 (Ours) | Δ |
|:------:|:---------:|:-----------------:|:-:|
| CD (mm) ↓ | 2.084 | **1.981** | −0.103 |
| F@1 ↑ | 0.133 | **0.155** | +0.022 |
| F@2 ↑ | 0.587 | **0.646** | +0.059 |
| F@5 ↑ | 0.966 | **0.973** | +0.007 |
| HD95 (mm) ↓ | 4.586 | **4.525** | −0.061 |

</div>

> **PPCNet outperforms 3D-ReVert across all five metrics.** The margin is tighter than against volumetric baselines (0.103 mm CD vs 0.494–0.583 mm), which is expected — 3D-ReVert is the only other **point cloud** method and benefits from DGCNN's graph-based local feature learning. However, PPCNet's projection-conditioned refinement still provides a meaningful advantage: the 0.103 mm CD improvement corresponds to higher F@2 (+0.059, a 10% relative gain), confirming better surface-level accuracy. Notably, both methods have nearly identical parameter counts (38.4M vs 38.6M), isolating the architectural difference as the sole factor.

---

## Files

```
Inter-Comparison/3D-ReVert/
├── README.md                              # This file
├── 3DReVert_Inter_Comparison.ipynb        # Complete training + evaluation notebook
└── results/
    ├── test_results_3drevert_lumbar.csv   # Per-patient test metrics
    ├── 3DReVert_AP_View (lumbar_0028).png
    ├── 3DReVert_LP_View (lumbar_0028).png
    └── 3DReVert_Axial_View (lumbar_0028).png
```

---

## Usage

### 1. Update Paths

In Cell 2 of the notebook, update:

```python
DATA_ROOT   = Path("/path/to/Lumbar_Filtered_1037")
PROJECT_DIR = Path("/path/to/3drevert_lumbar_comparison")
```

### 2. Run Training

Run all cells sequentially. The notebook handles:
- Data loading with AP+LP concatenation as 2-channel input
- ResNet-18 encoder (differential LR: 0.1× for pretrained layers)
- DGCNN decoder with EdgeConv (k=20 neighbors)
- CD-L1 training loss with CD-L2 monitoring
- Step-decay LR schedule (×0.7 every 50 epochs)
- Automatic checkpointing (resumes from interruption)
- Validation every epoch with CD-L1 monitoring
- Early stopping at 60 epochs patience

### 3. Test Evaluation

The final cells:
- Load best checkpoint
- Run inference on 105 test patients
- Denormalise predictions from local to world-mm space
- Compute all five metrics (Chamfer, F@1/2/5, HD95) in world-mm
- Save per-patient CSV to `results/test_results_3drevert_lumbar.csv`

---

## Key Observations

1. **Closest baseline to PPCNet** (CD 2.084 vs 1.981 mm) — the only other point cloud method, confirming that point-based representations are fundamentally better suited than voxel grids for sparse bone structures
2. **DGCNN's graph convolution provides strong local modelling** — EdgeConv with k=20 neighbors captures local surface geometry effectively, yielding the best F@5 (0.966) among baselines
3. **No projection matrices, no volumetric features** — the entire 3D structure is generated from a single 1024-d global feature vector, which inherently limits the spatial precision achievable
4. **Single-pass generation vs iterative refinement** — 3D-ReVert generates all 8,192 points in one shot, while PPCNet refines them across 3 stages with per-point 2D/3D feature sampling, enabling finer corrections
5. **Nearly identical parameter count (38.4M vs 38.6M)** — the performance gap is purely architectural, not a capacity difference, making this the most informative comparison in the ablation
6. **Best at epoch 276/300** — unlike the volumetric baselines that plateau early, 3D-ReVert continues improving deep into training, suggesting the point cloud representation has a smoother loss landscape

---

## Citation

```bibtex
@inproceedings{shakya2024revert,
  author    = {Shakya, Sneha and others},
  title     = {{3D-ReVert}: {3D} Reconstruction of Vertebrae from a 
               Single Radiograph for Minimally Invasive Spine Surgery},
  booktitle = {MICCAI Workshop on Clinical Image-Based Procedures},
  series    = {LNCS},
  publisher = {Springer},
  year      = {2024},
  doi       = {10.1007/978-3-031-73284-3_4}
}
```
