# Swin-X2S — Inter-Comparison Baseline

## Paper Reference

> **Swin-X2S: Reconstructing 3D Shape from 2D Biplanar X-ray with Swin Transformers**
> Liu, K., et al.
> *arXiv preprint arXiv:2501.05961, 2025*
> [Paper](https://doi.org/10.48550/arXiv.2501.05961)

---

## Adaptation Protocol

This is a **re-implementation** of Swin-X2S for fair comparison with PPCNet. The following protocol ensures a controlled evaluation:

| Aspect | What we keep from Swin-X2S | What we keep from PPCNet |
|--------|---------------------------|-------------------------|
| **Architecture** | Dual Swin-T encoders, dimension expanding, cross-attention, 3D U-decoder | — |
| **Loss functions** | DiceCE + KL cross-view consistency | — |
| **Optimizer** | AdamW (lr=3e-5, wd=0.5), warmup + cosine decay | — |
| **Dataset** | — | Lumbar_Filtered_1037 (829/103/105 split) |
| **Coordinate system** | — | `pts_to_local` / `local_to_world` / `compute_scale` |
| **GT representation** | 64³ occupancy volume from GT point cloud | GT point cloud for evaluation |
| **Evaluation metrics** | — | Chamfer, F@1/2/5 mm, HD95 (world-mm) |
| **Output points** | — | 8,192 (extracted from predicted volume) |

> **Key difference from PPCNet:** Swin-X2S uses two independent Swin Transformer encoders to extract hierarchical 2D features, expands them into 3D via a dimension expanding module, and fuses the two views through cross-attention at the bottleneck. It does **not** use calibrated projection matrices — view correspondence is learned entirely through attention.

---

## Architecture

```
Input: AP DRR (224×224) + LP DRR (224×224)
                    ↓
    ┌───────────────┴───────────────┐
    │   Swin-T Encoder ×2           │
    │   (ImageNet pretrained)       │
    │   4 stages: 96→192→384→768    │
    │   Resolutions: 56→28→14→7     │
    └───────────────┬───────────────┘
                    ↓
    ┌───────────────┴───────────────┐
    │   Dimension Expanding Module  │
    │   2D features → 3D volumes    │
    │   Per-stage: resize + expand  │
    │   + Conv3D refinement         │
    └───────────────┬───────────────┘
                    ↓
    ┌───────────────┴───────────────┐
    │   Cross-Attention Fusion      │
    │   AP queries × LP keys/values │
    │   + LP queries × AP keys/vals │
    │   At bottleneck (4³, 768-ch)  │
    └───────────────┬───────────────┘
                    ↓
    ┌───────────────┴───────────────┐
    │   3D U-Decoder                │
    │   Skip connections from       │
    │   dimension-expanded features │
    │   4³→8³→16³→32³→64³           │
    └───────────────┬───────────────┘
                    ↓
    Output: 64³ occupancy volume
                    ↓
    Point cloud extraction (8,192 pts)
```

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| Swin-X2S (2× Swin-T + DimExpand + CrossAttn + Decoder) | **67.3M** |

---

## Loss Functions (Swin-X2S Original)

| Loss | Weight | Description |
|------|--------|-------------|
| DiceCE | dice=1.0, ce=1.0 | Combined Dice + cross-entropy loss |
| KL Cross-View | λ = 0.1 | KL divergence between single-view predictions for consistency |

> **No PPCNet losses used.** No Chamfer, no gap penalty, no axial density, no curvature loss.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=3e-5, wd=0.5) |
| Scheduler | Linear warmup (20 epochs) + cosine decay |
| Epochs | 200 (early stopped at 154) |
| Batch Size | 2 |
| Input Resolution | 224×224 (Swin-T default, ImageNet pretrained) |
| Output Resolution | 64³ voxel grid |
| Gradient Clip | 1.0 |
| Precision | fp32 |
| GPU | NVIDIA A100-SXM4-80GB |

### Training Progress

```
Ep   1/200: L=0.xxx  Val: Chamfer=8.xxx mm
    ...
Ep  94/200: Val: Chamfer=2.307 mm                                ← Best
    ...
Ep 154/200: Early stop (60 epochs without improvement)
```

**Best checkpoint:** Epoch 94 (val Chamfer = 2.307 mm)

---

## Test Results (105 patients)

<div align="center">

| Metric | Mean ± Std | Median | 95th Percentile |
|:------:|:----------:|:------:|:---------------:|
| **Chamfer (mm) ↓** | 2.475 ± 1.053 | 2.295 | 3.129 |
| **F@1 ↑** | 0.081 ± 0.031 | 0.077 | 0.134 |
| **F@2 ↑** | 0.449 ± 0.106 | 0.469 | 0.606 |
| **F@5 ↑** | 0.937 ± 0.073 | 0.955 | 0.980 |
| **HD95 (mm) ↓** | 6.576 ± 5.426 | 5.587 | 8.029 |

</div>

### Comparison with PPCNet

<div align="center">

| Metric | Swin-X2S | PPCNet-v11 (Ours) | Δ |
|:------:|:--------:|:-----------------:|:-:|
| CD (mm) ↓ | 2.475 | **1.981** | −0.494 |
| F@1 ↑ | 0.081 | **0.155** | +0.074 |
| F@2 ↑ | 0.449 | **0.646** | +0.197 |
| F@5 ↑ | 0.937 | **0.973** | +0.036 |
| HD95 (mm) ↓ | 6.576 | **4.525** | −2.051 |

</div>

> **PPCNet outperforms Swin-X2S across all five metrics**, reducing Chamfer distance by 0.494 mm (20.0%) and HD95 by 2.051 mm (31.2%). Despite leveraging Swin Transformers with hierarchical attention and cross-view KL consistency — the most sophisticated implicit fusion mechanism among all baselines — Swin-X2S still cannot match explicit projection conditioning. The cross-attention at the bottleneck operates on learned features without geometric grounding, whereas PPCNet's projection step directly maps each 3D query to its corresponding 2D locations via the calibrated matrices.

---

## Files

```
Inter-Comparison/Swin-X2S/
├── README.md                              # This file
├── SwinX2S_Inter_Comparison.ipynb         # Complete training + evaluation notebook
└── results/
    ├── test_results_swinx2s.csv           # Per-patient test metrics
    ├── SwinX2S_AP_View (lumbar_0028).png
    ├── SwinX2S_LP_View (lumbar_0028).png
    └── SwinX2S_Axial_View (lumbar_0028).png
```

---

## Usage

### 1. Update Paths

In Cell 1 of the notebook, update:

```python
DATA_ROOT   = Path("/path/to/Lumbar_Filtered_1037")
PROJECT_DIR = Path("/path/to/inter_comparison_swinx2s")
```

### 2. Run Training

Run all cells sequentially. The notebook handles:
- Data loading with GT volume generation (64³)
- Dual Swin-T encoder training with DiceCE + KL losses
- 20-epoch linear warmup followed by cosine decay
- Automatic checkpointing (resumes from interruption)
- Validation every epoch with Chamfer distance monitoring
- Early stopping at 60 epochs patience

### 3. Test Evaluation

The final cells:
- Load best checkpoint
- Run inference on 105 test patients
- Extract 8,192-point clouds from predicted volumes
- Compute all five metrics in world-mm space
- Save per-patient CSV to `results/test_results_swinx2s.csv`

---

## Key Observations

1. **Best validation among volumetric baselines** (2.307 mm) — Swin Transformer's hierarchical attention captures multi-scale features better than DenseNet (X2CT-GAN) or VGG (BX2S-Net), but this advantage does not fully transfer to test performance (2.475 mm test vs 2.489 X2CT-GAN)
2. **Cross-attention learns approximate correspondence** — the KL cross-view loss encourages single-view predictions to agree, providing a soft geometric constraint, but it remains implicit compared to PPCNet's explicit projection
3. **67.3M parameters** — larger than PPCNet (38.6M) and BX2S-Net (23.2M) due to the dual Swin-T backbones, yet the additional capacity does not translate to proportional accuracy gains
4. **Heavy weight decay (0.5)** — required to prevent overfitting with the large Swin-T encoders, highlighting that the transformer's capacity is not well-matched to the volumetric reconstruction task without geometric guidance
5. **224×224 input resolution** — the highest among all baselines (X2CT-GAN: 128, BX2S-Net: 64), yet still outputs at 64³, suggesting that higher 2D resolution alone cannot overcome the volumetric bottleneck

---

## Citation

```bibtex
@article{liu2025swinx2s,
  author  = {Liu, Kuan and others},
  title   = {{Swin-X2S}: Reconstructing {3D} Shape from {2D} Biplanar 
             {X}-ray with {S}win Transformers},
  journal = {arXiv preprint arXiv:2501.05961},
  year    = {2025},
  doi     = {10.48550/arXiv.2501.05961}
}
```
