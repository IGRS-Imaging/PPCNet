# X2CT-GAN — Inter-Comparison Baseline

## Paper Reference

> **X2CT-GAN: Reconstructing CT from Biplanar X-rays with Generative Adversarial Networks**
> Ying, X., Guo, H., Ma, K., Wu, J., Weng, Z., Zheng, Y.
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019*
> [Paper](https://doi.org/10.1109/CVPR.2019.01087) | [Original Code](https://github.com/kylekma/X2CT)

---

## Adaptation Protocol

This is a **re-implementation** of X2CT-GAN for fair comparison with PPCNet. The following protocol ensures a controlled evaluation:

| Aspect | What we keep from X2CT-GAN | What we keep from PPCNet |
|--------|---------------------------|-------------------------|
| **Architecture** | DenseNet encoder, Connection-A/B/C modules, 3D decoder, PatchGAN discriminator | — |
| **Loss functions** | MSE reconstruction + LSGAN adversarial + projection L1 | — |
| **Optimizer** | Adam (lr=2e-4, β₁=0.5, β₂=0.99), linear decay after ep 50 | — |
| **Dataset** | — | Lumbar_Filtered_1037 (829/103/105 split) |
| **Coordinate system** | — | `pts_to_local` / `local_to_world` / `compute_scale` |
| **GT representation** | 64³ occupancy volume from GT point cloud | GT point cloud for evaluation |
| **Evaluation metrics** | — | Chamfer, F@1/2/5 mm, HD95 (world-mm) |
| **Output points** | — | 8,192 (extracted from predicted volume) |

> **Key difference from PPCNet:** X2CT-GAN outputs a 64³ voxel grid via a volumetric GAN, from which a point cloud is extracted for evaluation. It does **not** use calibrated projection matrices — view fusion happens through learned DenseNet encoders and Connection modules.

---

## Architecture

```
Input: AP DRR (128×128) + LP DRR (128×128)
                    ↓
    ┌───────────────┴───────────────┐
    │  DenseNet-style 2D Encoder    │ ×2 (AP + LP)
    │  (5 DenseBlocks, InstanceNorm)│
    │  32→64→128→256→512 channels   │
    └───────────────┬───────────────┘
                    ↓
         Connection-A (1D→3D)
         Reshape to 4×4×4 volume
                    ↓
         Connection-B (3D refine)
         Conv3D blocks, upsample
                    ↓
         Connection-C (feature merge)
         Fuse AP + LP volumes
                    ↓
         3D Decoder (progressive)
         4³ → 8³ → 16³ → 32³ → 64³
                    ↓
    Output: 64³ occupancy volume
                    ↓
    Point cloud extraction (8,192 pts)
```

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| Generator (DenseNet + 3D Decoder) | **344.1M** |
| Discriminator (PatchGAN) | **11.1M** |
| **Total** | **355.2M** |

---

## Loss Functions (X2CT-GAN Original)

| Loss | Weight | Description |
|------|--------|-------------|
| MSE Reconstruction | λ = 10.0 | Voxel-wise MSE between predicted and GT volume |
| LSGAN Adversarial | λ = 0.1 | Least-squares GAN loss for realistic volume generation |
| Projection L1 | λ = 10.0 | L1 distance on orthogonal projections (sum along each axis) |

> **No PPCNet losses used.** No Chamfer, no gap penalty, no axial density, no curvature loss.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer (G) | Adam (lr=2e-4, β₁=0.5, β₂=0.99) |
| Optimizer (D) | Adam (lr=2e-4, β₁=0.5, β₂=0.99) |
| LR Schedule | Constant for 50 epochs, linear decay to 0 over remaining 50 |
| Epochs | 100 |
| Batch Size | 1 |
| Input Resolution | 128×128 (DRRs resized from 512) |
| Output Resolution | 64³ voxel grid |
| Gradient Clip | 1.0 |
| Precision | fp32 |
| GPU | NVIDIA A100-SXM4-80GB (MIG 1g.10gb) |

### Training Progress

```
Ep   1/100: G=168.882  recon=1.316  adv=0.0537  proj=167.512  D=0.0101
Ep   2/100: G=84.159   recon=0.601  adv=0.0509  proj=83.507   D=0.0024
    ...
Ep  57/100: G=4.795    recon=0.091  adv=0.0501  proj=4.654    D=0.0004  ← Best
    ...
Ep 100/100: Training complete
```

**Best checkpoint:** Epoch 57 (val Chamfer = 2.324 mm)

---

## Test Results (105 patients)

<div align="center">

| Metric | Mean ± Std | Median | 95th Percentile |
|:------:|:----------:|:------:|:---------------:|
| **Chamfer (mm) ↓** | 2.489 ± 0.924 | 2.354 | 3.067 |
| **F@1 ↑** | 0.079 ± 0.031 | 0.074 | 0.134 |
| **F@2 ↑** | 0.444 ± 0.107 | 0.442 | 0.605 |
| **F@5 ↑** | 0.933 ± 0.074 | 0.949 | 0.979 |
| **HD95 (mm) ↓** | 6.624 ± 4.651 | 5.728 | 8.387 |

</div>

### Comparison with PPCNet

<div align="center">

| Metric | X2CT-GAN | PPCNet-v11 (Ours) | Δ |
|:------:|:--------:|:-----------------:|:-:|
| CD (mm) ↓ | 2.489 | **1.981** | −0.508 |
| F@1 ↑ | 0.079 | **0.155** | +0.076 |
| F@2 ↑ | 0.444 | **0.646** | +0.202 |
| F@5 ↑ | 0.933 | **0.973** | +0.040 |
| HD95 (mm) ↓ | 6.624 | **4.525** | −2.099 |

</div>

> **PPCNet outperforms X2CT-GAN across all five metrics**, reducing Chamfer distance by 0.508 mm (20.4%) and HD95 by 2.099 mm (31.7%), despite having 9× fewer parameters (38.6M vs 344.1M). This demonstrates that projection-conditioned point cloud refinement provides a stronger inductive bias than volumetric GAN-based reconstruction with dense encoders.

---

## Files

```
Inter-Comparison/X2CT-GAN/
├── README.md                              # This file
├── x2ctgan_inter_comparison.ipynb         # Complete training + evaluation notebook
└── results/
    └── test_results_x2ctgan.csv, AP, LP & Axial - Views (lumbar_0028 test set)  # Per-patient test metrics
```

---

## Usage

### 1. Update Paths

In Cell 1 of the notebook, update:

```python
DATA_ROOT   = Path("./data/Lumbar_Filtered_1037")
PROJECT_DIR = Path("./inter_comparison_x2ctgan")
```

### 2. Run Training

Run all cells sequentially. The notebook handles:
- Data loading with GT volume generation (64³)
- Generator + Discriminator training with X2CT-GAN losses
- Automatic checkpointing (resumes from interruption)
- Validation every epoch with Chamfer distance monitoring
- Early stopping at 60 epochs patience

### 3. Test Evaluation

The final cells:
- Load best checkpoint
- Run inference on 105 test patients
- Extract 8,192-point clouds from predicted volumes
- Compute all five metrics in world-mm space
- Save per-patient CSV to `results/test_results_x2ctgan.csv`

---

## Key Observations

1. **Projection loss dominates early training** (167.5 at ep 1 → 4.6 at ep 57) — this is expected as the orthogonal volume projections start far from the ground truth
2. **Best performance at epoch 57/100** — the model peaks mid-training then slightly degrades, consistent with GAN training instability
3. **344.1M parameters yet 2.489 mm Chamfer** — the dense volumetric representation and lack of projection matrix conditioning limit surface accuracy despite massive model capacity
4. **64³ resolution bottleneck** — each voxel spans ~2–3 mm for a lumbar spine, fundamentally limiting fine structure representation (e.g., inter-vertebral gaps)

---

## Citation

```bibtex
@inproceedings{ying2019x2ctgan,
  author    = {Ying, Xingde and Guo, Heng and Ma, Kai and Wu, Jian 
               and Weng, Zhengxin and Zheng, Yefeng},
  title     = {{X2CT-GAN}: Reconstructing {CT} from Biplanar {X}-rays 
               with Generative Adversarial Networks},
  booktitle = {Proc. IEEE/CVF CVPR},
  pages     = {10619--10628},
  year      = {2019},
  doi       = {10.1109/CVPR.2019.01087}
}
```
