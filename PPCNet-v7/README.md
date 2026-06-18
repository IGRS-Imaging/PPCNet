# PPCNet-v7: Gap-Occupancy + GAN Refiner

## Overview

PPCNet-v7 introduces gap-aware reconstruction through a 96³ gap-occupancy volume, occupancy-gated query initialisation, and a family of gap-preserving losses. A post-hoc SN-GAN refiner is also trained to further sharpen the point distribution, though its metric improvement is marginal (CD: 2.485→2.484 mm).

## Architecture

| Component | Details |
|-----------|---------|
| Encoder | 2× ResNet-18 (ImageNet pretrained, 1-ch adapted) |
| Feature Maps | 192-ch at 32×32 |
| Volume | 128-ch at 32³ |
| Query Points | 5,120 |
| Refinement Stages | 3 (independent weights) |
| MLP Dimension | 256 |
| Gap Volume | 96³ non-dilated occupancy |
| Total Parameters | 21.8M (generator) + 0.08M (GAN refiner) |

## Training Configuration

**Phase 1 — Generator:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=8e-5) |
| Scheduler | Linear warmup (10 epochs) + cosine decay |
| Epochs | 250 |
| Batch Size | 2 |
| Losses | Chamfer + gap penalty + axial density + auxiliary occupancy + repulsion |
| Gap Ramp | Gap losses start at epoch 15 |

**Phase 2 — SN-GAN Refiner:**

| Parameter | Value |
|-----------|-------|
| Generator LR | 2e-5 |
| Discriminator LR | 1e-4 |
| Epochs | 150 |

## Test Results (105 patients)

| CD (mm)↓ | F@1↑ | F@2↑ | F@5↑ | HD95 (mm)↓ |
|----------|------|------|------|------------|
| 2.484 | 0.112 | 0.504 | 0.937 | 6.726 |

## Files

- `ppc_training_v7_gap_perfect.ipynb` — Generator training notebook
- `ppc_v7_gan_refiner.ipynb` — SN-GAN refiner training notebook

## Usage

1. Update paths in the config cell of both notebooks
2. **Train generator first** using `ppc_training_v7_gap_perfect.ipynb`
3. **Then train refiner** using `ppc_v7_gan_refiner.ipynb` (loads generator checkpoint)
4. Test evaluation is in the generator notebook (with refiner applied)
