# PPCNet-v8: Unified Pipeline

## Overview

PPCNet-v8 unifies the generator and GAN refiner into a single end-to-end training pipeline, retrained from scratch. This eliminates the two-stage training complexity of v7 while maintaining the gap-aware losses. The unified approach achieves a cleaner training trajectory and slightly improved metrics.

## Architecture

| Component | Details |
|-----------|---------|
| Encoder | 2× ResNet-18 (ImageNet pretrained, 1-ch adapted) |
| Feature Maps | 192-ch at 32×32 |
| Volume | 128-ch at 32³ |
| Query Points | 5,120 |
| Refinement Stages | 3 (independent weights) |
| MLP Dimension | 256 |
| Total Parameters | 21.8M |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=8e-5) |
| Scheduler | Linear warmup (10 epochs) + cosine decay |
| Generator Epochs | 250 |
| GAN Epochs | 150 |
| Batch Size | 2 |
| Losses | Chamfer + gap penalty + axial density + auxiliary occupancy + projection consistency + adversarial (unified) |

## Test Results (105 patients)

| CD (mm)↓ | F@1↑ | F@2↑ | F@5↑ | HD95 (mm)↓ |
|----------|------|------|------|------------|
| 2.452 | 0.115 | 0.512 | 0.942 | 6.506 |

## Files

- `ppc_training_v8_unified.ipynb` — Unified training and evaluation notebook

## Usage

1. Update paths in the config cell
2. Run all cells sequentially — both generator and GAN phases are in one notebook
3. Checkpoints saved to `PROJECT_DIR/checkpoints/`
