# PPCNet-v1: Baseline

## Overview

PPCNet-v1 is the baseline version establishing the core projection-conditioned refinement architecture. It introduces the fundamental pipeline: dual 2D encoders → 3D feature lifting → biplanar fusion → coarse U-Net → occupancy-gated query initialisation → iterative projection-conditioned refinement.

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
| Optimizer | AdamW (lr=1e-4, wd=1e-5) |
| Scheduler | Linear warmup (8 epochs) + cosine decay |
| Epochs | 200 |
| Batch Size | 2 |
| Precision | fp32 |
| Losses | Chamfer (symmetric) + projection consistency (warm-down 0.10→0.02 over 30 epochs) + auxiliary occupancy (BCE+Dice) |

## Test Results (105 patients)

| CD (mm)↓ | F@1↑ | F@2↑ | F@5↑ | HD95 (mm)↓ |
|----------|------|------|------|------------|
| 2.672 | 0.091 | 0.458 | 0.933 | 6.364 |

## Files

- `ppc_training_v1_fixed.ipynb` — Complete training and evaluation notebook

## Usage

1. Update paths in the config cell:
   ```python
   DATA_ROOT   = Path("/path/to/Lumbar_Filtered_1037")
   PROJECT_DIR = Path("/path/to/ppc_network_v1")
   ```
2. Run all cells sequentially
3. Training checkpoints are saved to `PROJECT_DIR/checkpoints/`
4. Test results are saved to `PROJECT_DIR/results/`
