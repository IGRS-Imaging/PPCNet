# PPCNet-v4: Warm-Start + Chamfer-Ramp Curriculum

## Overview

PPCNet-v4 introduces warm-starting from the v8 checkpoint and a Chamfer-ramp curriculum that gradually increases the Chamfer loss weight during early training. It also upgrades the gap volume to 128³ resolution and adds a Z-axis gap loss. This version achieves the best HD95 (6.051 mm) among all fixed-backbone variants.

## Architecture

| Component | Details |
|-----------|---------|
| Encoder | 2× ResNet-18 (ImageNet pretrained, 1-ch adapted) |
| Feature Maps | 192-ch at 32×32 |
| Volume | 128-ch at 32³ |
| Query Points | 5,120 |
| Refinement Stages | 3 (independent weights) |
| MLP Dimension | 256 |
| Gap Volume | 128³ (upgraded from 96³) |
| Total Parameters | 21.8M |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=8e-5, wd=1e-5) |
| Scheduler | Linear warmup (10 epochs) + cosine decay |
| Epochs | 150 |
| Batch Size | 2 |
| Warm-Start | Loaded from v8 checkpoint |
| Chamfer Ramp | Starts at epoch 16, full weight at epoch 41 |
| Losses | Chamfer (ramped) + gap penalty + Z-gap + axial density + auxiliary occupancy |

## Test Results (105 patients)

| CD (mm)↓ | F@1↑ | F@2↑ | F@5↑ | HD95 (mm)↓ |
|----------|------|------|------|------------|
| 2.450 | 0.111 | 0.514 | 0.944 | 6.051 |

## Files

- `ppc_v4_hybrid_copy.ipynb` — Training and evaluation notebook

## Usage

1. **Requires v2 checkpoint** — train PPCNet-v2 first
2. Update paths in config cell, including path to v2 checkpoint
3. Run all cells sequentially
