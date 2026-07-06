# PPCNet-v5: Stability-Focused with Phased Losses

## Overview

PPCNet-v5 introduces a stability-focused training regime with three-phase loss scheduling. Phase 1 establishes core geometry, Phase 2 adds distribution-matching losses (sliced Wasserstein, projection consistency, smoothness, spacing, boundary), and Phase 3 introduces inter-vertebral gap and vertebral separation losses. This careful phasing produces a substantial jump from 2.450 to 1.999 mm Chamfer distance — the single largest improvement in the ablation study.

## Architecture

| Component | Details |
|-----------|---------|
| Encoder | 2× ResNet-18 (ImageNet pretrained, 1-ch adapted) |
| Feature Maps | 192-ch at 32×32 |
| Volume | 128-ch at 32³ |
| Auxiliary Volume | 48³ occupancy |
| Query Points | 5,120 |
| Refinement Stages | 3 (independent weights) |
| MLP Dimension | 256 |
| Total Parameters | 21.8M |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-4) |
| Scheduler | Linear warmup (10 epochs) + cosine decay |
| Epochs | 300 |
| Batch Size | 2 |
| Encoder Frozen | Epochs 1–5 |
| Phase 1 (epoch 1+) | Chamfer + gap penalty + axial density + auxiliary occupancy + extent match |
| Phase 2 (epoch 5+) | + Sliced Wasserstein + projection consistency + smoothness + spacing + boundary |
| Phase 3 (epoch 15+) | + Inter-vertebral gap + vertebral separation + gap width |
| Early Stopping | 60 epochs patience |

## Test Results (105 patients)

| CD (mm)↓ | F@1↑ | F@2↑ | F@5↑ | HD95 (mm)↓ |
|----------|------|------|------|------------|
| 1.999 | 0.144 | 0.659 | 0.972 | 4.474 |

## Files

- `ppc_v5_stable.ipynb` — Training and evaluation notebook

## Usage

1. Update paths in the config cell:
   ```python
   DATA_ROOT   = Path("/path/to/Lumbar_Filtered_1037")
   PROJECT_DIR = Path("/path/to/ppc_network_v10")
   ```
2. Run all cells sequentially — trained from scratch (no warm-start needed)
3. Three-phase loss scheduling is handled automatically
