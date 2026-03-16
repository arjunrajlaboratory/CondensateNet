# Training Run v4b — 120 Epochs with Warmup + Cosine + Early Stopping — 2026-03-16

## Changes from Run 4

Same config as Run 4 but with 120 epochs instead of 80, giving the cosine schedule more room before decaying to near-zero LR.

## Setup

- Branch: `feature/training-pipeline`
- Command: `python -m training.train` (120 epochs, warmup=5, patience=25, calibration_weight=0.5)
- Same data split (158 train, 26 val)
- Raw log: [2026-03-16-training-run-v4b-raw.txt](2026-03-16-training-run-v4b-raw.txt)

## Results

**Best validation Dice: 0.516 at epoch 69**
**Early stopping triggered at epoch 94** (25 epochs without improvement)

### Five-Run Comparison

| Metric | Run 1 | Run 2 | Run 3 | Run 4 | Run 4b |
|--------|-------|-------|-------|-------|--------|
| **Changes** | baseline | +contrast | +calibration | +warmup/cosine | +120 epochs |
| Best val Dice | 0.503 | 0.480 | 0.509 | 0.495 | **0.516** |
| Inference Dice | 0.521 ± 0.200 | 0.514 ± 0.158 | 0.538 ± 0.159 | 0.524 ± 0.178 | 0.534 ± **0.155** |
| Precision | 0.467 | 0.526 | 0.535 | **0.538** | 0.517 |
| Recall | **0.624** | 0.523 | 0.571 | 0.551 | 0.588 |
| Threshold mean | 0.600 | 0.343 | 0.379 | 0.407 | **0.479** |
| Worst batch | 0.209 | 0.301 | **0.347** | 0.277 | 0.288 |
| Best batch | **0.873** | 0.737 | 0.835 | 0.831 | 0.768 |
| Median Dice | 0.519 | — | 0.542 | 0.536 | **0.575** |
| Dice std | 0.200 | 0.158 | 0.159 | 0.178 | **0.155** |
| Epochs run | 80 | 80 | 80 | 80 | 94 (early stopped) |
| Grad explosions | 1 | 3 | 3 | 0 | 1 (ep 94) |

### Inference Per-Batch

| Batch | Dice | Threshold |
|-------|------|-----------|
| 0 | 0.519 | 0.90 |
| 1 | 0.575 | 0.20 |
| 2 | 0.459 | 0.30 |
| 3 | 0.581 | 0.30 |
| 4 | 0.768 | 0.60 |
| 5 | 0.288 | 0.10 |
| 6 | 0.445 | 0.95 |

### Best Dice Progression

| Epoch | Val Dice | Notes |
|-------|----------|-------|
| 5 | 0.220 | End of warmup |
| 12 | 0.384 | |
| 27 | 0.460 | |
| 48 | 0.487 | |
| 63 | 0.499 | |
| 66 | 0.507 | |
| 69 | 0.516 | **Best** |
| 94 | — | Early stopping triggered |

## Observations

1. **Best val Dice of all runs** (0.516). The extra epochs gave the cosine schedule enough room — best model came at epoch 69, well into the cosine decay phase where the LR was ~4e-5.

2. **Best threshold mean** (0.479) — nearest to ideal 0.5 of any run. A fixed threshold of 0.5 would work reasonably across all images.

3. **Lowest Dice variance** (0.155) and **best median Dice** (0.575) — the most consistent model across batches.

4. **Early stopping worked**: Stopped at epoch 94 after 25 epochs without improvement past epoch 69. Saved ~26 epochs of compute vs running the full 120.

5. **One gradient explosion**: At epoch 94 (grad norm 82.5), right at the end. This may have been a contributing factor to the stopping — the model was becoming unstable at very low LR.

6. **Run 3 still wins on worst-case** (0.347 vs 0.288) and **peak performance** (0.835 vs 0.768). The cosine schedule's gradual decay may produce a more "averaged" model that's consistent but lacks the sharp performance peaks that ReduceLROnPlateau's sudden LR drops can produce.

## Cumulative Improvements (Run 1 → Run 4b)

- Val Dice: 0.503 → 0.516 (+2.6%)
- Inference Dice: 0.521 → 0.534 (+2.5%)
- Dice std: 0.200 → 0.155 (-22.5%)
- Threshold mean: 0.600 → 0.479 (toward ideal 0.5)
- Worst-case: 0.209 → 0.288 (+38%)
- Median Dice: 0.519 → 0.575 (+10.8%)
