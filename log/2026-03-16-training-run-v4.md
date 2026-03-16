# Training Run v4 — Warmup + Cosine Annealing + Early Stopping — 2026-03-16

## Changes from Run 3

Replaced ReduceLROnPlateau with linear warmup + cosine annealing, added early stopping (commit `305d762`):

- **Linear warmup**: LR ramps from ~0 to 2e-4 over first 5 epochs
- **Cosine annealing**: Smooth decay from 2e-4 to 1e-6 over remaining 75 epochs
- **Early stopping**: Patience of 25 epochs (did not trigger — ran full 80)

All prior changes (contrast augmentation, calibration loss) retained.

## Setup

- Branch: `feature/training-pipeline`
- Command: `python -m training.train` (80 epochs, warmup=5, patience=25)
- Same data split (158 train, 26 val)
- Raw log: [2026-03-16-training-run-v4-raw.txt](2026-03-16-training-run-v4-raw.txt)

## Results

**Best validation Dice: 0.495 at epoch 70**

### Four-Way Comparison

| Metric | Run 1 (baseline) | Run 2 (+contrast) | Run 3 (+calibration) | Run 4 (+warmup/cosine) |
|--------|-------------------|--------------------|-----------------------|------------------------|
| Best val Dice | 0.503 (ep 38) | 0.480 (ep 45) | **0.509 (ep 78)** | 0.495 (ep 70) |
| Inference Dice | 0.521 ± 0.200 | 0.514 ± 0.158 | **0.538 ± 0.159** | 0.524 ± 0.178 |
| Precision | 0.467 | 0.526 | 0.535 | **0.538** |
| Recall | **0.624** | 0.523 | 0.571 | 0.551 |
| Threshold mean | 0.600 | 0.343 | 0.379 | **0.407** |
| Worst batch | 0.209 | 0.301 | **0.347** | 0.277 |
| Best batch | **0.873** | 0.737 | 0.835 | 0.831 |
| Grad explosions | 1 | 3 | 3 | **0** |

### Inference Per-Batch

| Batch | Dice | Threshold |
|-------|------|-----------|
| 0 | 0.476 | 0.90 |
| 1 | 0.536 | 0.30 |
| 2 | 0.476 | 0.20 |
| 3 | 0.576 | 0.30 |
| 4 | 0.831 | 0.50 |
| 5 | 0.277 | 0.10 |
| 6 | 0.496 | 0.55 |

## Observations

1. **Zero gradient explosions**: The warmup + cosine annealing completely eliminated inf gradient events. This is the biggest win — all three prior runs had at least one explosion.

2. **Best precision and threshold mean**: Precision of 0.538 is highest across all runs. Threshold mean of 0.407 is closest to the ideal 0.5.

3. **Didn't beat Run 3 on Dice**: 0.495 vs 0.509 val Dice, 0.524 vs 0.538 inference Dice. The cosine schedule may have decayed LR too aggressively — by epoch 75 the LR was already at 4e-6, leaving little room for late improvement.

4. **Worse worst-case**: 0.277 vs Run 3's 0.347. The model struggled more on the hardest batch.

5. **Slower early learning**: The warmup meant the model didn't start learning meaningfully until epoch 4-5, effectively losing a few epochs vs prior runs.

6. **Training thresholds very stable late**: Epochs 65-80 showed thresholds consistently in 0.30-0.47 range.

## Analysis

The warmup/cosine infrastructure is good (eliminates gradient explosions), but 80 epochs may not be enough with 5 warmup epochs eating into useful training time. The cosine schedule decays to near-zero LR by the end, so the model can't make late improvements the way ReduceLROnPlateau allowed in Run 3.

**Next step**: Try 120 epochs with the same setup. This gives the cosine schedule more room and early stopping (patience=25) will protect against wasting compute if the model plateaus.
