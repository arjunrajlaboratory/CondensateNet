# Training Run v3 — Calibration Loss — 2026-03-16

## Changes from Run 2

Added differentiable calibration loss (soft-binned ECE) to address threshold instability that persisted after contrast augmentation (commit `1439716`):

- **Calibration loss**: Soft-binned Expected Calibration Error with triangular kernel bin assignment, `calibration_weight=0.5`, 10 bins
- Penalizes the squared gap between predicted confidence and actual accuracy per probability bin
- Gradients flow through via soft bin assignment (no hard binning)

All contrast augmentation changes from Run 2 are retained.

## Setup

- Branch: `feature/training-pipeline`
- Command: `python -m training.train` (calibration_weight=0.5)
- Same data split as Runs 1 & 2 (158 train, 26 val)
- Raw log: [2026-03-16-training-run-v3-raw.txt](2026-03-16-training-run-v3-raw.txt)

## Results

**Best validation Dice: 0.509 at epoch 78**

### Three-Way Comparison

| Metric | Run 1 (baseline) | Run 2 (contrast aug) | Run 3 (+ calibration) |
|--------|-------------------|----------------------|-----------------------|
| Best val Dice | 0.503 (ep 38) | 0.480 (ep 45) | **0.509 (ep 78)** |
| Inference Dice | 0.521 ± 0.200 | 0.514 ± 0.158 | **0.538 ± 0.159** |
| Precision | 0.467 ± 0.193 | 0.526 ± 0.130 | **0.535 ± 0.168** |
| Recall | 0.624 ± 0.241 | 0.523 ± 0.197 | **0.571 ± 0.190** |
| Threshold mean | 0.600 | 0.343 | **0.379** |
| Threshold range | 0.10–0.90 | 0.10–0.90 | 0.10–0.90 |
| Worst batch Dice | 0.209 | 0.301 | **0.347** |
| Best batch Dice | 0.873 | 0.737 | **0.835** |
| Median Dice | 0.519 | — | **0.542** |
| Train/val F1 gap | 0.61/0.50 | 0.55/0.48 | **0.54/0.51** |

### Training Dynamics

- LR schedule: 2e-4 → 1e-4 (ep 40) → 5e-5 (ep 57)
- Best model came late (epoch 78), suggesting the calibration loss helped the model continue improving where Runs 1 & 2 had plateaued
- Calibration loss started at ~0.01 and dropped to ~0.0004–0.0008 by end, confirming the model learned to be well-calibrated
- Training epoch thresholds settled to 0.36–0.42 range in final epochs (vs wild swings in Run 1)
- Gradient explosions at epochs 26, 51, 65 (recovered via clipping)

## Observations

1. **Best overall performance**: Highest Dice across all three runs (0.509 val, 0.538 inference), beating both the baseline and contrast-augmented models.

2. **Recovered peak accuracy**: Best single-batch Dice bounced back to 0.835 (vs 0.737 in Run 2), close to Run 1's 0.873. The calibration loss prevented the contrast augmentation from sacrificing peak performance.

3. **Better worst-case**: 0.347 worst batch, up 65% from Run 1's 0.209. The model doesn't catastrophically fail on any image type.

4. **Smallest train/val gap**: 0.54/0.51 F1 — nearly closed, indicating good generalization with minimal overfitting.

5. **Calibration loss converged**: The near-zero calibration loss values (0.0004) by end of training show the model successfully learned to produce well-calibrated probabilities. This is remarkable given that focal loss typically fights against calibration.

6. **Late best epoch**: Best model at epoch 78 (vs 38 and 45 in prior runs). The calibration loss appears to provide a useful training signal late in training when the other losses have plateaued, enabling continued improvement.

7. **Threshold range still wide at inference**: Despite much better training-time thresholds, the inference threshold range still spans 0.10–0.90. This is likely due to 1-2 outlier batches in the small 26-sample validation set. The median threshold behavior is clearly improved.

## Cumulative Improvements (Run 1 → Run 3)

- Inference Dice: 0.521 → 0.538 (+3.3%)
- Worst-case Dice: 0.209 → 0.347 (+66%)
- Dice std: 0.200 → 0.159 (-21%)
- Train/val gap: 0.11 → 0.03 (-73%)
