# Training Run v2 — Stronger Contrast Augmentation — 2026-03-16

## Changes from Run 1

Added stronger contrast augmentation to address threshold instability / poor confidence calibration observed in Run 1 (commit `d66f437`):

- **Wider brightness/contrast ranges** across all presets (e.g. heavy: ±0.12 → ±0.35 brightness, ±0.30 → ±0.45 contrast)
- **Wider gamma range** (e.g. heavy: 80-120 → 50-200)
- **Intensity scaling** (new): randomly multiply pixel values by 0.3x–2.0x to simulate low/high contrast conditions
- **CLAHE** (new): adaptive histogram equalization to simulate auto-contrast adjustment

## Setup

- Branch: `feature/training-pipeline`
- Command: `python -m training.train` (all defaults)
- Same data split as Run 1 (158 train, 26 val)
- Raw log: [2026-03-16-training-run-v2-raw.txt](2026-03-16-training-run-v2-raw.txt)

## Results

**Best validation Dice: 0.480 at epoch 45**

### Comparison with Run 1

| Metric | Run 1 | Run 2 | Change |
|--------|-------|-------|--------|
| Best val Dice | 0.503 (ep 38) | 0.480 (ep 45) | -0.023 |
| Inference Dice | 0.521 ± 0.200 | 0.514 ± 0.158 | Similar mean, lower variance |
| Precision | 0.467 ± 0.193 | 0.526 ± 0.130 | +0.06, much tighter |
| Recall | 0.624 ± 0.241 | 0.523 ± 0.197 | -0.10, tighter |
| Threshold mean | 0.600 | 0.343 | Lower |
| Threshold range | 0.10–0.90 | 0.10–0.90 | Same |
| Dice std | 0.200 | 0.158 | 21% less variance |
| Best batch Dice | 0.873 | 0.737 | Lower peak |
| Worst batch Dice | 0.209 | 0.301 | Better floor |
| Train/val F1 gap | 0.61 / 0.50 | 0.55 / 0.48 | Smaller gap |

### Inference Per-Batch

| Batch | Dice | Threshold |
|-------|------|-----------|
| 0 | 0.468 | 0.90 |
| 1 | 0.544 | 0.20 |
| 2 | 0.500 | 0.10 |
| 3 | 0.543 | 0.20 |
| 4 | 0.737 | 0.60 |
| 5 | 0.508 | 0.10 |
| 6 | 0.301 | 0.30 |

### Training Dynamics

- LR schedule: 2e-4 → 1e-4 (ep 40) → 5e-5 (ep 57) → 2.5e-5 (ep 73)
- Gradient explosions at epochs 26, 51, 73 (recovered via clipping)
- Additional large gradient warnings at epochs 65, 78, 79

## Observations

1. **Reduced overfitting**: The train/val gap shrank. Train F1 peaked lower (0.55 vs 0.61) but val performance held steady, confirming the augmentation acts as effective regularization.

2. **More consistent predictions**: Dice std dropped 21% (0.200 → 0.158). The worst-case batch improved significantly (0.209 → 0.301). The model is more robust across different image types.

3. **Better precision**: Precision improved from 0.467 to 0.526 with tighter std (0.193 → 0.130), meaning fewer false positives.

4. **Threshold still unstable**: Despite the improvements, the optimal threshold range is still 0.10–0.90. The contrast augmentation helped generalization but didn't fully solve the calibration problem. This confirms that a calibration loss term is needed to address the root cause (focal loss distorting the probability scale).

5. **Peak performance tradeoff**: Best single-batch Dice dropped (0.873 → 0.737). This is the expected tradeoff with stronger augmentation — the model trades peak accuracy on "easy" images for better generalization on harder ones.

6. **More gradient instability**: More gradient explosion events (3 inf + 2 large warnings vs 1 inf in Run 1), likely because the wider contrast augmentation creates more extreme training examples. May want to reduce gradient clip threshold or add gradient norm monitoring.

## Next Steps

- Add calibration loss to directly address threshold instability
- Consider early stopping (best model was epoch 45, last 35 epochs didn't improve)
