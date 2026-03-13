# First Training Run — 2026-03-13

## Setup

- Branch: `feature/training-pipeline`
- Command: `python -m training.train` (all defaults from `PipelineConfig`)
- GPU: CUDA (RTX 3060, 12 GB VRAM)
- Duration: ~30 minutes (80 epochs, ~22 sec/epoch)
- Raw log: [2026-03-13-training-run-raw.txt](2026-03-13-training-run-raw.txt)

## Data

- 255 `.tif` images, 252 `.npy` masks available
- 245 pairs passed validation (7 rejected for constant intensity)
- Train: 158 samples, Val: 26 samples (stratified 85/15 split with experiment balancing)
- Experiment caps applied (e.g., 15minrepeat001 capped 141 to 80)
- Augmentation intensity by experiment size: light (large), medium (mid), heavy (small)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 8 |
| Tile size | 320x320 |
| Optimizer | AdamW |
| Initial LR | 2e-4 |
| Scheduler | ReduceLROnPlateau (patience=10) |
| Mixed precision | Yes |
| Gradient clipping | 5.0 |
| Model params | ~22M (EfficientNetV2-S backbone) |

## Results

**Best validation Dice: 0.503 at epoch 38**

| Metric | Value |
|--------|-------|
| Val Dice | 0.503 |
| Val IoU | 0.337 |
| Val Precision | 0.474 |
| Val Recall | 0.554 |
| Val F1 | 0.503 |
| Final train loss | 0.254 |
| Final val loss | 0.369 |
| Final train F1 | 0.611 |

### Learning Rate Schedule

The ReduceLROnPlateau scheduler reduced LR three times:

| Epoch | LR |
|-------|-----|
| 1-20 | 2.0e-4 |
| 21-49 | 1.0e-4 |
| 50-65 | 5.0e-5 |
| 66-80 | 2.5e-5 |

### Best Dice Progression

| Epoch | Val Dice | Notes |
|-------|----------|-------|
| 1 | 0.008 | Near-zero, model predicting almost nothing |
| 3 | 0.202 | First meaningful segmentation |
| 5 | 0.297 | |
| 6 | 0.353 | |
| 9 | 0.382 | |
| 31 | 0.400 | After first LR reduction |
| 34 | 0.409 | |
| 35 | 0.433 | |
| 36 | 0.459 | Rapid improvement phase |
| 37 | 0.492 | |
| 38 | 0.503 | **Best** |

## Observations

1. **Overfitting**: Clear train/val gap after epoch 38. Train F1 reached 0.61 while val Dice plateaued around 0.43-0.50. Train loss continued dropping (0.59 to 0.25) while val loss stalled (~0.35-0.37).

2. **Gradient instability**: One gradient explosion (inf norm) at epoch 41, batch 4. The model recovered immediately thanks to gradient clipping at 5.0. The dice loss component was high (0.759) at that batch.

3. **Loss components**: By end of training, focal loss was very small (~0.013 train, ~0.017 val), dice loss dominated (~0.41 train, ~0.58 val), and flow loss was minimal (~0.006 train, ~0.004 val). The model is mostly learning the mask segmentation task.

4. **Threshold drift**: The optimal threshold fluctuated significantly across epochs (0.17 to 0.90), suggesting the model's confidence calibration is unstable. This may warrant post-training threshold tuning on the validation set.

5. **Small validation set**: Only 26 validation samples means high variance in val metrics epoch-to-epoch. This likely explains some of the val Dice fluctuation.

## Potential Improvements

- **More data or stronger augmentation** to address overfitting
- **Larger validation set** (or k-fold cross-validation) for more stable evaluation
- **Early stopping** — best model was at epoch 38, the remaining 42 epochs didn't improve
- **Learning rate warmup** — the model jumped to high recall/low precision early, a warmup might help
- **Threshold calibration** on the validation set after training
- **Longer training with cosine annealing** instead of ReduceLROnPlateau
