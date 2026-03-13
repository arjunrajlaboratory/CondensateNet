# Training Pipeline Setup Log — 2026-03-13

## Goal

Create a self-contained, simplified training pipeline for the CondensateNet condensate segmentation model, extracted from a 4,692-line Colab notebook export.

## Source Materials

### Training Script

`extra_stuff/simpler_condensate_celly_11_11_25.py` — a monolithic Colab notebook export containing model architecture, data pipeline, training loop, diagnostics, and inference code. Approximately 60% was dead code (duplicates, diagnostics, Colab boilerplate).

A refactoring plan was generated separately and stored at `extra_stuff/condensate_refactoring_plan.md`.

### Training Data

Downloaded from Google Drive as 15 split zip files in `extra_stuff/CondensateModelData/` (~25 GB total compressed). These are **not** parts of a single split archive — each zip is an independent archive containing a different subset of files from a single top-level `Condensate_Model/` directory.

## Data Findings

### What's in the zips (3,928 files total)

| Category | Count | Description |
|----------|-------|-------------|
| `.npy` arrays | 1,820 | Masks, instance segmentations, flow fields, refined masks, flow caches |
| `.png` images | 849 | Visualizations, QC reports, training curves |
| `.tif` images | 347 | Raw microscopy images |
| `.csv` files | 272 | QC metrics, evaluation results, manifests |
| `.pt` checkpoints | 228 | PyTorch model checkpoints across ~20 experiment dirs |
| Web app source | 167 | NimbusImage Vue/TypeScript app (unrelated) |
| Python source | 67 | Earlier pipeline code |
| Other | ~60 | YAML, markdown, etc. |

### What we actually need for training

Only two directories matter:

- `Condensate_Model/data/images/` — raw `.tif` microscopy images
- `Condensate_Model/data/refined_masks/` — cleaned `.npy` binary masks

Everything else (checkpoints, visualizations, NimbusImage web app, QC reports, earlier pipeline code, Jupyter notebooks) is historical artifacts from experimentation.

### Extracted Data Statistics

Extracted from 11 of the 15 zips into `data/` at repo root (git-ignored):

- **254 `.tif` images** in `data/images/`
- **252 `.npy` masks** in `data/refined_masks/` (flat directory, no subdirectories)
- **252 matched image-mask pairs** (3 images have no corresponding mask)
- **Total size: 2.2 GB**
- **Image dimensions**: 400x400 to 2048x2048 (mean ~1392x1392)
- **All 252 pairs pass validation**

### Experiment Distribution

| Experiment | Count | Notes |
|------------|-------|-------|
| 15minrepeat001 | 141 | Time-lapse, multiple xy positions/z-slices. Capped to 80 in training. |
| HSPA1A | 33 | Heat shock protein 43C 26hr. Capped to 32. |
| ddx5 | 32 | ddx5_srrm2 splicing-related. Capped to 30. |
| Denoised | 12 | Various denoised/deconvolved microscopy |
| G3BP1-mClover | 11 | Stress granule experiments (sorbitol, NaAsO2) |
| 2025-06-30 | 8 | Date-labeled experiments |
| TIAR | 4 | Immunofluorescence fixed cells |
| GFP | 3 | GFP fusion protein constructs |
| Other (6 exps) | 8 | 1-2 samples each |
| **Total** | **252** | |

The `EXPERIMENT_CAPS` in `training/config.py` limits overrepresented experiments to balance training.

## What Was Built

### Package Structure

Refactored 4,692 lines into ~2,000 clean lines across 10 modules:

```
training/
├── __init__.py
├── config.py            (60 lines)   MODEL_PARAMS, EXPERIMENT_CAPS, PipelineConfig
├── model.py             (200 lines)  EfficientNetV2-S encoder, style-modulated FPN, dual heads
├── data/
│   ├── __init__.py
│   ├── registry.py      (550 lines)  DataRegistry, StratifiedDataSplitter, quick_split
│   ├── tiling.py        (170 lines)  RandomTilingStrategy, ValidationTilingStrategy
│   ├── augmentation.py  (300 lines)  Albumentations + numpy fallback, flow-safe
│   ├── flows.py         (200 lines)  Cellpose heat-diffusion flow generation
│   └── dataset.py       (300 lines)  CondensateDataset, create_dataloaders
├── loss.py              (100 lines)  Focal + Tversky + flow MSE (tuned defaults)
├── train.py             (250 lines)  Training loop + __main__ entry point
├── inference.py         (400 lines)  CondensateInference class + CLI
└── requirements.txt                  Training-specific dependencies
```

### Key Design Decisions

1. **Separate from `condensatenet/` package** — the existing package handles a different model for inference. This training pipeline is fully self-contained.

2. **Data lives in `data/` at repo root** (git-ignored) — not versioned, will need a download/extraction script once we decide on permanent storage.

3. **Tuned hyperparameters baked in as defaults** — the loss function defaults (`alpha=0.90`, `gamma=2.5`, `focal_weight=0.4`, `dice_weight=0.6`, `dice_mode='soft'`, `tversky_alpha=0.5`, `tversky_beta=0.5`) come from the final tuning round in the source notebook.

4. **Model architecture unchanged** — CondensateSegmentationNet with EfficientNetV2-S encoder (~22M parameters), style-modulated FPN, dual output heads (mask logits + 2D flow).

### Bugs Found and Fixed During Review

1. **Foreground sampling probability** — was hardcoded as 0.7 instead of using config's 0.95. For sparse condensate images, this significantly reduces the chance of getting useful training tiles.

2. **Image size validation** — registry validated against `min_image_size=256` but tiles are 320x320. Images between 256-319px would pass validation but crash during tile extraction. Fixed to validate against `max(min_image_size, tile_size)`.

3. **Pixel transforms applied to flow fields** — albumentations was applying brightness, contrast, noise, and blur transforms to flow field ground truth (because flows were registered as `'image'` type in a single pipeline). Split into separate geometric and pixel pipelines so only geometric transforms affect flows.

### Domain Logic Preserved Exactly

- Flow generation (`FlowGenerator._extend_centers_gpu`) — Cellpose heat-diffusion algorithm
- Augmentation-flow consistency — flow vectors negated on flips, rotated on 90-degree rotations, clipped to [-1,1]
- NaN sanitization — at flow generation, post-augmentation, and pre-tensor stages
- Percentile normalization (0.1/99.9) — critical for microscopy outlier pixels
- Foreground-biased tiling — essential for sparse condensate images
- Weight initialization — Kaiming for convs, Xavier for linear, mask head bias=-2.0

## Running Instructions

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 12+ GB VRAM (tested target: RTX 3060)
- CUDA toolkit compatible with PyTorch 2.0+

### Setup

```bash
# 1. Clone the repo and switch to the training branch
git clone <repo-url>
cd CondensateNet
git checkout feature/training-pipeline

# 2. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt

# 3. Place training data
# Copy the data/ directory (2.2 GB) to the repo root:
#   data/images/        — 254 .tif files
#   data/refined_masks/ — 252 .npy files
```

### Training

```bash
source .venv/bin/activate
python -m training.train
```

This will:
1. Scan `data/images/` and `data/refined_masks/` for matching pairs
2. Create stratified train/val splits (85/15) with experiment balancing
3. Save CSV manifests to `data/processed/`
4. Train for 80 epochs with AdamW (lr=2e-4), ReduceLROnPlateau, mixed precision
5. Save best model to `data/processed/checkpoints/best_model.pt`

Expected training config: batch_size=8, tile_size=320x320, ~22M parameter model. Should fit comfortably in 12 GB VRAM.

### Inference

```bash
python -m training.inference \
    --checkpoint data/processed/checkpoints/best_model.pt \
    --output-dir inference_results \
    --batch-size 4
```

### Key Configuration

All defaults are in `training/config.py`. Override by modifying `PipelineConfig` in the `__main__` block of `train.py`, or by constructing a config programmatically:

```python
from training.config import PipelineConfig

config = PipelineConfig(
    batch_size=4,        # reduce if GPU OOM
    num_workers=2,       # increase if CPU bottlenecked
    tile_size=256,       # reduce if GPU OOM
    flow_device='cuda',  # 'cpu' is safer but slower
)
```

## File Inventory

### Committed to `feature/training-pipeline`

- `training/` — the complete training package (10 modules)
- `training/requirements.txt` — Python dependencies
- `docs/superpowers/specs/2026-03-13-training-pipeline-design.md` — design spec
- `docs/superpowers/plans/2026-03-13-training-pipeline.md` — implementation plan
- `.gitignore` — updated to exclude `/data/`

### Not committed (git-ignored, must be transferred separately)

- `data/images/` — 254 `.tif` microscopy images (part of 2.2 GB)
- `data/refined_masks/` — 252 `.npy` mask files (part of 2.2 GB)

### Not needed (can be deleted from `extra_stuff/`)

- `extra_stuff/CondensateModelData/` — 25 GB of zip files (data already extracted)
- `extra_stuff/simpler_condensate_celly_11_11_25.py` — original notebook (refactored into `training/`)
- `extra_stuff/condensate_refactoring_plan.md` — plan (executed and completed)
