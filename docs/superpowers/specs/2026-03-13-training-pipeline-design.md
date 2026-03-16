# Training Pipeline Design Spec

## Goal

Create a self-contained, modular training pipeline for the CondensateNet segmentation model. Refactor the 4,692-line Colab export (`extra_stuff/simpler_condensate_celly_11_11_25.py`) into ~1,800 clean lines across ~10 files. The pipeline should be independent of the existing `condensatenet/` inference package.

## Context

- **Source**: `extra_stuff/simpler_condensate_celly_11_11_25.py` — a Colab notebook export containing model architecture, data pipeline, training loop, diagnostics, and inference code. Approximately 60% is dead code (duplicates, diagnostics, Colab boilerplate).
- **Refactoring guide**: `extra_stuff/condensate_refactoring_plan.md` — detailed analysis of what to keep, delete, and restructure.
- **Training data**: ~255 `.tif` microscopy images and matching `.npy` masks from ~20 experiments. Stored in `data/` at repo root (git-ignored).
- **Target hardware**: RTX 3060 (12GB VRAM). Batch size 8, 320x320 tiles, EfficientNetV2-S backbone should fit comfortably.

## Package Structure

```
training/
├── __init__.py
├── config.py            PipelineConfig, paths, constants, MODEL_PARAMS
├── model.py             All nn.Module classes + create_condensate_model()
├── data/
│   ├── __init__.py
│   ├── registry.py      DataRegistry, RegistrySample, StratifiedDataSplitter
│   ├── tiling.py        RandomTilingStrategy, ValidationTilingStrategy, helpers
│   ├── augmentation.py  Augmentation v2 (albumentations-based)
│   ├── flows.py         FlowGenerator (Cellpose heat diffusion)
│   └── dataset.py       CondensateDataset, create_dataloaders()
├── loss.py              CondensateLoss (focal + Tversky + flow MSE)
├── train.py             Training loop, metrics, __main__ entry point
├── inference.py         Load checkpoint, segment images, flow-to-instances
└── requirements.txt     Training-specific dependencies

data/                    (git-ignored, at repo root)
├── images/              .tif microscopy images (~255 files)
└── refined_masks/       .npy masks by experiment ID
```

## Module Specifications

### config.py (~60 lines)

Extracts from: lines 67-73 (MODEL_PARAMS) and lines 263-375 (PipelineConfig) of source.

Contains:
- `MODEL_PARAMS` dict (encoder_variant, pyramid_channels, attention config, dropout)
- `EXPERIMENT_CAPS` dict (per-experiment sample limits for balancing)
- `PipelineConfig` dataclass with fields for:
  - Tiling: tile_size=320, stride=224, min_foreground_pixels=10, foreground_sampling_prob=0.95, samples_per_image=2, min_image_size=256
  - Flow: flow_niter=200, flow_use_bbox=True, flow_min_size=10, flow_device='cpu'
  - Augmentation: use_albumentations=True, aug_mode='medium' (references presets in augmentation.py, does NOT duplicate them)
  - Training: val_ratio=0.15, random_seed=42, batch_size=8, num_workers=0, pin_memory=True
  - Loss weights: flow_scale=5.0, focal_weight=0.4, dice_weight=0.6, flow_weight=1.0 (tuned values from final training run)
  - Paths: images_dir, masks_dir, output_dir (default to `data/` at repo root)

Changes from source:
- Remove `aug_presets` nested dict (redundant with AugmentationPresets class)
- Remove Colab/Drive-specific paths; default to repo-relative `data/`
- Remove `cache_dir` and `enable_cache` (caching removed for simplicity)
- Remove `target_foreground_ratio` and `flow_validate` (unused in the actual training pipeline)

### model.py (~200 lines)

Extracts from: lines 60-261 of source.

Contains (in dependency order):
- `SparseAttention` — spatial attention with avg/max pooling
- `NormProjection` — BatchNorm + 1x1 conv
- `ConvBlock` — BatchNorm + SiLU + 3x3 conv
- `StyleModulatedConv` — ConvBlock with learned style bias
- `DualResidualBlock` — two style-modulated convs with residual connections
- `MultiScaleEncoder` — EfficientNetV2-S backbone via `timm.create_model()`, 4 pyramid levels
- `CondensateSegmentationNet` — full model: encoder + FPN + dual heads (mask logits, 2D flow)
- `create_condensate_model()` — factory function using MODEL_PARAMS

No changes to architecture or weight initialization logic. Imports `MODEL_PARAMS` from `config.py`.

### data/registry.py (~200 lines)

Extracts from: lines 377-942 of source.

Contains:
- `RegistrySample` — container for image-mask pair with validation state
- `DataRegistry` — discovers image-mask pairs by scanning directories, validates shapes/sizes/content
- `ExperimentAnalyzer` — prints distribution statistics and suggests caps (could be a function, but keeping class for now)
- `StratifiedDataSplitter` — creates balanced train/val splits with experiment stratification, applies caps with even spacing
- `quick_split()` — convenience function combining splitter + summary

Changes from source:
- Remove inline execution statements (lines 753-758)
- Consolidate imports

### data/tiling.py (~120 lines)

Extracts from: lines 944-1375 of source (subset).

Contains:
- `RandomTilingStrategy` — foreground-biased random tile sampling for training
- `ValidationTilingStrategy` — deterministic center-of-mass-based tiling for validation
- `extract_tile()` — extract image+mask tile at position
- `extract_flow_tile()` — extract flow field tile at position

Deleted from source:
- `GridTilingStrategy` (inference-only, over-abstracted)
- `AdaptiveTilingStrategy` (inference-only)
- `TilingStrategyFactory` (unnecessary indirection)
- `TilePosition` dataclass (only used by deleted strategies)

### data/augmentation.py (~200 lines)

Extracts from: lines 1775-2103 of source (v2 "FIXED" version only).

Contains:
- `AugmentationPresets` — light/medium/heavy parameter dicts
- `AlbumentationsAugmentation` — builds A.Compose pipeline, handles flow field transformation (CHW<->HWC conversion, NaN sanitization, clipping to [-1,1])
- `SimpleAugmentation` — numpy fallback with proper flow vector rotation on flips
- `ExperimentAwareAugmentation` — selects preset based on experiment frequency
- `AugmentationFactory` — creates augmentation from frequency string or sample count

Deleted from source:
- Entire first copy of augmentation module (lines 1376-1773)
- Inline NaN verification loop

Key preservation: flow field consistency (vector negation on flips, clipping to [-1,1], NaN guards).

### data/flows.py (~180 lines)

Extracts from: lines 2104-2326 of source.

Contains:
- `FlowGenerator` — generates Cellpose-style flow fields via heat diffusion
  - `_masks_to_flows()` — orchestrates per-instance flow computation
  - `_extend_centers_gpu()` — GPU-accelerated heat diffusion (the core algorithm)
  - Flow field validation and normalization

No changes. This is subtle, correct domain logic.

### data/dataset.py (~200 lines)

Extracts from: lines 2328-2706 of source (subset).

Contains:
- `CondensateDataset(torch.utils.data.Dataset)` — loads image-mask pairs, extracts tiles, generates flows, applies augmentation
  - `_load_image()` — tifffile reader with percentile normalization (0.1/99.9)
  - `_load_mask()` — numpy loader with binarization
  - `_get_sample()` — full pipeline: load, tile, generate flows, augment, to tensor
  - NaN sanitization at every stage
- `create_dataloaders()` — builds train/val DataLoader pair from config

Data flow: `train.py __main__` runs DataRegistry scan → StratifiedDataSplitter → saves CSV manifests to `output_dir` → `create_dataloaders()` reads those manifests.

Changes from source:
- Replace TilingStrategyFactory with direct instantiation of RandomTilingStrategy / ValidationTilingStrategy
- Import from sibling modules instead of inline definitions
- Merge `create_condensate_dataloaders()` into `create_dataloaders()` (takes config directly, looks for manifests in `config.output_dir`)

Deleted from source:
- `test_dataset()`, `analyze_dataloaders()`, `analyze_dataloaders_complete()`, `visualize_samples()`, `diagnose_masks()`, `debug_flows()` — all diagnostic functions

### loss.py (~100 lines)

Extracts from: lines 3465-3581 of source.

Contains:
- `CondensateLoss` — composite loss function:
  - Focal loss (for class imbalance in sparse condensates)
  - Tversky loss (asymmetric dice variant, tuned alpha/beta for small objects)
  - Flow MSE loss (weighted by mask, scaled by flow_scale)
  - Configurable weights for each component

Changes from source:
- Remove inline instantiation with hardcoded overrides (lines 3568-3581)
- The tuned hyperparameters from that inline block (alpha=0.90, gamma=2.5, focal_weight=0.4, dice_weight=0.6) become the class defaults, since they represent the final tuned values

### train.py (~200 lines)

Extracts from: lines 3583-4074 of source (subset).

Contains:
- `compute_condensate_metrics()` — per-batch metrics (dice, flow error, precision, recall)
- `adaptive_threshold_search()` — finds optimal probability threshold on validation set
- `train_condensate_model()` — full training loop with:
  - AdamW optimizer, ReduceLROnPlateau scheduler
  - Mixed precision (AMP) support
  - Gradient clipping
  - Validation every N epochs
  - Best model checkpointing
  - Training history tracking
- `__main__` block — clean entry point:
  1. Create config
  2. Scan data, create splits
  3. Build dataloaders
  4. Create model + loss
  5. Train

Deleted from source:
- Three ad-hoc training round invocations (lines 4076-4136)
- `train_with_config()` wrapper (replaced by `__main__`)
- Inline plotting (training curves can be added back later or use TensorBoard)

### inference.py (~350 lines)

Extracts from: lines 4138-4692 of source.

Contains:
- `CondensateInference` — loads trained checkpoint, runs inference:
  - Preprocessing (same percentile normalization as training)
  - Model forward pass
  - Probability thresholding
  - Flow-to-instances conversion (flow integration + clustering)
  - Instance filtering (min/max size)
- Basic visualization (overlay masks on images)
- `__main__` block for running inference on a directory of images

Deleted from source:
- `test_on_validation_set()` and `generate_summary_plots()` — evaluation scripts that can be added back later

### requirements.txt

```
torch>=2.0.0
timm>=0.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-image>=0.19.0
scipy>=1.7.0
tifffile>=2021.7.0
albumentations>=1.3.0
matplotlib>=3.5.0
tqdm>=4.60.0
```

## Data Flow (Training)

```
1. DataRegistry.scan_and_validate()     → manifest_df (all valid image-mask pairs)
2. quick_split(manifest_df, caps)       → train_df, val_df (saves CSV manifests to output_dir)
3. create_dataloaders(config)           → reads CSV manifests, creates CondensateDataset instances
4. CondensateDataset.__getitem__():
   a. Load image (tifffile) + mask (numpy)
   b. Generate flow fields (FlowGenerator)
   c. Sample tile (RandomTilingStrategy or ValidationTilingStrategy)
   d. Extract tile from image, mask, flows
   e. Apply augmentation (with flow consistency)
   f. NaN sanitization + convert to tensors
   g. Return {image, mask, flows}
```

## Module Dependency Graph

```
config.py (standalone)
model.py ← config (MODEL_PARAMS)
data/tiling.py (standalone)
data/augmentation.py (standalone)
data/flows.py (standalone, uses scipy + torch)
data/registry.py ← config (EXPERIMENT_CAPS, paths)
data/dataset.py ← config, tiling, augmentation, flows, registry
data/__init__.py — empty (or re-exports CondensateDataset, create_dataloaders)
loss.py (standalone)
train.py ← config, model, data/registry, data/dataset, loss
inference.py ← config, model, train (for compute_condensate_metrics)
```

## Data Layout

```
data/
├── images/              Extracted from zip: Condensate_Model/data/images/
│   └── *.tif            ~255 microscopy images from ~20 experiments
└── refined_masks/       Extracted from zip: Condensate_Model/data/refined_masks/
    ├── HSPA1A_43C_26hr/
    │   └── *_mask.npy
    ├── 15minrepeat001/
    │   └── *_mask.npy
    ├── ddx5_srrm2_1-636/
    │   └── *_mask.npy
    └── .../             ~20 experiment subdirectories
```

Only `data/images/` and `data/refined_masks/` are needed from the zip files. All other contents (checkpoints, visualizations, NimbusImage, etc.) are excluded.

A `.gitignore` entry will exclude `data/` from version control.

## What NOT to change

These components contain important domain logic preserved exactly as-is:

1. **Flow generation** (`FlowGenerator._extend_centers_gpu`) — Cellpose heat-diffusion algorithm
2. **Augmentation-flow consistency** — flow vectors negated on flips, clipped to [-1,1]
3. **NaN sanitization** in dataset — guards against flow generation/augmentation edge cases
4. **Percentile normalization** (99.9th/0.1th) — critical for microscopy outlier pixels
5. **Foreground-biased tiling** — essential for sparse condensate images
6. **Tversky loss alpha/beta** — tuned for sparse small objects
7. **Weight initialization** — Kaiming for convs, Xavier for linear, specific head init (mask bias=-2.0)

## Entry Points

**Training**: `python -m training.train` (or `python training/train.py`)
**Inference**: `python -m training.inference --checkpoint path/to/best.pt --input path/to/images/`
