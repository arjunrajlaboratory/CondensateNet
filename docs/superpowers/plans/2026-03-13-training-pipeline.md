# Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a self-contained training pipeline from a 4,692-line Colab export into ~10 modular files under `training/`.

**Architecture:** Each module is extracted from specific line ranges in `extra_stuff/simpler_condensate_celly_11_11_25.py`. Modules follow the dependency graph in the spec. No new logic is written — this is a pure extraction and cleanup refactoring.

**Tech Stack:** PyTorch, timm, albumentations, tifffile, scipy, numpy, pandas

**Spec:** `docs/superpowers/specs/2026-03-13-training-pipeline-design.md`
**Source:** `extra_stuff/simpler_condensate_celly_11_11_25.py`

---

## Chunk 1: Scaffolding and Standalone Modules

### Task 1: Create package scaffolding

**Files:**
- Create: `training/__init__.py`
- Create: `training/data/__init__.py`
- Create: `training/requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Create directory structure and init files**

```python
# training/__init__.py
"""CondensateNet training pipeline."""
```

```python
# training/data/__init__.py
"""Data pipeline for condensate segmentation training."""
```

- [ ] **Step 2: Create requirements.txt**

```
# training/requirements.txt
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

- [ ] **Step 3: Add data/ to .gitignore**

Append to `.gitignore`:
```
# Training data (large, not versioned)
data/
```

- [ ] **Step 4: Set up virtual environment and install dependencies**

Check if `.venv` already exists. If not, create and install:
```bash
cd /Users/arjunraj/code/CondensateNet
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
```

If `.venv` already exists, just activate and install:
```bash
source .venv/bin/activate
pip install -r training/requirements.txt
```

**Note:** All subsequent `python` commands assume the venv is active.

- [ ] **Step 5: Commit**

```bash
git add training/__init__.py training/data/__init__.py training/requirements.txt .gitignore
git commit -m "scaffold: create training package structure"
```

---

### Task 2: Create config.py

**Files:**
- Create: `training/config.py`

Extract from source lines 67-73 (`MODEL_PARAMS`) and lines 263-375 (`PipelineConfig`).

Changes from source:
- Remove `aug_presets` nested dict (redundant with AugmentationPresets)
- Replace Colab/Drive paths with repo-relative `data/`
- Remove `cache_dir`, `enable_cache`, `target_foreground_ratio`, `flow_validate`
- Remove `__post_init__` directory creation (caller's responsibility)

- [ ] **Step 1: Write config.py**

```python
# training/config.py
"""Configuration for the condensate segmentation training pipeline."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional

# Repository root (parent of training/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"

MODEL_PARAMS = {
    'encoder_variant': 'rw_s',
    'pyramid_channels': [24, 48, 64, 160],
    'use_spatial_attention': True,
    'spatial_kernel_size': 11,
    'dropout_rate': 0.15,
}

EXPERIMENT_CAPS = {
    'HSPA1A_43C_26hr': 32,
    'ddx5_srrm2_1-636': 30,
    '15minrepeat001': 80,
}


@dataclass
class PipelineConfig:
    # Tiling
    tile_size: int = 320
    stride: int = 224
    min_foreground_pixels: int = 10
    foreground_sampling_prob: float = 0.95
    min_image_size: int = 256
    samples_per_image: int = 2

    # Flow
    flow_niter: int = 200
    flow_use_bbox: bool = True
    flow_min_size: int = 10
    flow_device: str = 'cpu'

    # Augmentation
    use_albumentations: bool = True
    aug_mode: str = 'medium'

    # Training
    val_ratio: float = 0.15
    random_seed: int = 42
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True

    # Loss
    flow_scale: float = 5.0
    focal_weight: float = 0.4
    dice_weight: float = 0.6
    flow_weight: float = 1.0

    # Paths
    images_dir: Path = field(default_factory=lambda: _DATA_DIR / "images")
    masks_dir: Path = field(default_factory=lambda: _DATA_DIR / "refined_masks")
    output_dir: Path = field(default_factory=lambda: _DATA_DIR / "processed")
    experiment_caps: Dict[str, int] = field(default_factory=lambda: dict(EXPERIMENT_CAPS))
```

- [ ] **Step 2: Verify import works**

Run: `cd /path/to/CondensateNet && python -c "from training.config import PipelineConfig, MODEL_PARAMS; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add training/config.py
git commit -m "feat: add training config module"
```

---

### Task 3: Create model.py

**Files:**
- Create: `training/model.py`

Extract from source lines 60-261. This is the complete model architecture — no changes to any logic, initialization, or forward pass. Only change: import `MODEL_PARAMS` from config.

- [ ] **Step 1: Write model.py**

Extract these classes verbatim from source lines 75-261:
- `SparseAttention` (lines 75-87)
- `NormProjection` (lines 89-96)
- `ConvBlock` (lines 98-106)
- `StyleModulatedConv` (lines 108-117)
- `DualResidualBlock` (lines 119-133)
- `MultiScaleEncoder` (lines 135-154)
- `CondensateSegmentationNet` (lines 156-257)
- `create_condensate_model()` (lines 259-261)

The file should have:
```python
# training/model.py
"""CondensateNet model architecture for condensate segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from .config import MODEL_PARAMS

# ... all classes extracted verbatim from source ...
```

Preserve all weight initialization logic in `_init_weights()` exactly as-is, including the specific `mask_head` bias of -2.0.

- [ ] **Step 2: Verify import and model creation**

Run: `python -c "from training.model import create_condensate_model; m = create_condensate_model(); print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}')"`
Expected: prints parameter count (should be ~7-8M)

- [ ] **Step 3: Commit**

```bash
git add training/model.py
git commit -m "feat: add model architecture module"
```

---

### Task 4: Create data/tiling.py

**Files:**
- Create: `training/data/tiling.py`

Extract from source lines 963-1032 (`RandomTilingStrategy`), 1035-1109 (`ValidationTilingStrategy`), 1267-1303 (`extract_tile`, `extract_flow_tile`).

Delete: `GridTilingStrategy`, `AdaptiveTilingStrategy`, `TilingStrategyFactory`, `TilePosition`.

- [ ] **Step 1: Write tiling.py**

```python
# training/data/tiling.py
"""Tiling strategies for condensate segmentation dataset."""

import numpy as np
from typing import Tuple, Optional


class RandomTilingStrategy:
    # ... extract verbatim from source lines 963-1032 ...


class ValidationTilingStrategy:
    # ... extract verbatim from source lines 1035-1109 ...


def extract_tile(image: np.ndarray, mask: np.ndarray,
                 top: int, left: int, tile_size: int) -> Tuple[np.ndarray, np.ndarray]:
    # ... extract verbatim from source lines 1267-1285 ...


def extract_flow_tile(flows: np.ndarray, top: int, left: int,
                      tile_size: int) -> np.ndarray:
    # ... extract verbatim from source lines 1288-1303 ...
```

- [ ] **Step 2: Verify import**

Run: `python -c "from training.data.tiling import RandomTilingStrategy, ValidationTilingStrategy; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add training/data/tiling.py
git commit -m "feat: add tiling strategies module"
```

---

### Task 5: Create data/augmentation.py

**Files:**
- Create: `training/data/augmentation.py`

Extract the **v2 (FIXED) versions only** from source lines 1775-2103. Delete the v1 copy (lines 1376-1773) entirely.

- [ ] **Step 1: Write augmentation.py**

Extract these classes from source lines 1788-2102 (the v2 block):
- `AugmentationPresets` (lines 1788-1839)
- `AlbumentationsAugmentation` (lines 1841-1962)
- `SimpleAugmentation` (lines 1964-2038)
- `ExperimentAwareAugmentation` (lines 2040-2066)
- `AugmentationFactory` (lines 2068-2102)

```python
# training/data/augmentation.py
"""Experiment-aware data augmentation for condensate segmentation."""

import numpy as np
from typing import Dict, Tuple, Optional, Union
import warnings

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    warnings.warn("albumentations not available, using simple augmentation")

# ... all v2 classes extracted verbatim ...
```

Key preservation points:
- Flow field CHW<->HWC conversion in `AlbumentationsAugmentation.__call__`
- `np.clip(aug_flows, -1.0, 1.0)` after augmentation
- Flow vector negation on flips in `SimpleAugmentation`
- Flow vector rotation on 90-degree rotations in `SimpleAugmentation`

- [ ] **Step 2: Verify import**

Run: `python -c "from training.data.augmentation import AugmentationFactory; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add training/data/augmentation.py
git commit -m "feat: add augmentation module (v2 only)"
```

---

### Task 6: Create data/flows.py

**Files:**
- Create: `training/data/flows.py`

Extract verbatim from source lines 2106-2326. **No changes whatsoever** — this is the Cellpose heat-diffusion algorithm and must be preserved exactly.

- [ ] **Step 1: Write flows.py**

```python
# training/data/flows.py
"""Cellpose-style flow field generation via heat diffusion."""

import numpy as np
import warnings
from typing import Tuple
from scipy.ndimage import find_objects

import torch
import torch.nn.functional as F


class FlowGenerator:
    # ... extract verbatim from source lines 2122-2326 ...
```

Extract these methods exactly:
- `__init__` (lines 2125-2137)
- `generate` (lines 2139-2150)
- `_generate_full` (lines 2152-2198)
- `_generate_with_bbox` (lines 2200-2233)
- `_compute_instance_flows_diffusion` (lines 2235-2286)
- `_extend_centers_gpu` (lines 2288-2305)
- `_get_centers` (lines 2307-2326)

Remove the `TORCH_AVAILABLE` guard at the top — torch is a required dependency.

- [ ] **Step 2: Verify import**

Run: `python -c "from training.data.flows import FlowGenerator; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add training/data/flows.py
git commit -m "feat: add flow field generation module"
```

---

### Task 7: Create data/registry.py

**Files:**
- Create: `training/data/registry.py`

Extract from source lines 387-942:
- `RegistrySample` (lines 387-396)
- `DataRegistry` (lines 399-699)
- `ExperimentAnalyzer` (lines 708-748)
- `StratifiedDataSplitter` (lines 769-919)
- `quick_split()` (lines 921-942)

Delete: all inline execution statements (lines 753-758).

- [ ] **Step 1: Write registry.py**

```python
# training/data/registry.py
"""Data registry for condensate segmentation pipeline."""

import re
import numpy as np
import pandas as pd
import tifffile
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple


class RegistrySample:
    # ... extract from lines 387-396 ...


class DataRegistry:
    # ... extract from lines 399-699 ...


class ExperimentAnalyzer:
    # ... extract from lines 708-748 ...


class StratifiedDataSplitter:
    # ... extract from lines 769-919 ...


def quick_split(manifest_df: pd.DataFrame,
                experiment_caps: Optional[Dict[str, int]] = None,
                val_ratio: float = 0.15,
                output_dir: Optional[Path] = None,
                random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ... extract from lines 921-942 ...
```

- [ ] **Step 2: Verify import**

Run: `python -c "from training.data.registry import DataRegistry, quick_split; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add training/data/registry.py
git commit -m "feat: add data registry and splitting module"
```

---

## Chunk 2: Dataset, Loss, Training Loop, and Inference

### Task 8: Create data/dataset.py

**Files:**
- Create: `training/data/dataset.py`

Extract from source lines 2343-2706:
- `CondensateDataset` (lines 2343-2639)
- `create_dataloaders()` (lines 2642-2684) — merged with `create_condensate_dataloaders()` (lines 2687-2706)

Changes from source:
- Replace `TilingStrategyFactory.create_for_training(...)` with direct `RandomTilingStrategy(...)` instantiation
- Replace `TilingStrategyFactory.create_for_validation(...)` with direct `ValidationTilingStrategy(...)` instantiation
- Merge `create_condensate_dataloaders()` behavior into `create_dataloaders()`: takes config, looks for manifests in `config.output_dir`
- Import from sibling modules: `.tiling`, `.augmentation`, `.flows`

- [ ] **Step 1: Write dataset.py**

```python
# training/data/dataset.py
"""Production dataset for condensate segmentation with Cellpose-style flows."""

import numpy as np
import pandas as pd
import torch
import tifffile
import warnings
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, Tuple

from ..config import PipelineConfig
from .tiling import RandomTilingStrategy, ValidationTilingStrategy, extract_tile, extract_flow_tile
from .augmentation import AugmentationFactory
from .flows import FlowGenerator


class CondensateDataset(Dataset):
    # ... extract from source lines 2343-2639 ...
    #
    # In _setup_components(), replace:
    #   TilingStrategyFactory.create_for_training(...) -> RandomTilingStrategy(...)
    #   TilingStrategyFactory.create_for_validation(...) -> ValidationTilingStrategy(...)


def create_dataloaders(config: PipelineConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from config.

    Looks for train_manifest.csv and val_manifest.csv in config.output_dir.
    """
    train_manifest = config.output_dir / "train_manifest.csv"
    val_manifest = config.output_dir / "val_manifest.csv"

    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError(
            f"Manifests not found. Expected:\n"
            f"  {train_manifest}\n"
            f"  {val_manifest}\n"
            f"Run data scanning and splitting first."
        )

    train_ds = CondensateDataset(str(train_manifest), "train", config)
    val_ds = CondensateDataset(str(val_manifest), "val", config)

    print("\n=== Training Dataset ===")
    train_ds.print_augmentation_info()
    print("\n=== Validation Dataset ===")
    val_ds.print_augmentation_info()

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        persistent_workers=config.num_workers > 0
    )

    return train_loader, val_loader
```

- [ ] **Step 2: Verify import**

Run: `python -c "from training.data.dataset import CondensateDataset, create_dataloaders; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add training/data/dataset.py
git commit -m "feat: add dataset and dataloader module"
```

---

### Task 9: Create loss.py

**Files:**
- Create: `training/loss.py`

Extract `CondensateLoss` from source lines 3471-3566. Use the tuned hyperparameters from the inline instantiation (lines 3569-3581) as the new defaults:
- `alpha=0.90` (was 0.8)
- `gamma=2.5` (was 2.0)
- `focal_weight=0.4` (was 0.5)
- `dice_weight=0.6` (was 0.5)
- `tversky_alpha=0.5` (was 0.7)
- `tversky_beta=0.5` (was 0.3)
- `dice_mode='soft'` (was 'dice')
- `mask_flows='soft'` (was 'soft', unchanged)

- [ ] **Step 1: Write loss.py**

```python
# training/loss.py
"""Combined loss function for condensate segmentation."""

import torch
import torch.nn as nn


class CondensateLoss(nn.Module):
    """
    Combined focal + Tversky + flow MSE loss for condensate segmentation.

    Default parameters are tuned for sparse, small, low-contrast condensates.
    flow_scale scales the loss contribution, NOT the target values.
    """

    def __init__(self,
                 flow_scale=5.0,
                 alpha=0.90,
                 gamma=2.5,
                 focal_weight=0.4,
                 dice_weight=0.6,
                 flow_weight=1.0,
                 dice_mode='soft',
                 mask_flows='soft',
                 tversky_alpha=0.5,
                 tversky_beta=0.5,
                 smooth=1e-6):
        super().__init__()
        self.flow_scale = flow_scale
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.flow_weight = flow_weight
        self.dice_mode = dice_mode.lower()
        self.mask_flows = mask_flows.lower()
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.smooth = smooth

    def forward(self, preds, batch):
        # ... extract verbatim from source lines 3507-3566 ...
```

The `forward` method is extracted exactly from source. It returns `(total_loss, loss_dict)`.

- [ ] **Step 2: Verify import and basic forward pass**

Run:
```python
python -c "
import torch
from training.loss import CondensateLoss
loss_fn = CondensateLoss()
preds = {'mask': torch.randn(2, 1, 32, 32), 'flows': torch.randn(2, 2, 32, 32)}
batch = {'mask': torch.randint(0, 2, (2, 1, 32, 32)).float(), 'flows': torch.randn(2, 2, 32, 32)}
loss, d = loss_fn(preds, batch)
print(f'Loss: {loss.item():.4f}, components: {d}')
"
```
Expected: prints loss value and component dict without errors

- [ ] **Step 3: Commit**

```bash
git add training/loss.py
git commit -m "feat: add loss function with tuned defaults"
```

---

### Task 10: Create train.py

**Files:**
- Create: `training/train.py`

Extract from source lines 3605-3957 (metrics, threshold search, training loop). Add a clean `__main__` block.

Delete:
- `train_with_config()` wrapper (lines 4039-4074)
- Three ad-hoc training invocations (lines 4076-4136)
- Inline plotting block (lines 3960-4031) — keep the history dict so plots can be added later

- [ ] **Step 1: Write train.py**

```python
# training/train.py
"""Training loop for condensate segmentation model."""

import time
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from pathlib import Path

from .config import PipelineConfig
from .model import create_condensate_model
from .loss import CondensateLoss
from .data.registry import DataRegistry, quick_split
from .data.dataset import create_dataloaders


def compute_condensate_metrics(pred_mask_logits, pred_flows, gt_mask, gt_flows, thr=0.5):
    # ... extract verbatim from source lines 3605-3660 ...


def adaptive_threshold_search(pred_mask_logits, gt_mask, thresholds=None):
    # ... extract verbatim from source lines 3663-3681 ...


def train_condensate_model(
    model, train_loader, val_loader, loss_fn, config,
    num_epochs=80, lr=2e-4, weight_decay=1e-4,
    save_dir=None, device=None, mixed_precision=True,
    gradient_clip_value=5.0
):
    """Train condensate segmentation model."""
    # ... extract from source lines 3688-3957, with these changes:
    #   - Remove optimizer_name param (always AdamW)
    #   - Remove scheduler_name param (always ReduceLROnPlateau)
    #   - Remove monitor_gradients and log_interval params (always monitor)
    #   - Remove the 70-line plotting block at the end (lines 3960-4031)
    #   - Keep saving training_curves data in history dict
    #   - Save history dict in the final checkpoint
    #   - Default lr=2e-4, num_epochs=80 (from the final training round)


if __name__ == "__main__":
    config = PipelineConfig()

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data pipeline
    print("Scanning for image-mask pairs...")
    registry = DataRegistry(config.images_dir, config.masks_dir)
    manifest_df = registry.scan_and_validate()

    print("\nCreating train/val splits...")
    train_df, val_df = quick_split(
        manifest_df,
        experiment_caps=config.experiment_caps,
        val_ratio=config.val_ratio,
        output_dir=config.output_dir,
        random_seed=config.random_seed
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # 2. Dataloaders
    train_loader, val_loader = create_dataloaders(config)

    # 3. Model + loss
    model = create_condensate_model()
    loss_fn = CondensateLoss()

    # 4. Train
    model, history = train_condensate_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
    )
```

- [ ] **Step 2: Verify import**

Run: `python -c "from training.train import train_condensate_model, compute_condensate_metrics; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add training/train.py
git commit -m "feat: add training loop with clean entry point"
```

---

### Task 11: Create inference.py

**Files:**
- Create: `training/inference.py`

Extract from source lines 4166-4519:
- `CondensateInference` class (lines 4166-4519)

Delete:
- `test_on_validation_set()` (lines 4522-4602)
- `generate_summary_plots()` (lines 4605-4659)
- Colab-specific `sys.path` hack (lines 4162-4163)

Add a `__main__` block for command-line usage.

- [ ] **Step 1: Write inference.py**

```python
# training/inference.py
"""Inference pipeline for condensate segmentation model."""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from skimage import measure
from typing import Dict, List, Tuple, Optional
import warnings

from .config import PipelineConfig
from .model import create_condensate_model
from .train import compute_condensate_metrics


class CondensateInference:
    # ... extract from source lines 4166-4519 ...
    #
    # Changes:
    # - Remove Colab sys.path hack
    # - In _load_model(), use create_condensate_model() import
    # - Remove seaborn dependency (not used in the class itself)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run condensate segmentation inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--val-manifest", default=None, help="Path to validation manifest CSV")
    parser.add_argument("--output-dir", default="inference_results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-viz", type=int, default=5, help="Number of visualizations")
    args = parser.parse_args()

    config = PipelineConfig()

    val_manifest = args.val_manifest or str(config.output_dir / "val_manifest.csv")

    from .data.dataset import CondensateDataset
    from torch.utils.data import DataLoader

    val_dataset = CondensateDataset(val_manifest, split="val", config=config)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    inferencer = CondensateInference(
        model_path=args.checkpoint,
        device=args.device,
        adaptive_threshold=True
    )

    df_metrics = inferencer.test_dataloader(
        val_loader,
        num_visualizations=args.num_viz,
        save_dir=Path(args.output_dir)
    )

    if len(df_metrics) > 0:
        df_metrics.to_csv(Path(args.output_dir) / "metrics.csv", index=False)
        print(f"\nResults saved to {args.output_dir}/")
```

- [ ] **Step 2: Verify import**

Run: `python -c "from training.inference import CondensateInference; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add training/inference.py
git commit -m "feat: add inference module"
```

---

## Chunk 3: Integration Verification

### Task 12: Full import smoke test and final commit

**Files:**
- None new — verification only

- [ ] **Step 1: Run full import chain**

```bash
python -c "
from training.config import PipelineConfig, MODEL_PARAMS, EXPERIMENT_CAPS
from training.model import CondensateSegmentationNet, create_condensate_model
from training.data.registry import DataRegistry, StratifiedDataSplitter, quick_split
from training.data.tiling import RandomTilingStrategy, ValidationTilingStrategy
from training.data.augmentation import AugmentationFactory, AlbumentationsAugmentation
from training.data.flows import FlowGenerator
from training.data.dataset import CondensateDataset, create_dataloaders
from training.loss import CondensateLoss
from training.train import train_condensate_model, compute_condensate_metrics
from training.inference import CondensateInference
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 2: Verify model forward pass end-to-end**

```bash
python -c "
import torch
from training.model import create_condensate_model
from training.loss import CondensateLoss

model = create_condensate_model()
loss_fn = CondensateLoss()

x = torch.randn(2, 1, 320, 320)
preds = model(x)
print(f'mask: {preds[\"mask\"].shape}, flows: {preds[\"flows\"].shape}')

batch = {'mask': torch.randint(0, 2, (2, 1, 320, 320)).float(), 'flows': torch.randn(2, 2, 320, 320)}
loss, d = loss_fn(preds, batch)
print(f'Loss: {loss.item():.4f}')
print('Forward pass + loss OK')
"
```
Expected: prints shapes and loss without errors

- [ ] **Step 3: Verify data modules work together**

```bash
python -c "
import numpy as np
from training.data.tiling import RandomTilingStrategy, extract_tile, extract_flow_tile
from training.data.augmentation import AugmentationFactory
from training.data.flows import FlowGenerator

# Test tiling
mask = np.zeros((512, 512), dtype=np.int32)
mask[100:120, 100:120] = 1
strategy = RandomTilingStrategy(tile_size=320, foreground_bias=0.7)
top, left = strategy.sample_tile(mask)
print(f'Tile position: ({top}, {left})')

# Test flow generation
flow_gen = FlowGenerator(device='cpu', niter=50)
flows = flow_gen.generate(mask)
print(f'Flows shape: {flows.shape}, range: [{flows.min():.3f}, {flows.max():.3f}]')

# Test augmentation
aug = AugmentationFactory.create_from_count(10, use_albumentations=True)
image = np.random.rand(320, 320).astype(np.float32)
tile_mask = mask[top:top+320, left:left+320].copy()
tile_flows = flows[:, top:top+320, left:left+320].copy()
aug_img, aug_mask, aug_flows = aug(image, tile_mask, tile_flows)
print(f'Augmented flows range: [{aug_flows.min():.3f}, {aug_flows.max():.3f}]')
print('Data pipeline OK')
"
```
Expected: prints values without errors

- [ ] **Step 4: Final commit**

```bash
git add -A training/
git commit -m "feat: complete training pipeline extraction

Self-contained training pipeline extracted from Colab notebook.
~1,800 lines across 10 modules (down from 4,692 in one file).

Modules: config, model, data/{registry,tiling,augmentation,flows,dataset}, loss, train, inference"
```
