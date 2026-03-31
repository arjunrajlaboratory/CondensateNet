# CondensateNet Retraining: Package Training API

## Overview

Add training/fine-tuning capability to the condensatenet Python package so that users can retrain the model on their own annotated data via NimbusImage. This is the **first step** — the package-side training API must be in place before the NimbusImage training worker can be built (see the companion `RETRAINING.md` in `ImageAnalysisProject/workers/annotations/condensatenet/todo/`).

## Background

### Model Architecture

The model (`CondensateSegmentationNet`) is defined in `extra_stuff/minimal_condensate_segmentation.ipynb` and on HuggingFace at `rajlab/condensatenet`. Key components:

- **Encoder**: `MultiScaleEncoder` using `efficientnetv2_rw_s` from timm (`features_only=True`, `in_chans=1`, `out_indices=[0,1,2,3]`)
- **Channel adapters**: 1x1 convs projecting encoder channels to `pyramid_channels=[24, 48, 64, 160]`
- **FPN Decoder**: `DualResidualBlock` modules with `StyleModulatedConv` (style vector from global avg pool of C4, L2-normalized)
- **Upsample + combine**: Multi-scale feature maps upsampled and summed at highest resolution
- **Attention**: `SparseAttention` (spatial attention using mean+max channel pooling → conv → sigmoid)
- **Dropout**: `Dropout2d(0.15)` (or 0.1 in training config)
- **Output heads**: Two 1x1 conv heads:
  - `mask_head`: `Conv2d(32, 1, 1)` → mask logits
  - `flow_head`: `Conv2d(32, 2, 1)` → flow vectors (dy, dx)
- **Final interpolation**: Both outputs bilinear-interpolated to input size

The model's `forward()` also does internal normalization: `(x - x.mean()) / x.std()` before encoding.

### Model Outputs

- `out['mask']` — shape `(B, 1, H, W)` — binary segmentation logits (apply `sigmoid` to get probabilities)
- `out['flows']` — shape `(B, 2, H, W)` — flow vectors `[dy, dx]` pointing toward instance centers

### Inference Pipeline

`preprocess (percentile normalize 0.5-99.5) → model forward → sigmoid(mask) → flow integration (200 steps, adaptive step size) → endpoint clustering → size filtering`

### Model Persistence

The model is loaded via HuggingFace `AutoModel.from_pretrained()` with `trust_remote_code=True` and saved as `config.json` + `model.safetensors`. Training must produce models in the same format so they can be loaded by the existing `CondensateNetPipeline.from_local()` without any changes to the inference code.

### Existing Package Files

| File | Contents |
|------|----------|
| `condensatenet/__init__.py` | Exports: `CondensateNetPipeline`, `normalize_image`, `flow_to_instances`, `download_model`, `load_model`, etc. |
| `condensatenet/pipeline.py` | `CondensateNetPipeline` class with `from_pretrained()`, `from_local()`, `segment()`, `predict()`, `postprocess()` |
| `condensatenet/model.py` | `load_model()` (uses `AutoModel.from_pretrained`), `download_model()` (from HuggingFace) |
| `condensatenet/preprocessing.py` | `normalize_image(image, low_percentile=0.5, high_percentile=99.5)`, `prepare_tensor(image, device)` → `(1,1,H,W)` |
| `condensatenet/postprocessing.py` | `flow_to_instances(mask_probs, flows, ...)` — flow integration + clustering |
| `condensatenet/utils.py` | `get_device()`, `get_cache_dir()` |
| `pyproject.toml` | Dependencies: `torch>=2.0.0`, `transformers>=4.30.0`, `timm>=0.9.0`, `numpy>=1.21.0`, `scikit-image>=0.19.0`, `huggingface-hub>=0.16.0` |

---

## Original Training Details (from `extra_stuff/condensate_28_12_2025.ipynb`)

The notebook in `extra_stuff/condensate_28_12_2025.ipynb` contains the full original training pipeline (v2.1). These details are critical for implementing compatible fine-tuning.

### Flow Vector Convention — UNIT-NORMALIZED RADIAL FLOWS

**This was the key unknown, now resolved.** The original training uses `FlowGenerator` (simple radial flows) which produces **unit-normalized** vectors:

```python
class FlowGenerator:
    """Simple radial flow generator - fast, works for round objects."""

    def generate(self, mask):
        flows = np.zeros((2, *mask.shape), dtype=np.float32)
        for lbl in np.unique(mask):
            if lbl == 0: continue
            slices = find_objects(mask == lbl)
            if not slices or slices[0] is None: continue
            bbox = slices[0]
            inst = (mask[bbox] == lbl)
            if inst.sum() < self.min_size: continue

            yi, xi = np.nonzero(inst)
            cy, cx = int(yi.mean()), int(xi.mean())

            # Ensure center is inside instance
            if not inst[cy, cx]:
                d = (xi - cx)**2 + (yi - cy)**2
                cy, cx = yi[d.argmin()], xi[d.argmin()]

            # Simple radial flow toward center
            yy, xx = np.meshgrid(np.arange(inst.shape[0]),
                                 np.arange(inst.shape[1]), indexing='ij')
            dy, dx = cy - yy, cx - xx
            mag = np.sqrt(dy**2 + dx**2 + 1e-8)
            dy, dx = dy / mag, dx / mag              # <-- UNIT NORMALIZED
            flows[0][bbox][inst] = dy[inst]
            flows[1][bbox][inst] = dx[inst]

        return flows
```

**Key details:**
- Flow vectors are **unit-length** (normalized by magnitude + 1e-8 epsilon)
- Direction: **from pixel toward centroid** (`cy - yy`, `cx - xx`)
- Format: `flows[0]` = dy, `flows[1]` = dx
- Background pixels: zero flow
- Small instances (< `min_instance_size=3`) are skipped
- If centroid falls outside the instance mask, snap to nearest instance pixel
- After augmentation, flows are clipped to `[-1, 1]`

There is also a `CellposeFlowGenerator` (heat-diffusion based) for irregular shapes, but the default config uses simple radial flows (`use_cellpose = False`). For fine-tuning via NimbusImage, simple radial flows are appropriate since condensates are round objects.

### Loss Function — WeightedBCEDiceLoss

The original training uses `WeightedBCEDiceLoss` (recommended for extreme class imbalance where foreground is ~0.05% of pixels):

```python
class WeightedBCEDiceLoss(nn.Module):
    def forward(self, pred, target):
        pred_mask = pred['mask']       # (B, 1, H, W) logits
        pred_flows = pred['flows']     # (B, 2, H, W)
        gt_mask = target['mask']       # (B, 1, H, W) binary
        gt_flows = target['flows']     # (B, 2, H, W)

        # 1. Weighted BCE loss (high pos_weight for class imbalance)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_mask, gt_mask,
            pos_weight=torch.tensor([pos_weight], device=pred_mask.device)
        )

        # 2. Dice loss
        pred_prob = torch.sigmoid(pred_mask.clamp(-20, 20))
        intersection = (pred_prob * gt_mask).sum()
        union = pred_prob.sum() + gt_mask.sum()
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

        # 3. Flow loss (MSE, foreground-only)
        flow_diff = (pred_flows - gt_flows) ** 2
        flow_weight_mask = gt_mask.expand(-1, 2, -1, -1)    # expand (B,1,H,W) → (B,2,H,W)
        flow_weight_sum = flow_weight_mask.sum()
        if flow_weight_sum > 0:
            flow_loss = (flow_diff * flow_weight_mask).sum() / (flow_weight_sum + smooth)
            flow_loss = flow_loss * flow_weight * flow_scale
        else:
            flow_loss = 0.0

        # Combined
        total = bce_weight * bce_loss + dice_weight * dice_loss + flow_loss
        return total, {'bce': ..., 'dice': ..., 'flow': ..., 'total': ...}
```

**Default loss config:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| `loss_type` | `'weighted_bce_dice'` | Recommended over focal_tversky |
| `pos_weight` | `500.0` | ~inverse of foreground ratio; critical for class imbalance |
| `bce_weight` | `1.0` | |
| `dice_weight` | `0.5` | |
| `flow_weight` | `1.0` | |
| `flow_scale` | `3.0` | Multiplied with flow_weight; total flow contribution = 3.0 |
| `smooth` | `1e-6` | Epsilon for dice denominator |

**Note on `pos_weight`**: For fine-tuning via NimbusImage where users define specific training regions around condensates, the foreground ratio will be higher than original training. A lower `pos_weight` (e.g., 50-100) may be more appropriate. Consider making this configurable or auto-computing from the training data.

### Training Configuration

| Parameter | Original Value | Notes |
|-----------|---------------|-------|
| `optimizer` | AdamW | |
| `learning_rate` | `2e-4` | For fine-tuning, lower (e.g., `1e-4`) may be better |
| `weight_decay` | `1e-4` | |
| `batch_size` | `16` | May need to be smaller (4-8) for NimbusImage Docker workers |
| `scheduler` | `CosineAnnealingWarmRestarts(T_0=20, T_mult=2)` | |
| `warmup_epochs` | `5` | Linear warmup: `lr * (epoch+1) / warmup_epochs` |
| `gradient_clip_value` | `5.0` | `clip_grad_norm_` |
| `mixed_precision` | `True` | Uses `torch.amp.autocast` + `GradScaler` |
| `num_epochs` | `150` | For fine-tuning, fewer (50-100) likely sufficient |
| `early_stopping_patience` | `30` | Based on val dice score |
| `min_lr` | `1e-6` | |

### Dataset & Tiling

The original training uses `CondensateDataset` with:

**Foreground-biased tile sampling** (`RandomTiling`, critical for sparse data):
```python
class RandomTiling:
    def __init__(self, tile_size=256, fg_bias=1.0):
        self.ts = tile_size
        self.fg = fg_bias  # 1.0 = always sample around foreground

    def sample(self, mask, key=None):
        h, w = mask.shape
        mt, ml = max(0, h - self.ts), max(0, w - self.ts)
        if mask.max() > 0 and np.random.random() < self.fg:
            fg = np.argwhere(mask > 0)
            cy, cx = fg[np.random.randint(len(fg))]  # random FG pixel
            jy = np.random.randint(-self.ts//3, self.ts//3)  # jitter
            jx = np.random.randint(-self.ts//3, self.ts//3)
            t = int(np.clip(cy - self.ts//2 + jy, 0, mt))
            l = int(np.clip(cx - self.ts//2 + jx, 0, ml))
            return t, l
        # Fallback to random
        return random_position(...)
```

**Key dataset behavior:**
- Images normalized per-tile: `percentile(0.1, 99.9)` clipped to `[0, 1]`
- Mask converted to binary: `(mask > 0).astype(float32)` — shape `(1, H, W)`
- Flows: shape `(2, H, W)`, clipped to `[-1, 1]`
- Tile size: 256x256 (must be divisible by 32 for FPN)
- Padding: zero-pad if image smaller than tile size
- Returns dict: `{'image': (1,H,W), 'mask': (1,H,W), 'flows': (2,H,W)}`

### Augmentation

Uses albumentations with special flow handling:

```python
self.transform = A.Compose(
    transforms,
    additional_targets={'mask': 'mask', 'flows': 'image'}  # flows treated as image for spatial transforms
)
```

**Augmentation levels** (selected by dataset size):
- **Light** (>20 images): flip 0.3, rotation 5deg, no elastic
- **Medium** (5-20 images): flip 0.5, rotation 15deg, elastic(30, 4, p=0.15), brightness/contrast 0.5
- **Heavy** (<5 images): flip 0.7, rotation 20deg, elastic(50, 5, p=0.25), brightness/contrast 0.6

**Flow augmentation** (fallback without albumentations):
- Horizontal flip: `flw = np.fliplr(flw).copy(); flw[1] = -flw[1]` (negate dx)
- Vertical flip: `flw = np.flipud(flw).copy(); flw[0] = -flw[0]` (negate dy)
- After any augmentation: `np.clip(np.nan_to_num(flw), -1, 1)`

### Training Loop Details

```python
# In _train_epoch():
self.model.train()
for batch in self.train_loader:
    images = batch['image'].to(self.device)       # (B, 1, H, W)
    masks = batch['mask'].to(self.device)          # (B, 1, H, W)
    flows = batch['flows'].to(self.device)         # (B, 2, H, W)

    self.optimizer.zero_grad(set_to_none=True)

    with autocast(device_type=self.device.type, enabled=mixed_precision):
        predictions = self.model(images)           # {'mask': (B,1,H,W), 'flows': (B,2,H,W)}
        loss, _ = self.loss_fn(predictions, {'mask': masks, 'flows': flows})

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_value)
    self.scaler.step(self.optimizer)
    self.scaler.update()
```

**Model saving** (original training saves as checkpoint):
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': self.model.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'val_dice': val_dice,
    'config': self.cfg
}, self.save_dir / 'best_model.pt')
```

**For our fine-tuning API**, we should instead use `model.save_pretrained(output_path)` which saves in HuggingFace format (`config.json` + `model.safetensors`), directly compatible with `CondensateNetPipeline.from_local()`. This avoids needing to strip state_dict prefixes or reconstruct the model.

---

## What Needs to Be Built

### 1. `condensatenet/data.py` — Training Data Preparation

#### `labels_to_flows(label_mask: np.ndarray) -> np.ndarray`

Convert instance segmentation label mask to **unit-normalized radial flow** vectors, exactly matching the original `FlowGenerator`.

```python
from scipy.ndimage import find_objects

def labels_to_flows(label_mask, min_size=3):
    """Convert instance segmentation mask to unit-normalized flow vectors.

    For each foreground pixel, computes a unit vector pointing toward
    the centroid of its instance.

    Args:
        label_mask: Integer array (H, W). 0=background, 1,2,...=instances.
        min_size: Skip instances smaller than this (pixels).

    Returns:
        flows: Float32 array (2, H, W). flows[0]=dy, flows[1]=dx.
               Unit-length vectors on foreground, zeros on background.
    """
    flows = np.zeros((2, *label_mask.shape), dtype=np.float32)

    for lbl in np.unique(label_mask):
        if lbl == 0:
            continue
        slices = find_objects(label_mask == lbl)
        if not slices or slices[0] is None:
            continue
        bbox = slices[0]
        inst = (label_mask[bbox] == lbl)
        if inst.sum() < min_size:
            continue

        yi, xi = np.nonzero(inst)
        if len(yi) == 0:
            continue
        cy, cx = int(yi.mean()), int(xi.mean())

        # Ensure center is inside instance
        if not inst[cy, cx]:
            d = (xi - cx)**2 + (yi - cy)**2
            cy, cx = yi[d.argmin()], xi[d.argmin()]

        # Radial flow toward center, unit-normalized
        yy, xx = np.meshgrid(
            np.arange(inst.shape[0]),
            np.arange(inst.shape[1]),
            indexing='ij'
        )
        dy, dx = cy - yy, cx - xx
        mag = np.sqrt(dy**2 + dx**2 + 1e-8)
        dy, dx = dy / mag, dx / mag

        flows[0][bbox][inst] = dy[inst]
        flows[1][bbox][inst] = dx[inst]

    return flows
```

**Critical**: Uses `scipy.ndimage.find_objects` with bounding box slicing for efficiency (same as original). Add `scipy` as a dependency if not already present.

#### `labels_to_binary_mask(label_mask: np.ndarray) -> np.ndarray`

```python
def labels_to_binary_mask(label_mask):
    return (label_mask > 0).astype(np.float32)
```

#### `CondensateNetDataset` (PyTorch Dataset)

```python
class CondensateNetDataset(Dataset):
    """Training dataset for CondensateNet fine-tuning.

    Each sample returns a dict:
        'image': (1, H, W) float32 tensor, percentile-normalized
        'mask':  (1, H, W) float32 tensor, binary foreground
        'flows': (2, H, W) float32 tensor, unit-normalized radial flows
    """
    def __init__(self, images, label_masks, patch_size=256,
                 foreground_bias=1.0, augment=True, min_size=3,
                 samples_per_epoch=None, rng=None):
        """
        Args:
            images: list of 2D numpy arrays (raw microscopy images)
            label_masks: list of integer arrays (instance segmentation)
            patch_size: crop size (must be divisible by 32)
            foreground_bias: probability of centering crop on foreground (0-1, default 1.0)
            augment: apply random flips (horizontal, vertical)
            min_size: skip instances smaller than this in flow generation
            samples_per_epoch: number of samples per epoch (default: len(images) * 100)
            rng: numpy random Generator for reproducibility
        """
```

**Key implementation details:**

1. **Flow computation**: Pre-compute `labels_to_flows()` for each label mask in `__init__` (or lazily with caching). Flows are computed on the full image, then cropped to match the tile.

2. **Foreground-biased tiling** (from `RandomTiling`): Pick a random foreground pixel, center the crop around it with random jitter of `±patch_size//3`. This ensures training patches almost always contain condensates. Fall back to random crop if no foreground (handles edge case of empty masks).

3. **Normalization**: Per-tile percentile normalization matching `preprocessing.normalize_image()`:
   ```python
   img_float = tile.astype(np.float32)
   low, high = np.percentile(img_float, [0.5, 99.5])
   normalized = np.clip((img_float - low) / (high - low + 1e-8), 0, 1)
   ```

4. **Augmentation**: For fine-tuning simplicity, use numpy-based augmentation (no albumentations dependency):
   - Random horizontal flip: flip image, mask, flows; negate `flows[1]` (dx)
   - Random vertical flip: flip image, mask, flows; negate `flows[0]` (dy)
   - After augmentation: `np.clip(np.nan_to_num(flows), -1, 1)`

5. **Padding**: If image smaller than patch_size, zero-pad image/mask/flows.

6. **Return format**: Dict `{'image': tensor(1,H,W), 'mask': tensor(1,H,W), 'flows': tensor(2,H,W)}` matching what the loss function expects.

7. **`__len__`**: Return `samples_per_epoch` (default `len(images) * 100`). Each call to `__getitem__` randomly selects an image and crop position, so the same image is sampled many times with different crops/augmentations.

#### `prepare_training_data(images, label_masks, ...) -> CondensateNetDataset`

Convenience function that validates inputs and returns a `CondensateNetDataset`:
- Check all images are 2D numpy arrays
- Check label_masks have matching shapes
- Check patch_size is divisible by 32
- Warn if any label_mask has no instances

---

### 2. `condensatenet/training.py` — Training Loop

#### `WeightedBCEDiceLoss`

Implement the exact loss function from the original training:

```python
class WeightedBCEDiceLoss(nn.Module):
    """Combined weighted BCE + Dice + Flow loss for CondensateNet.

    Designed for extreme class imbalance (condensates are ~0.05% of pixels).

    Args:
        pos_weight: BCE positive class weight (default 100.0 for fine-tuning)
        bce_weight: Weight for BCE component (default 1.0)
        dice_weight: Weight for Dice component (default 0.5)
        flow_weight: Weight for flow MSE component (default 1.0)
        flow_scale: Multiplier for flow loss (default 3.0)
        smooth: Epsilon for Dice denominator (default 1e-6)
    """
    def __init__(self, pos_weight=100.0, bce_weight=1.0, dice_weight=0.5,
                 flow_weight=1.0, flow_scale=3.0, smooth=1e-6):
        super().__init__()
        self.pos_weight = pos_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.flow_weight = flow_weight
        self.flow_scale = flow_scale
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: dict with 'mask' (B,1,H,W) logits and 'flows' (B,2,H,W)
            target: dict with 'mask' (B,1,H,W) binary and 'flows' (B,2,H,W)

        Returns:
            (total_loss, loss_dict) where loss_dict has 'bce', 'dice', 'flow', 'total'
        """
        pred_mask = pred['mask']
        pred_flows = pred['flows']
        gt_mask = target['mask']
        gt_flows = target['flows']

        # Weighted BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_mask, gt_mask,
            pos_weight=torch.tensor([self.pos_weight], device=pred_mask.device)
        )

        # Dice loss
        pred_prob = torch.sigmoid(pred_mask.clamp(-20, 20))
        intersection = (pred_prob * gt_mask).sum()
        union = pred_prob.sum() + gt_mask.sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)

        # Flow loss (foreground-masked MSE)
        flow_diff = (pred_flows - gt_flows) ** 2
        fg_mask = gt_mask.expand(-1, 2, -1, -1)
        fg_sum = fg_mask.sum()
        if fg_sum > 0:
            flow_loss = (flow_diff * fg_mask).sum() / (fg_sum + self.smooth)
            flow_loss = flow_loss * self.flow_weight * self.flow_scale
        else:
            flow_loss = torch.tensor(0.0, device=pred_mask.device)

        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss + flow_loss
        return total, {
            'bce': bce_loss.detach().item(),
            'dice': dice_loss.detach().item(),
            'flow': flow_loss.detach().item() if isinstance(flow_loss, torch.Tensor) else flow_loss,
            'total': total.detach().item()
        }
```

**Note on `pos_weight`**: Original training uses 500.0 because training images have ~0.05% foreground. For NimbusImage fine-tuning where users define small regions densely packed with condensates, a lower value like 50-100 is more appropriate. Could auto-compute from training data as `(num_bg_pixels / num_fg_pixels)`.

#### `train_model()`

```python
def train_model(
    dataset,                          # CondensateNetDataset
    initial_model_path: str,          # Path to pretrained model dir
    output_path: str,                 # Where to save fine-tuned model
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    warmup_epochs: int = 5,
    gradient_clip: float = 5.0,
    mixed_precision: bool = True,
    pos_weight: float = 100.0,
    device: Optional[str] = None,
    random_seed: int = 42,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
```

**Training loop implementation** (follows original `Trainer` class):

1. **Load model**: `model = load_model(initial_model_path, device=device)` then `model.train()`

2. **Optimizer**: `AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)`

3. **Warmup**: Linear warmup for first `warmup_epochs`:
   ```python
   if epoch < warmup_epochs:
       warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
       for param_group in optimizer.param_groups:
           param_group['lr'] = warmup_lr
   ```

4. **Scheduler**: `CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)` (applied after warmup period)

5. **Mixed precision**: `torch.amp.autocast` + `GradScaler` (enabled by default, significant speedup on GPU)

6. **Gradient clipping**: `clip_grad_norm_(model.parameters(), gradient_clip)` after `scaler.unscale_(optimizer)`

7. **DataLoader**: `DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device=='cuda'))`

8. **Training step** (per batch):
   ```python
   optimizer.zero_grad(set_to_none=True)
   with autocast(device_type=device.type, enabled=mixed_precision):
       predictions = model(images)
       loss, loss_dict = criterion(predictions, {'mask': masks, 'flows': flows})
   scaler.scale(loss).backward()
   scaler.unscale_(optimizer)
   clip_grad_norm_(model.parameters(), gradient_clip)
   scaler.step(optimizer)
   scaler.update()
   ```

9. **Save model**: `model.save_pretrained(output_path)` — saves `config.json` + `model.safetensors` in HuggingFace format, compatible with existing `CondensateNetPipeline.from_local()`.

10. **Progress**: Call `progress_callback(fraction, message)` each epoch if provided.

---

### 3. Updates to Existing Files

#### `condensatenet/__init__.py`

Add new exports:

```python
from .training import train_model, WeightedBCEDiceLoss
from .data import prepare_training_data, CondensateNetDataset, labels_to_flows, labels_to_binary_mask

__all__ = [
    # ... existing exports ...

    # Training
    "train_model",
    "WeightedBCEDiceLoss",

    # Data preparation
    "prepare_training_data",
    "CondensateNetDataset",
    "labels_to_flows",
    "labels_to_binary_mask",
]
```

#### `condensatenet/utils.py`

Add a `MODELS_DIR` constant:

```python
from pathlib import Path

MODELS_DIR = Path.home() / '.condensatenet' / 'models'
```

#### `pyproject.toml`

Add `scipy` to dependencies (needed for `find_objects` in `labels_to_flows`):

```toml
dependencies = [
    # ... existing ...
    "scipy>=1.7.0",
]
```

---

### 4. File Summary

| File | Action | Description |
|------|--------|-------------|
| `condensatenet/data.py` | **CREATE** | `labels_to_flows()`, `labels_to_binary_mask()`, `CondensateNetDataset`, `prepare_training_data()` |
| `condensatenet/training.py` | **CREATE** | `WeightedBCEDiceLoss`, `train_model()` |
| `condensatenet/__init__.py` | **MODIFY** | Add exports for training and data modules |
| `condensatenet/utils.py` | **MODIFY** | Add `MODELS_DIR` constant |
| `pyproject.toml` | **MODIFY** | Add `scipy` dependency |

---

### 5. Testing

- **Unit test `labels_to_flows()`**: Create synthetic mask with two squares, verify:
  - Flow vectors are unit-length on foreground
  - Flow vectors point toward centroids
  - Background pixels have zero flow
  - Small instances below `min_size` are skipped
  - Centroid-outside-mask snapping works

- **Unit test `CondensateNetDataset`**: Verify:
  - Output shapes: `image (1,H,W)`, `mask (1,H,W)`, `flows (2,H,W)`
  - Image values in [0, 1] after normalization
  - Flow values in [-1, 1]
  - Foreground-biased sampling: most crops contain foreground
  - Augmentation: flips correctly negate flow components

- **Integration test**: Create synthetic data, run `train_model()` for 2-3 epochs, verify:
  - Model saves in HuggingFace format (`config.json` + `model.safetensors`)
  - `CondensateNetPipeline.from_local(output_path)` loads without error
  - Loaded model produces output with correct shapes
  - `model.train()` / `model.eval()` switching works (batch norm, dropout)

- **Loss function test**: Verify `WeightedBCEDiceLoss` with known inputs:
  - Perfect prediction → low loss
  - All-zero prediction → high bce + dice, zero flow
  - Flow loss zero when no foreground

### 6. Reference Files

- **Original training pipeline**: `extra_stuff/condensate_28_12_2025.ipynb` — full v2.1 training code
- **Model architecture**: `extra_stuff/minimal_condensate_segmentation.ipynb` — standalone model definition
- **Inference demo**: `extra_stuff/condensatenet_minimal_demo.ipynb` — HuggingFace loading + inference
- **NimbusImage worker plan**: `ImageAnalysisProject/workers/annotations/condensatenet/todo/RETRAINING.md`
