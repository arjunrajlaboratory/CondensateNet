# CondensateNet Retraining: Package Training API

## Overview

Add training/fine-tuning capability to the condensatenet Python package so that users can retrain the model on their own annotated data via NimbusImage. This is the **first step** — the package-side training API must be in place before the NimbusImage training worker can be built (see the companion `RETRAINING.md` in `ImageAnalysisProject/workers/annotations/condensatenet/todo/`).

## Background

The CondensateNet model is an FPN (Feature Pyramid Network) with EfficientNetV2-S encoder that produces two outputs:
- `out['mask']` — binary segmentation logits (sigmoid → probability map)
- `out['flows']` — 2D flow vectors pointing toward instance centers

Inference pipeline: `preprocess (percentile normalize) → model forward → flow integration → instance clustering → size filtering`

The model is loaded via HuggingFace `AutoModel.from_pretrained()` and saved as `config.json` + `model.safetensors`. Training must produce models in the same format so they can be loaded by the existing `CondensateNetPipeline.from_local()` without any changes to the inference code.

## What Needs to Be Built

### 1. `condensatenet/data.py` — Training Data Preparation

This module converts raw images + instance segmentation masks into a PyTorch Dataset suitable for training.

#### `labels_to_flows(label_mask: np.ndarray) -> np.ndarray`

Convert an instance segmentation label mask to ground-truth flow vectors. This is the inverse of the `flow_to_instances()` postprocessing in `postprocessing.py`.

- **Input**: Integer array of shape `(H, W)` where 0=background, 1,2,3,...=instances
- **Output**: Float array of shape `(2, H, W)` with `(dy, dx)` flow vectors
- **Logic**: For each foreground pixel, compute a flow vector pointing toward the centroid of its instance. Background pixels get zero flow.
- **Reference**: The postprocessing in `postprocessing.py` (lines 50-68) shows how flow integration works — pixels follow flow vectors to converge at instance centers. The training target is the reverse: vectors from pixels toward their instance center.

```python
def labels_to_flows(label_mask):
    flows = np.zeros((2, *label_mask.shape), dtype=np.float32)
    for inst_id in range(1, label_mask.max() + 1):
        mask = label_mask == inst_id
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        cy, cx = ys.mean(), xs.mean()  # centroid
        # Flow = direction from pixel to centroid
        flows[0, mask] = cy - ys  # dy
        flows[1, mask] = cx - xs  # dx
        # Normalize by distance (or cap magnitude — TBD based on testing)
    return flows
```

**Design question**: Should flow vectors be normalized to unit length, scaled by distance, or capped at some maximum magnitude? The original training likely used a specific convention — check what `flow_to_instances()` in `postprocessing.py` expects. The flow integration uses adaptive step sizes (`step_scale / (flow_mag + 1e-6)`), so vectors with larger magnitude near the boundary and smaller near the centroid would be natural.

#### `labels_to_binary_mask(label_mask: np.ndarray) -> np.ndarray`

Simple conversion: `(label_mask > 0).astype(np.float32)` → binary foreground mask target for the mask head.

#### `CondensateNetDataset` (PyTorch Dataset)

```python
class CondensateNetDataset(Dataset):
    """
    Training dataset for CondensateNet.

    Each item returns:
        image_tensor: (1, H, W) normalized float tensor
        mask_target:  (1, H, W) binary mask target
        flow_target:  (2, H, W) flow vector target
    """
    def __init__(self, images, label_masks, patch_size=256, augment=True, rng=None):
        # images: list of 2D numpy arrays (raw microscopy images)
        # label_masks: list of integer arrays (instance segmentation, same shapes)
        # patch_size: size of random crops (must be divisible by 32 for FPN)
        # augment: apply random flips and 90-degree rotations
        ...
```

**Key implementation details**:
- **Random cropping**: Each `__getitem__` call picks a random image, crops a `patch_size x patch_size` region. Ensure patches contain at least some foreground (or accept some empty patches for negative examples).
- **Normalization**: Use `preprocessing.normalize_image()` with percentile normalization (0.5 to 99.5) — must match inference preprocessing exactly.
- **Augmentation**: Random horizontal/vertical flips, random 90-degree rotations. These are safe for flow vectors if you transform the flows to match (flip dy/dx signs and swap channels appropriately).
- **Flow computation**: Call `labels_to_flows()` on the cropped label mask.
- **`__len__`**: Should return a reasonable number per epoch. Suggestion: `num_images * (mean_image_area / patch_area) * augmentation_multiplier`. Or a fixed value like `len(images) * 100`.

**Augmentation of flow vectors**: When flipping/rotating images, the flow targets must be transformed consistently:
- Horizontal flip: negate dx (flows[1]), flip both arrays along axis=-1
- Vertical flip: negate dy (flows[0]), flip both arrays along axis=-2
- 90-degree rotation: swap dy/dx channels, adjust signs

#### `prepare_training_data(images, label_masks, patch_size=256, augment=True, random_seed=42) -> CondensateNetDataset`

Convenience function that creates and returns a `CondensateNetDataset`. Validates inputs (matching shapes, non-empty masks, patch_size divisible by 32).

---

### 2. `condensatenet/training.py` — Training Loop

#### `CondensateNetLoss`

Combined loss for the two model heads:

```python
class CondensateNetLoss(nn.Module):
    """
    Loss = mask_weight * BCE(mask_logits, mask_target)
         + flow_weight * MSE(flow_pred, flow_target)  [foreground only]
    """
    def __init__(self, mask_weight=1.0, flow_weight=1.0):
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, mask_logits, flow_pred, mask_target, flow_target):
        mask_loss = self.bce(mask_logits, mask_target)

        # Flow loss only on foreground pixels (background flow is undefined)
        fg_mask = mask_target.expand_as(flow_pred)  # (B, 2, H, W)
        flow_loss_map = self.mse(flow_pred, flow_target)
        flow_loss = (flow_loss_map * fg_mask).sum() / (fg_mask.sum() + 1e-8)

        total = mask_weight * mask_loss + flow_weight * flow_loss
        return {'total': total, 'mask_loss': mask_loss, 'flow_loss': flow_loss}
```

**Design notes**:
- BCE with logits for the mask head (model outputs raw logits, sigmoid is applied in inference via `torch.sigmoid(out['mask'])` in `pipeline.py` line 195)
- MSE for flow vectors, masked to foreground only — background pixels have no meaningful flow target
- The weighting between mask and flow losses may need tuning. Start with equal weights (1.0 each) and adjust if one loss dominates.

#### `train_model()`

```python
def train_model(
    dataset: CondensateNetDataset,
    initial_model_path: str,     # Path to pretrained model dir (config.json + model.safetensors)
    output_path: str,            # Where to save fine-tuned model
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
    warmup_fraction: float = 0.1,
    mask_weight: float = 1.0,
    flow_weight: float = 1.0,
    device: Optional[str] = None,
    random_seed: int = 42,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
```

**Training loop details**:

1. **Load model**: `model = load_model(initial_model_path, device=device)` using existing `model.py:load_model()` (line 68). Then set `model.train()`.

2. **Optimizer**: AdamW with specified learning_rate and weight_decay. AdamW is standard for fine-tuning pretrained models.

3. **LR schedule**: Cosine annealing with linear warmup.
   ```python
   warmup_epochs = int(epochs * warmup_fraction)
   warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
   cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
   scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
   ```

4. **DataLoader**: `DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device=='cuda'))`. Use `num_workers=0` to avoid multiprocessing issues in Docker containers.

5. **Training loop**:
   ```python
   for epoch in range(epochs):
       for images, mask_targets, flow_targets in dataloader:
           images, mask_targets, flow_targets = [t.to(device) for t in ...]
           optimizer.zero_grad()
           out = model(images)  # returns {'mask': logits, 'flows': vectors}
           losses = criterion(out['mask'], out['flows'], mask_targets, flow_targets)
           losses['total'].backward()
           optimizer.step()
       scheduler.step()
       if progress_callback:
           progress_callback((epoch + 1) / epochs, f"Epoch {epoch+1}/{epochs}, Loss: ...")
   ```

6. **Save model**: `model.save_pretrained(output_path)` — this saves `config.json` + `model.safetensors` in HuggingFace format, fully compatible with `CondensateNetPipeline.from_local(output_path)`.

**Important**: The model is loaded from HuggingFace's `AutoModel.from_pretrained()` with `trust_remote_code=True` (see `model.py` line 116-119). This means it already has `save_pretrained()` available from the HuggingFace `PreTrainedModel` base class. No additional save logic needed.

**Progress callback**: The NimbusImage worker passes `sendProgress` through this callback, allowing training progress to appear in the UI. The package itself has no dependency on `annotation_client`.

---

### 3. Updates to Existing Files

#### `condensatenet/__init__.py`

Add new exports:

```python
from .training import train_model, CondensateNetLoss
from .data import prepare_training_data, CondensateNetDataset, labels_to_flows, labels_to_binary_mask

__all__ = [
    # ... existing exports ...

    # Training
    "train_model",
    "CondensateNetLoss",

    # Data preparation
    "prepare_training_data",
    "CondensateNetDataset",
    "labels_to_flows",
    "labels_to_binary_mask",
]
```

#### `condensatenet/utils.py`

Add a `MODELS_DIR` constant for local model storage (used by the NimbusImage worker for downloading/uploading models from Girder):

```python
from pathlib import Path

MODELS_DIR = Path.home() / '.condensatenet' / 'models'
```

#### `pyproject.toml`

The existing dependencies (`torch`, `numpy`, `scikit-image`, `transformers`) should cover everything needed for training. No new dependencies required. Optionally add a `[project.optional-dependencies.train]` section if any training-specific deps emerge during development.

---

### 4. File Summary

| File | Action | Description |
|------|--------|-------------|
| `condensatenet/data.py` | **CREATE** | `labels_to_flows()`, `labels_to_binary_mask()`, `CondensateNetDataset`, `prepare_training_data()` |
| `condensatenet/training.py` | **CREATE** | `CondensateNetLoss`, `train_model()` |
| `condensatenet/__init__.py` | **MODIFY** | Add exports for training and data modules |
| `condensatenet/utils.py` | **MODIFY** | Add `MODELS_DIR` constant |
| `pyproject.toml` | **POSSIBLY MODIFY** | Add `[project.optional-dependencies.train]` if needed |

---

### 5. Testing

- Unit test `labels_to_flows()` with a simple synthetic mask (e.g., two squares) — verify flow vectors point toward centroids
- Unit test `CondensateNetDataset` — verify shapes, augmentation correctness (especially flow vector transforms), normalization matches inference
- Integration test: create synthetic training data, run `train_model()` for a few epochs, verify model saves in correct format, verify `CondensateNetPipeline.from_local()` can load the saved model and produce output
- Verify `model.train()` / `model.eval()` switching works correctly (batch norm, dropout behavior)

### 6. Open Questions

1. **Flow vector convention**: Should flows be raw (centroid - pixel) or normalized? The `flow_to_instances()` postprocessing uses adaptive step sizes, so raw vectors should work. But the original training may have used a specific normalization — if possible, check what the HuggingFace model was trained with.

2. **Patch size vs full image**: Should training always use random patches, or also support full-image training for small images? Patches are more memory-efficient and provide more augmentation diversity, but full images preserve global context.

3. **Validation split**: Should `train_model()` support a validation split for monitoring overfitting? Piscis uses `train_size=1.0, test_size=0.0` (no validation). For a first version, no validation split is fine — the user can evaluate by running inference and checking results visually.

4. **Early stopping**: Not included in first version. Can be added later if users report overfitting issues.
