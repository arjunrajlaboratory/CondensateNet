"""Upload trained CondensateNet model to Hugging Face Hub.

Uploads in transformers-compatible format so that the condensatenet package
can load via AutoModel.from_pretrained(). This requires:
  - model.safetensors (weights)
  - config.json (with auto_map pointing to custom classes)
  - configuration_condensatenet.py (PretrainedConfig subclass)
  - modeling_condensatenet.py (PreTrainedModel subclass)
  - README.md (model card)
"""

import json
import torch
from pathlib import Path
from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download
from safetensors.torch import save_file


REPO_ID = "rajlab/condensatenet-v1"
COLLECTION_SLUG = "rajlab/condensatenet-6945cf98e22aa0be561d9906"

# The original repo has the transformers-compatible model/config classes
# that we reuse (same architecture, different weights)
SOURCE_REPO_ID = "rajlab/condensatenet"


def create_model_card(checkpoint: dict) -> ModelCard:
    """Generate a model card with training metadata."""
    card_data = ModelCardData(
        language="en",
        license="mit",
        library_name="pytorch",
        tags=[
            "image-segmentation",
            "microscopy",
            "condensate-detection",
            "cellpose-flows",
            "efficientnetv2",
        ],
        metrics=[
            {"type": "dice", "value": round(checkpoint["val_dice"], 4)},
            {"type": "f1", "value": round(checkpoint["val_f1"], 4)},
            {"type": "iou", "value": round(checkpoint["val_iou"], 4)},
        ],
    )

    content = f"""\
---
{card_data.to_yaml()}
---

# CondensateNet v1

Semantic segmentation model for detecting biomolecular condensates (puncta / stress granules) in fluorescence microscopy images.

## Model Description

- **Architecture**: EfficientNetV2-S encoder with style-modulated Feature Pyramid Network (FPN) and dual output heads (binary mask + Cellpose-style flow fields)
- **Parameters**: 22.2M
- **Input**: Single-channel grayscale microscopy tile (320x320), percentile-normalized to [0, 1]
- **Output**: Binary segmentation mask logits + 2-channel flow field

## Training

- **Data**: 245 validated image-mask pairs from 14 experiment types (fluorescence microscopy of various condensate-forming proteins)
- **Split**: 158 train / 26 val (stratified by experiment, with caps on overrepresented experiments)
- **Loss**: Focal + Tversky + flow MSE + differentiable calibration (soft-binned ECE)
- **Optimizer**: AdamW with linear warmup (5 epochs) + cosine annealing (120 epochs total, early stopped at epoch 94)
- **Augmentation**: Experiment-aware augmentation tiers (light/medium/heavy) with strong contrast scaling, CLAHE, intensity scaling (0.3x-2.0x), geometric transforms

### Metrics (validation set)

| Metric | Value |
|--------|-------|
| Dice | {checkpoint["val_dice"]:.4f} |
| F1 | {checkpoint["val_f1"]:.4f} |
| IoU | {checkpoint["val_iou"]:.4f} |

### Inference metrics (full validation set, per-tile with threshold search)

| Metric | Mean | Std |
|--------|------|-----|
| Dice | 0.534 | 0.155 |
| Precision | 0.517 | 0.151 |
| Recall | 0.588 | 0.203 |

## Usage

```python
from condensatenet import CondensateNetPipeline

# Load from HuggingFace
pipeline = CondensateNetPipeline.from_pretrained("rajlab/condensatenet-v1")

# Segment an image
instances = pipeline.segment(image)
print(f"Found {{instances.max()}} condensates")

# Or load from a local download
pipeline = CondensateNetPipeline.from_local("/path/to/model")
```

## Intended Use

Research tool for detecting biomolecular condensates in fluorescence microscopy. Not intended for clinical or diagnostic use.

## Training Runs

This model was iteratively improved over 5 training runs:

| Run | Changes | Val Dice |
|-----|---------|----------|
| 1 | Baseline | 0.503 |
| 2 | +contrast augmentation | 0.480 |
| 3 | +calibration loss | 0.509 |
| 4 | +warmup/cosine/early stop (80 ep) | 0.495 |
| 4b | Same, 120 epochs | **0.516** |

Full training logs are available in the [GitHub repository](https://github.com/arjunrajlaboratory/CondensateNet).
"""
    return ModelCard(content)


def create_config_json(checkpoint: dict) -> dict:
    """Create transformers-compatible config.json.

    Must include auto_map so AutoModel.from_pretrained() can find the
    custom configuration and model classes.
    """
    return {
        "architectures": ["CondensateNet"],
        "model_type": "condensatenet",
        "auto_map": {
            "AutoConfig": "configuration_condensatenet.CondensateNetConfig",
            "AutoModel": "modeling_condensatenet.CondensateNet",
        },
        "encoder_variant": "rw_s",
        "pyramid_channels": [24, 48, 64, 160],
        "pyramid_dim": 32,
        "use_spatial_attention": True,
        "spatial_kernel_size": 11,
        "dropout_rate": 0.15,
        "num_classes": 1,
        "dtype": "float32",
        "training": {
            "num_epochs_run": checkpoint["epoch"] + 1,
            "loss": "focal + tversky + flow_mse + calibration_ece",
            "calibration_weight": 0.5,
            "optimizer": "AdamW",
            "lr": 2e-4,
            "warmup_epochs": 5,
            "scheduler": "cosine_annealing",
            "early_stopping_patience": 25,
        },
        "metrics": {
            "val_dice": checkpoint["val_dice"],
            "val_f1": checkpoint["val_f1"],
            "val_iou": checkpoint["val_iou"],
        },
    }


def main():
    checkpoint_path = Path("data/processed/checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    print(f"  Epoch: {checkpoint['epoch'] + 1}, Val Dice: {checkpoint['val_dice']:.4f}")

    api = HfApi()

    # 1. Create repo
    print(f"\nCreating repo {REPO_ID}...")
    api.create_repo(repo_id=REPO_ID, exist_ok=True)

    # 2. Upload model card
    print("Uploading model card...")
    card = create_model_card(checkpoint)
    card.push_to_hub(REPO_ID)

    # 3. Upload config.json (transformers-compatible with auto_map)
    print("Uploading config.json...")
    config_json = create_config_json(checkpoint)
    config_path = Path("/tmp/condensatenet_config.json")
    config_path.write_text(json.dumps(config_json, indent=2))
    api.upload_file(
        path_or_fileobj=str(config_path),
        path_in_repo="config.json",
        repo_id=REPO_ID,
    )

    # 4. Copy custom model/config classes from the original repo
    # (same architecture, just different trained weights)
    print(f"Copying model classes from {SOURCE_REPO_ID}...")
    for filename in ["configuration_condensatenet.py", "modeling_condensatenet.py"]:
        local_path = hf_hub_download(repo_id=SOURCE_REPO_ID, filename=filename)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=REPO_ID,
        )
        print(f"  Uploaded {filename}")

    # 5. Save weights as safetensors and upload
    print("Converting to safetensors format...")
    state_dict = checkpoint["model_state_dict"]
    safetensors_path = Path("/tmp/model.safetensors")
    save_file(state_dict, safetensors_path)
    size_mb = safetensors_path.stat().st_size / 1024 / 1024
    print(f"  model.safetensors: {size_mb:.1f} MB")

    print("Uploading model.safetensors...")
    api.upload_file(
        path_or_fileobj=str(safetensors_path),
        path_in_repo="model.safetensors",
        repo_id=REPO_ID,
    )

    # 6. Clean up old .pt file if it exists from a previous upload
    print("Cleaning up old .pt file if present...")
    try:
        api.delete_file(path_in_repo="best_model.pt", repo_id=REPO_ID)
        print("  Removed old best_model.pt")
    except Exception:
        pass

    # 7. Add to collection
    print(f"\nAdding to collection {COLLECTION_SLUG}...")
    try:
        api.add_collection_item(
            collection_slug=COLLECTION_SLUG,
            item_id=REPO_ID,
            item_type="model",
        )
        print("  Added to collection.")
    except Exception as e:
        print(f"  Could not add to collection: {e}")
        print("  You may need to add it manually at https://huggingface.co/collections/rajlab/condensatenet")

    print(f"\nDone! Model available at https://huggingface.co/{REPO_ID}")
    print("\nUploaded files:")
    for f in api.list_repo_files(REPO_ID):
        print(f"  {f}")


if __name__ == "__main__":
    main()
