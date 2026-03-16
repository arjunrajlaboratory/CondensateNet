"""Configuration for the condensate segmentation training pipeline."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

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
    num_epochs: int = 120
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
