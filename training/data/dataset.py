"""Production dataset for condensate segmentation with Cellpose-style flows."""

import numpy as np
import pandas as pd
import torch
import tifffile
import warnings
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, Tuple
from PIL import Image

from ..config import PipelineConfig
from .tiling import RandomTilingStrategy, ValidationTilingStrategy, extract_tile, extract_flow_tile
from .augmentation import AugmentationFactory
from .flows import FlowGenerator


class CondensateDataset(Dataset):
    """
    Condensate segmentation dataset with Cellpose-style flows.

    FIXED:
    - Validation tiling avoids empty crops
    - Robust error handling for augmentation
    - NaN sanitization for flows
    """

    def __init__(self, manifest_path: str, split: str = "train",
                 config: Optional['PipelineConfig'] = None):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.config = config if config is not None else PipelineConfig()

        self.manifest = self._load_manifest()
        self._setup_components()

    def _load_manifest(self) -> pd.DataFrame:
        """Load and validate manifest CSV."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        required = ["image_path", "mask_path", "experiment_id"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Manifest missing: {missing}")

        return df

    def _setup_components(self):
        """
        Set up tiling, augmentation, and flow generation components.

        FIXED:
        - Uses direct tiling strategy classes
        - Augmentation uses original counts
        - Flow generation configured properly
        """
        # Tiling strategy - FIXED: ValidationTilingStrategy avoids empty crops
        if self.split == "train":
            self.tiling_strategy = RandomTilingStrategy(
                tile_size=self.config.tile_size,
                samples_per_image=1,  # One random tile per image per epoch
                foreground_bias=0.7   # 70% chance to sample near foreground
            )
        elif self.split == "val":
            self.tiling_strategy = ValidationTilingStrategy(
                tile_size=self.config.tile_size,
                min_foreground_pixels=10  # Threshold for "empty" detection
            )
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Augmentation per experiment - uses original counts
        self.augmentations = {}
        for exp_id in self.manifest["experiment_id"].unique():
            exp_samples = self.manifest[self.manifest["experiment_id"] == exp_id]
            current_count = len(exp_samples)

            # Get original count (before capping) if available
            if 'original_count' in exp_samples.columns:
                original_count = exp_samples['original_count'].iloc[0]
            else:
                original_count = current_count
                warnings.warn(
                    f"No 'original_count' column for experiment {exp_id}. "
                    f"Using current count ({current_count})."
                )

            if self.split == "train":
                # Create augmentation based on original experiment size
                self.augmentations[exp_id] = AugmentationFactory.create_from_count(
                    sample_count=current_count,
                    original_count=original_count,
                    use_albumentations=self.config.use_albumentations
                )
            else:
                self.augmentations[exp_id] = AugmentationFactory.create_no_augmentation()

        # Flow generator with Cellpose diffusion
        try:
            self.flow_generator = FlowGenerator(
                device=self.config.flow_device,
                niter=self.config.flow_niter,
                use_bbox=self.config.flow_use_bbox,
                min_size=self.config.flow_min_size
            )
        except Exception as e:
            warnings.warn(f"FlowGenerator init failed: {e}")
            self.flow_generator = None

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample with error handling."""
        try:
            return self._get_sample(idx)
        except Exception as e:
            warnings.warn(f"Sample {idx} failed: {e}. Trying next sample.")
            # Return next valid sample
            return self._get_sample((idx + 1) % len(self))

    def _get_sample(self, idx: int) -> Dict:
        """
        Load and process a single sample.

        FIXED:
        - Flow generation before augmentation
        - NaN sanitization after augmentation
        - Proper shape validation
        """
        row = self.manifest.iloc[idx]
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        experiment_id = row["experiment_id"]

        # 1. Load full image and mask
        image_full = self._load_image(image_path)
        mask_full = self._load_mask(mask_path)

        # Validate shapes
        if image_full.shape[:2] != mask_full.shape:
            raise ValueError(
                f"Shape mismatch: image {image_full.shape[:2]} vs mask {mask_full.shape}"
            )

        # 2. Generate flows from ORIGINAL mask (before tiling)
        if self.flow_generator is not None:
            flows_full = self.flow_generator.generate(mask_full)  # (2, H, W)

            # CRITICAL: Check for NaN in flow generation
            if np.isnan(flows_full).any() or np.isinf(flows_full).any():
                warnings.warn(
                    f"NaN/Inf in flow generation for {image_path.name}. "
                    f"Replacing with zeros."
                )
                flows_full = np.nan_to_num(flows_full, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            flows_full = np.zeros((2, *mask_full.shape), dtype=np.float32)

        # 3. Sample tile position - FIXED: ValidationTilingStrategy avoids empty
        cache_key = f"{self.split}_{idx}_{image_path.stem}"
        top, left = self.tiling_strategy.sample_tile(mask_full, cache_key)

        # 4. Extract tiles
        image_tile, mask_tile = extract_tile(image_full, mask_full, top, left,
                                             self.config.tile_size)
        flows_tile = extract_flow_tile(flows_full, top, left, self.config.tile_size)

        # Validate extracted tiles
        if image_tile.shape[:2] != (self.config.tile_size, self.config.tile_size):
            raise ValueError(f"Image tile wrong size: {image_tile.shape}")
        if mask_tile.shape != (self.config.tile_size, self.config.tile_size):
            raise ValueError(f"Mask tile wrong size: {mask_tile.shape}")
        if flows_tile.shape != (2, self.config.tile_size, self.config.tile_size):
            raise ValueError(f"Flow tile wrong size: {flows_tile.shape}")

        # 5. Apply augmentation to image, mask, AND flows together
        aug = self.augmentations[experiment_id]
        result = aug(image_tile, mask_tile, flows_tile)

        # Unpack results
        if len(result) == 3:
            aug_image, aug_mask, aug_flows = result
        else:
            aug_image, aug_mask = result
            aug_flows = flows_tile

        # === CRITICAL: NaN SANITIZATION AFTER AUGMENTATION ===
        # This catches any NaN that slipped through augmentation error handling
        aug_image = np.nan_to_num(aug_image, nan=0.0, posinf=1.0, neginf=0.0)
        aug_mask = np.nan_to_num(aug_mask, nan=0.0, posinf=1.0, neginf=0.0)
        aug_flows = np.nan_to_num(aug_flows, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clip to valid ranges
        aug_image = np.clip(aug_image, 0, 1)
        aug_mask = np.clip(aug_mask, 0, 1)
        aug_flows = np.clip(aug_flows, -1.0, 1.0)

        # Validate augmented shapes
        if aug_image.shape[:2] != aug_mask.shape:
            raise ValueError(
                f"Augmentation changed shapes: image {aug_image.shape[:2]} vs mask {aug_mask.shape}"
            )
        if aug_flows.shape != (2, self.config.tile_size, self.config.tile_size):
            raise ValueError(f"Augmentation changed flow shape: {aug_flows.shape}")

        # Convert to tensors
        image_tensor = self._to_tensor(aug_image)
        mask_tensor = self._to_tensor_mask(aug_mask)
        flows_tensor = torch.from_numpy(aug_flows).float()

        # Final validation
        assert image_tensor.shape[1:] == mask_tensor.shape[1:] == flows_tensor.shape[1:], \
            f"Tensor shape mismatch: img={image_tensor.shape}, mask={mask_tensor.shape}, flows={flows_tensor.shape}"

        # Final NaN check (should never happen, but belt-and-suspenders)
        assert not torch.isnan(image_tensor).any(), "NaN in image tensor!"
        assert not torch.isnan(mask_tensor).any(), "NaN in mask tensor!"
        assert not torch.isnan(flows_tensor).any(), "NaN in flows tensor!"

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "flows": flows_tensor
        }

    def _load_image(self, path: Path) -> np.ndarray:
        """
        Load and normalize microscopy images.

        FIXED: Robust percentile-based normalization.
        This is the ONLY normalization point in the pipeline.
        """
        ext = path.suffix.lower()

        # Load based on format
        if ext in [".tif", ".tiff"]:
            img = tifffile.imread(path)  # Preserves 16-bit
        elif ext == ".npy":
            img = np.load(path)
        else:
            img = np.array(Image.open(path))  # Fallback

        img = np.squeeze(img).astype(np.float32)

        # Validate
        if img.ndim != 2:
            raise ValueError(f"Expected 2D grayscale, got {img.shape}")

        # Robust percentile normalization
        vmax = np.percentile(img, 99.9)
        vmin = np.percentile(img, 0.1)

        if vmax > vmin:
            img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
        else:
            # Constant image - set to zero
            img = np.zeros_like(img)

        return img

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load segmentation mask."""
        mask = np.load(path)
        mask = np.squeeze(mask)

        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got {mask.shape}")

        return mask.astype(np.int32)

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to tensor (C, H, W)."""
        if image.ndim == 2:
            return torch.from_numpy(image).float().unsqueeze(0)
        elif image.ndim == 3:
            return torch.from_numpy(image).float().permute(2, 0, 1)
        raise ValueError(f"Unexpected image shape: {image.shape}")

    def _to_tensor_mask(self, mask: np.ndarray) -> torch.Tensor:
        """Convert mask to binary tensor (1, H, W)."""
        binary_mask = (mask > 0).astype(np.float32)
        return torch.from_numpy(binary_mask).unsqueeze(0)

    def print_augmentation_info(self):
        """Print augmentation information for debugging."""
        print(f"\n=== Dataset Augmentation Info ===")
        print(f"Split: {self.split}")

        for exp_id in sorted(self.manifest["experiment_id"].unique()):
            exp_samples = self.manifest[self.manifest["experiment_id"] == exp_id]
            current_count = len(exp_samples)

            if 'original_count' in exp_samples.columns:
                original_count = exp_samples['original_count'].iloc[0]
                was_capped = original_count > current_count
            else:
                original_count = current_count
                was_capped = False

            aug = self.augmentations[exp_id]

            # Extract augmentation mode
            if hasattr(aug, 'mode'):
                aug_mode = aug.mode
            elif hasattr(aug, 'frequency'):
                aug_mode = aug.frequency
            else:
                aug_mode = 'none' if self.split == 'val' else 'unknown'

            status = f"{original_count}->{current_count}" if was_capped else str(current_count)
            print(f"{exp_id:40s} | samples: {status:15s} | aug: {aug_mode}")


def create_dataloaders(config: 'PipelineConfig') -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from config."""
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
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        drop_last=True, persistent_workers=config.num_workers > 0
    )

    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        drop_last=False, persistent_workers=config.num_workers > 0
    )

    return train_loader, val_loader
