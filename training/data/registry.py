"""Data registry for condensate segmentation pipeline."""

import re
import numpy as np
import pandas as pd
import tifffile
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple


class RegistrySample:
    """Container for a single image-mask pair with validation."""

    def __init__(self, image_path: Path, mask_path: Path, experiment_id: str):
        self.image_path = image_path
        self.mask_path = mask_path
        self.experiment_id = experiment_id
        self.is_valid = False
        self.validation_message = ""
        self.image_shape = None
        self.mask_shape = None

class DataRegistry:
    """
    Registry for discovering and validating image-mask pairs.

    FIXED: Now validates minimum image size to prevent tile extraction issues.
    """

    def __init__(self, images_dir: Path, masks_dir: Path,
                 min_image_size: int = 256,
                 tile_size: int = 320):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.min_image_size = min_image_size
        self.tile_size = tile_size

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    def scan_and_validate(self, quick_check: bool = False,
                         sample_size: int = 5) -> pd.DataFrame:
        """
        Scan directories and validate image-mask pairs.

        FIXED: Now handles empty results gracefully and validates image sizes.

        Args:
            quick_check: If True, only validate a sample of pairs
            sample_size: Number of samples to check if quick_check=True

        Returns:
            DataFrame with validated pairs
        """
        print("Scanning for image-mask pairs...")

        samples = self._discover_pairs()
        print(f"Found {len(samples)} potential pairs")

        # FIXED: Handle case where no pairs are found
        if len(samples) == 0:
            print("\n⚠️  No image-mask pairs found!")
            print("\nTroubleshooting:")
            print(f"  1. Check images directory: {self.images_dir}")
            print(f"     - Looking for .tif files")
            print(f"     - Found subdirectories: {list(self.images_dir.glob('*')) if self.images_dir.exists() else 'None'}")
            print(f"  2. Check masks directory: {self.masks_dir}")
            print(f"     - Looking for *_mask.npy files")
            print(f"     - Expected structure: masks_dir/experiment_id/filename_mask.npy")

            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                'image_path', 'mask_path', 'experiment_id', 'is_valid',
                'validation_message', 'image_height', 'image_width',
                'mask_height', 'mask_width'
            ])

        if quick_check and len(samples) > sample_size:
            import random
            samples = random.sample(samples, sample_size)
            print(f"Quick check: validating {len(samples)} samples")

        # Validate each sample
        validated_samples = []
        for sample in samples:
            self._validate_sample(sample, full_validation=not quick_check)
            validated_samples.append(sample)

        # Convert to DataFrame
        manifest_df = self._to_dataframe(validated_samples)

        # Print summary
        n_valid = manifest_df['is_valid'].sum()
        n_invalid = len(manifest_df) - n_valid

        print(f"\nValidation complete:")
        print(f"  Valid: {n_valid}")
        print(f"  Invalid: {n_invalid}")

        if n_invalid > 0:
            print("\nInvalid samples by reason:")
            for reason, count in manifest_df[~manifest_df['is_valid']]['validation_message'].value_counts().items():
                print(f"  {reason}: {count}")

        # Print size statistics for valid samples
        if n_valid > 0:
            valid_df = manifest_df[manifest_df['is_valid']]
            heights = valid_df['image_height'].values
            widths = valid_df['image_width'].values

            print(f"\nValid image sizes:")
            print(f"  Height: min={heights.min()}, max={heights.max()}, mean={heights.mean():.1f}")
            print(f"  Width: min={widths.min()}, max={widths.max()}, mean={widths.mean():.1f}")

            # Warn about images that will need padding
            needs_padding = ((heights < self.tile_size) | (widths < self.tile_size)).sum()
            if needs_padding > 0:
                print(f"\n⚠️  Warning: {needs_padding} images smaller than tile_size ({self.tile_size}x{self.tile_size})")
                print(f"   These will be padded during tile extraction.")

        return manifest_df



    def _discover_pairs(self) -> list:
        """
        Discover all image-mask pairs.

        FIXED: Extract experiment ID from filename pattern instead of directory.
        """
        samples = []

        # Find all images
        image_patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
        all_images = []
        for pattern in image_patterns:
            all_images.extend(self.images_dir.rglob(pattern))

        if len(all_images) == 0:
            print(f"  ⚠️  No .tif images found in {self.images_dir}")
            return samples

        print(f"  Found {len(all_images)} .tif images")

        # For each image, extract experiment ID and find mask
        masks_found = 0
        for img_path in all_images:
            # ============================================================
            # EXTRACT EXPERIMENT ID FROM FILENAME
            # ============================================================
            filename = img_path.stem  # Without extension

            # Parse filename pattern: DATE_TIME_-_EXPERIMENT_...rest...
            # Example: "2025-03-28_11:08_-_15minrepeat001_xy0_t6_ch0_z4"
            parts = filename.split('_-_')

            if len(parts) >= 2:
                # Everything after the date/time prefix and before first underscore
                exp_part = parts[1]  # "15minrepeat001_xy0_t6_ch0_z4"

                # Extract experiment name (before position markers)
                # Split on common position markers: xy, t, ch, z
                match = re.match(r'^([^_]+?)(?:_xy|_t\d|_ch|_z|_Point|_C\d)', exp_part)
                if match:
                    exp_id = match.group(1)
                else:
                    # Fallback: take first part before underscore
                    exp_id = exp_part.split('_')[0]
            else:
                # Fallback for non-standard naming
                # Try to extract something meaningful from first few parts
                parts = filename.split('_')
                if len(parts) >= 3:
                    exp_id = parts[2]  # Usually experiment name is 3rd part
                else:
                    exp_id = 'unknown'

            # ============================================================
            # FIND CORRESPONDING MASK
            # ============================================================
            possible_mask_paths = [
                self.masks_dir / exp_id / f"{img_path.stem}_mask.npy",
                self.masks_dir / exp_id / f"{img_path.stem}.npy",
                self.masks_dir / f"{img_path.stem}_mask.npy",
                self.masks_dir / f"{img_path.stem}.npy",
            ]

            mask_path = None
            for possible_path in possible_mask_paths:
                if possible_path.exists():
                    mask_path = possible_path
                    break

            if mask_path is not None:
                samples.append(RegistrySample(img_path, mask_path, exp_id))
                masks_found += 1

        if masks_found == 0:
            print(f"  ⚠️  No matching masks found in {self.masks_dir}")
        else:
            print(f"  Matched {masks_found} image-mask pairs")

        return samples



    def _validate_sample(self, sample: RegistrySample, full_validation: bool = True):
        """
        Validate a single sample.

        FIXED: Now checks minimum image size requirements.
        """
        try:
            # Load image and mask
            img = tifffile.imread(sample.image_path)
            mask = np.load(sample.mask_path)

            # Store shapes
            sample.image_shape = img.shape
            sample.mask_shape = mask.shape

            # Basic shape validation
            if img.ndim not in [2, 3]:
                sample.validation_message = f"Invalid image dimensions: {img.ndim}D"
                return

            if img.ndim == 3:
                # Handle multi-channel images
                if img.shape[0] <= 4:  # Assume channels-first
                    img = img[0]
                else:  # Assume channels-last
                    img = img[:, :, 0]

            if mask.ndim != 2:
                sample.validation_message = f"Invalid mask dimensions: {mask.ndim}D"
                return

            # Check minimum size requirements (must be at least tile_size)
            h, w = img.shape[:2]
            min_required = max(self.min_image_size, self.tile_size)
            if h < min_required or w < min_required:
                sample.validation_message = (
                    f"Image too small: {h}x{w} < minimum {min_required}x{min_required}"
                )
                return

            # Shape matching
            if img.shape[:2] != mask.shape:
                sample.validation_message = f"Shape mismatch: img {img.shape[:2]} vs mask {mask.shape}"
                return

            # Content validation (if full validation)
            if full_validation:
                # Check if mask has any foreground
                if mask.max() == 0:
                    sample.validation_message = "Empty mask (no foreground)"
                    return

                # Check image intensity range
                if img.max() == img.min():
                    sample.validation_message = "Constant intensity image"
                    return

            # All checks passed
            sample.is_valid = True
            sample.validation_message = "OK"

        except Exception as e:
            sample.validation_message = f"Load error: {str(e)}"

    def _to_dataframe(self, samples: list) -> pd.DataFrame:
        """Convert samples to DataFrame with guaranteed columns."""
        data = []
        for sample in samples:
            row = {
                'image_path': str(sample.image_path),
                'mask_path': str(sample.mask_path),
                'experiment_id': sample.experiment_id,
                'is_valid': sample.is_valid,
                'validation_message': sample.validation_message,
                'image_height': None,
                'image_width': None,
                'mask_height': None,
                'mask_width': None,
            }

            # Add shape information if available
            if sample.image_shape is not None:
                if len(sample.image_shape) == 2:
                    row['image_height'] = sample.image_shape[0]
                    row['image_width'] = sample.image_shape[1]
                elif len(sample.image_shape) == 3:
                    # Handle different channel layouts
                    if sample.image_shape[0] <= 4:  # Channels-first
                        row['image_height'] = sample.image_shape[1]
                        row['image_width'] = sample.image_shape[2]
                    else:  # Channels-last
                        row['image_height'] = sample.image_shape[0]
                        row['image_width'] = sample.image_shape[1]

            if sample.mask_shape is not None:
                row['mask_height'] = sample.mask_shape[0]
                row['mask_width'] = sample.mask_shape[1]

            data.append(row)

        df = pd.DataFrame(data)

        # Ensure all expected columns exist even if empty
        expected_columns = [
            'image_path', 'mask_path', 'experiment_id', 'is_valid',
            'validation_message', 'image_height', 'image_width',
            'mask_height', 'mask_width'
        ]

        for col in expected_columns:
            if col not in df.columns:
                df[col] = None

        return df[expected_columns]


class ExperimentAnalyzer:
    def __init__(self, registry):
        if hasattr(registry, 'manifest_df'):
            manifest_df = registry.manifest_df
        else:
            manifest_df = registry

        self.valid_df = manifest_df[manifest_df['is_valid']].copy()
        self.distribution = self.valid_df['experiment_id'].value_counts()

    def analyze(self, target_max_pct=10.0):
        total = len(self.valid_df)
        target_max = int((target_max_pct / 100) * total)

        print(f"Total: {total} samples, {len(self.distribution)} experiments")
        print(f"Range: {self.distribution.min()}-{self.distribution.max()} samples/exp")
        print(f"Imbalance: {self.distribution.max() / self.distribution.min():.1f}x")

        # Dominant experiments
        dominant = [(exp, cnt) for exp, cnt in self.distribution.items()
                   if (cnt/total)*100 > 15]
        if dominant:
            print(f"\nDominant (>15%):")
            for exp, cnt in dominant:
                print(f"  {exp[:40]}: {cnt} ({cnt/total*100:.1f}%)")

        # Recommended caps
        caps = {exp: max(target_max, 10) for exp, cnt in self.distribution.items()
                if cnt > 10 and cnt > target_max}
        if caps:
            print(f"\nRecommended caps:")
            for exp, cap in sorted(caps.items(), key=lambda x: self.distribution[x[0]], reverse=True):
                print(f"  {exp[:40]}: {self.distribution[exp]} → {cap}")

        # Augmentation strategy
        heavy = sum(1 for c in self.distribution if c < 5)
        medium = sum(1 for c in self.distribution if 5 <= c <= 20)
        light = sum(1 for c in self.distribution if c > 20)
        print(f"\nAugmentation: {heavy} heavy, {medium} medium, {light} light")

        return caps


class StratifiedDataSplitter:
    """Create stratified train/val splits with experiment balancing."""

    def __init__(self,
                 manifest_df: pd.DataFrame,
                 experiment_caps: Optional[Dict[str, int]] = None,
                 val_ratio: float = 0.15,
                 random_seed: int = 42,
                 min_val_samples_per_exp: int = 1):
        self.manifest_df = manifest_df[manifest_df['is_valid']].copy()
        self.experiment_caps = experiment_caps or {}
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.min_val_samples = min_val_samples_per_exp

        np.random.seed(random_seed)

        self.train_df = None
        self.val_df = None
        # ADDED: Track original counts before capping
        self.experiment_original_counts = {}

    def create_splits(self, apply_caps: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create stratified train/val splits."""
        working_df = self._apply_caps() if apply_caps else self.manifest_df.copy()

        train_dfs = []
        val_dfs = []

        for exp_id in working_df['experiment_id'].unique():
            exp_df = working_df[working_df['experiment_id'] == exp_id]
            train_exp, val_exp = self._split_single_experiment(exp_df, exp_id)

            if len(train_exp) > 0:
                train_dfs.append(train_exp)
            if len(val_exp) > 0:
                val_dfs.append(val_exp)

        self.train_df = pd.concat(train_dfs, ignore_index=True)
        self.val_df = pd.concat(val_dfs, ignore_index=True)

        # ADDED: Add original_count column to both splits
        self.train_df['original_count'] = self.train_df['experiment_id'].map(self.experiment_original_counts)
        self.val_df['original_count'] = self.val_df['experiment_id'].map(self.experiment_original_counts)

        return self.train_df, self.val_df

    def _apply_caps(self) -> pd.DataFrame:
        """
        Apply experiment caps to dataset, with optional spacing
        to avoid selecting consecutive z-slices or time frames.

        FIXED: Now tracks original counts before capping for proper augmentation decisions.
        """
        if not self.experiment_caps:
            # ADDED: Even without caps, track original counts
            for exp_id in self.manifest_df["experiment_id"].unique():
                exp_df = self.manifest_df[self.manifest_df["experiment_id"] == exp_id]
                self.experiment_original_counts[exp_id] = len(exp_df)
            return self.manifest_df.copy()

        capped_dfs = []
        for exp_id in self.manifest_df["experiment_id"].unique():
            exp_df = self.manifest_df[self.manifest_df["experiment_id"] == exp_id]

            # ADDED: Store original count BEFORE capping
            original_count = len(exp_df)
            self.experiment_original_counts[exp_id] = original_count

            # If this experiment has a defined cap
            if exp_id in self.experiment_caps:
                cap = self.experiment_caps[exp_id]

                if len(exp_df) > cap:
                    # Sort by filename so frames are in logical order (z/t)
                    exp_df = exp_df.sort_values("image_path")

                    # Compute step size to spread selections evenly
                    step = max(1, len(exp_df) // cap)

                    # Take every "step"-th frame to avoid redundancy
                    exp_df = exp_df.iloc[::step][:cap]

                    # Optional fallback (use random sample if spacing fails)
                    if len(exp_df) < cap:
                        needed = cap - len(exp_df)
                        extra = self.manifest_df[self.manifest_df["experiment_id"] == exp_id]
                        extra = extra.sample(n=needed, random_state=self.random_seed)
                        exp_df = pd.concat([exp_df, extra]).head(cap)

            capped_dfs.append(exp_df)

        capped_df = pd.concat(capped_dfs, ignore_index=True)

        return capped_df

    def _split_single_experiment(self, exp_df: pd.DataFrame, exp_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split a single experiment into train/val."""
        n_samples = len(exp_df)
        n_val = max(self.min_val_samples, int(n_samples * self.val_ratio))
        n_val = min(n_val, n_samples - 1)

        if n_val == 0:
            return exp_df, pd.DataFrame()

        exp_df_shuffled = exp_df.sample(frac=1, random_state=self.random_seed)
        val_df = exp_df_shuffled.iloc[:n_val]
        train_df = exp_df_shuffled.iloc[n_val:]

        return train_df, val_df

    def save_splits(self, output_dir: Path, prefix: str = ""):
        """Save train/val splits to CSV files."""
        if self.train_df is None or self.val_df is None:
            raise ValueError("Splits not created yet")

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        train_path = output_dir / f"{prefix}train_manifest.csv"
        val_path = output_dir / f"{prefix}val_manifest.csv"

        # ADDED: Original counts are now included in saved CSVs
        self.train_df.to_csv(train_path, index=False)
        self.val_df.to_csv(val_path, index=False)

        return train_path, val_path

    def print_augmentation_summary(self):
        """Print summary of augmentation assignments based on original counts."""
        print("\n=== Augmentation Strategy Summary ===")
        for exp_id in sorted(self.experiment_original_counts.keys()):
            original_count = self.experiment_original_counts[exp_id]

            if exp_id in self.experiment_caps:
                capped_count = self.experiment_caps[exp_id]
                was_capped = original_count > capped_count
            else:
                capped_count = original_count
                was_capped = False

            # Determine augmentation based on ORIGINAL count
            if original_count < 5:
                aug_mode = "heavy"
            elif original_count <= 20:
                aug_mode = "medium"
            else:
                aug_mode = "light"

            status = f"capped {original_count}→{capped_count}" if was_capped else f"kept {original_count}"
            print(f"{exp_id:40s} | {status:20s} | augmentation: {aug_mode}")

def quick_split(manifest_df: pd.DataFrame,
                experiment_caps: Optional[Dict[str, int]] = None,
                val_ratio: float = 0.15,
                output_dir: Optional[Path] = None,
                random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Quick train/val split."""
    splitter = StratifiedDataSplitter(
        manifest_df,
        experiment_caps=experiment_caps,
        val_ratio=val_ratio,
        random_seed=random_seed
    )

    train_df, val_df = splitter.create_splits(apply_caps=True)

    # ADDED: Print augmentation summary
    splitter.print_augmentation_summary()

    if output_dir:
        splitter.save_splits(output_dir)

    return train_df, val_df
