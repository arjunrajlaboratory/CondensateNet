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

class AugmentationPresets:
    """Predefined augmentation presets."""

    @staticmethod
    def get_light_preset() -> Dict:
        return {
            'horizontal_flip_prob': 0.3,
            'vertical_flip_prob': 0.3,
            'rotation_limit': 3,
            'rotation_prob': 0.2,
            'brightness_limit': 0.05,
            'contrast_limit': 0.1,
            'brightness_contrast_prob': 0.3,
            'gamma_limit': (95, 105),
            'gamma_prob': 0.2,
        }

    @staticmethod
    def get_medium_preset() -> Dict:
        return {
            'horizontal_flip_prob': 0.5,
            'vertical_flip_prob': 0.5,
            'rotation_limit': 5,
            'rotation_prob': 0.3,
            'brightness_limit': 0.08,
            'contrast_limit': 0.2,
            'brightness_contrast_prob': 0.5,
            'gamma_limit': (90, 110),
            'gamma_prob': 0.3,
            'elastic_alpha': 50,
            'elastic_sigma': 5,
            'elastic_prob': 0.2,
        }

    @staticmethod
    def get_heavy_preset() -> Dict:
        return {
            'horizontal_flip_prob': 0.7,
            'vertical_flip_prob': 0.7,
            'rotation_limit': 10,
            'rotation_prob': 0.5,
            'brightness_limit': 0.12,
            'contrast_limit': 0.3,
            'brightness_contrast_prob': 0.7,
            'gamma_limit': (80, 120),
            'gamma_prob': 0.5,
            'elastic_alpha': 100,
            'elastic_sigma': 10,
            'elastic_prob': 0.4,
            'gaussian_noise_prob': 0.3,
            'gaussian_blur_prob': 0.2,
        }

class AlbumentationsAugmentation:
    """
    Augmentation using albumentations library.

    FIXED: Now properly separates geometric and pixel transforms.
    Flows only receive geometric transforms.
    """

    def __init__(self, mode: str = 'medium', custom_params: Optional[Dict] = None):
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations not available")

        self.mode = mode
        self.params = self._get_preset(mode)
        if custom_params:
            self.params.update(custom_params)

        self.transform = self._build_transform()

    def _get_preset(self, mode: str) -> Dict:
        if mode == 'light':
            return AugmentationPresets.get_light_preset()
        elif mode == 'medium':
            return AugmentationPresets.get_medium_preset()
        elif mode == 'heavy':
            return AugmentationPresets.get_heavy_preset()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _build_transform(self):
        """Build transform pipeline with proper flow handling."""
        # Geometric transforms (apply to image, mask, AND flows)
        geometric_transforms = []

        if self.params.get('horizontal_flip_prob', 0) > 0:
            geometric_transforms.append(A.HorizontalFlip(p=self.params['horizontal_flip_prob']))

        if self.params.get('vertical_flip_prob', 0) > 0:
            geometric_transforms.append(A.VerticalFlip(p=self.params['vertical_flip_prob']))

        if self.params.get('rotation_prob', 0) > 0:
            geometric_transforms.append(A.Rotate(
                limit=self.params['rotation_limit'],
                interpolation=1,
                border_mode=0,
                p=self.params['rotation_prob']
            ))

        if self.params.get('elastic_prob', 0) > 0:
            geometric_transforms.append(A.ElasticTransform(
                alpha=self.params.get('elastic_alpha', 50),
                sigma=self.params.get('elastic_sigma', 5),
                interpolation=1,
                border_mode=0,
                p=self.params['elastic_prob']
            ))

        # Pixel-level transforms (apply ONLY to image)
        # These will be filtered out for flows in __call__
        if self.params.get('brightness_contrast_prob', 0) > 0:
            geometric_transforms.append(A.RandomBrightnessContrast(
                brightness_limit=self.params['brightness_limit'],
                contrast_limit=self.params['contrast_limit'],
                brightness_by_max=True,
                p=self.params['brightness_contrast_prob']
            ))

        if self.params.get('gamma_prob', 0) > 0:
            geometric_transforms.append(A.RandomGamma(
                gamma_limit=self.params['gamma_limit'],
                p=self.params['gamma_prob']
            ))

        if self.params.get('gaussian_noise_prob', 0) > 0:
            geometric_transforms.append(A.GaussNoise(p=self.params['gaussian_noise_prob']))

        if self.params.get('gaussian_blur_prob', 0) > 0:
            geometric_transforms.append(A.GaussianBlur(
                blur_limit=(3, 5),
                p=self.params['gaussian_blur_prob']
            ))

        # Return composed transforms with flow support
        return A.Compose(
            geometric_transforms,
            additional_targets={'mask': 'mask', 'flows': 'image'}
        )

    def __call__(self, image: np.ndarray, mask: np.ndarray,
                 flows: Optional[np.ndarray] = None) -> Union[Tuple[np.ndarray, np.ndarray],
                                                               Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Apply augmentation to image, mask, and optionally flows.

        FIXED: Clips flows to [-1, 1] after augmentation to prevent drift.
        """
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        try:
            if flows is not None:
                flows = np.ascontiguousarray(flows)
                flows_hwc = np.transpose(flows, (1, 2, 0))  # (2, H, W) -> (H, W, 2)

                augmented = self.transform(image=image, mask=mask, flows=flows_hwc)

                # Convert flows back to (2, H, W)
                aug_flows = np.transpose(augmented['flows'], (2, 0, 1))

                # CRITICAL: Clip flows to valid range [-1, 1]
                aug_flows = np.clip(aug_flows, -1.0, 1.0)

                return augmented['image'], augmented['mask'], aug_flows
            else:
                augmented = self.transform(image=image, mask=mask)
                return augmented['image'], augmented['mask']

        except Exception as e:
            warnings.warn(f"Augmentation failed: {e}")
            if flows is not None:
                return image, mask, flows
            return image, mask

class SimpleAugmentation:
    """
    Simple numpy-based augmentation fallback.

    FIXED: Now supports flow field transformation.
    """

    def __init__(self, mode: str = 'medium', custom_params: Optional[Dict] = None):
        self.mode = mode
        self.params = self._get_preset(mode)
        if custom_params:
            self.params.update(custom_params)

    def _get_preset(self, mode: str) -> Dict:
        if mode == 'light':
            return AugmentationPresets.get_light_preset()
        elif mode == 'medium':
            return AugmentationPresets.get_medium_preset()
        elif mode == 'heavy':
            return AugmentationPresets.get_heavy_preset()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __call__(self, image: np.ndarray, mask: np.ndarray,
                 flows: Optional[np.ndarray] = None) -> Union[Tuple[np.ndarray, np.ndarray],
                                                               Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Apply simple augmentation to image, mask, and optionally flows.

        FIXED: Now supports flow field transformation.
        """
        aug_image = image.copy()
        aug_mask = mask.copy()
        aug_flows = flows.copy() if flows is not None else None

        # Horizontal flip
        if np.random.random() < self.params.get('horizontal_flip_prob', 0):
            aug_image = np.fliplr(aug_image)
            aug_mask = np.fliplr(aug_mask)
            if aug_flows is not None:
                aug_flows = np.fliplr(aug_flows)
                aug_flows[1] = -aug_flows[1]  # Flip x-component

        # Vertical flip
        if np.random.random() < self.params.get('vertical_flip_prob', 0):
            aug_image = np.flipud(aug_image)
            aug_mask = np.flipud(aug_mask)
            if aug_flows is not None:
                aug_flows = np.flipud(aug_flows)
                aug_flows[0] = -aug_flows[0]  # Flip y-component

        # 90-degree rotation
        if np.random.random() < self.params.get('rotation_prob', 0):
            k = np.random.choice([1, 2, 3])
            aug_image = np.rot90(aug_image, k=k)
            aug_mask = np.rot90(aug_mask, k=k)
            if aug_flows is not None:
                aug_flows = np.rot90(aug_flows, k=k, axes=(1, 2))
                # Rotate flow vectors: 90° rotation matrix applied k times
                for _ in range(k):
                    dy, dx = aug_flows[0].copy(), aug_flows[1].copy()
                    aug_flows[0] = -dx  # New dy = -old dx
                    aug_flows[1] = dy   # New dx = old dy

        # Brightness adjustment (image only)
        if np.random.random() < self.params.get('brightness_contrast_prob', 0):
            brightness_factor = 1.0 + np.random.uniform(
                -self.params['brightness_limit'],
                self.params['brightness_limit']
            )
            aug_image = np.clip(aug_image * brightness_factor, 0, aug_image.max())

        if flows is not None:
            return aug_image, aug_mask, aug_flows
        return aug_image, aug_mask

class ExperimentAwareAugmentation:
    """Experiment-aware augmentation selector."""

    def __init__(self, experiment_frequency: str = 'medium',
                 use_albumentations: bool = True,
                 custom_params: Optional[Dict] = None):
        frequency_to_mode = {'rare': 'heavy', 'moderate': 'medium', 'common': 'light'}

        if experiment_frequency not in frequency_to_mode:
            raise ValueError(f"Unknown frequency: {experiment_frequency}")

        mode = frequency_to_mode[experiment_frequency]
        self.frequency = experiment_frequency
        self.mode = mode

        if use_albumentations and ALBUMENTATIONS_AVAILABLE:
            self.augmentation = AlbumentationsAugmentation(mode, custom_params)
        else:
            if use_albumentations:
                warnings.warn("albumentations not available, using simple augmentation")
            self.augmentation = SimpleAugmentation(mode, custom_params)

    def __call__(self, image: np.ndarray, mask: np.ndarray,
                 flows: Optional[np.ndarray] = None) -> Union[Tuple[np.ndarray, np.ndarray],
                                                               Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Apply augmentation with optional flow transformation."""
        return self.augmentation(image, mask, flows)

class AugmentationFactory:
    """Factory for creating augmentation pipelines."""

    @staticmethod
    def create_from_frequency(frequency: str, **kwargs) -> ExperimentAwareAugmentation:
        return ExperimentAwareAugmentation(frequency, **kwargs)

    @staticmethod
    def create_from_count(sample_count: int,
                         original_count: Optional[int] = None,
                         **kwargs) -> ExperimentAwareAugmentation:
        """
        Create augmentation based on sample count.

        FIXED: Now uses original_count (before capping) if provided.
        """
        count_for_decision = original_count if original_count is not None else sample_count

        if count_for_decision < 5:
            frequency = 'rare'
        elif count_for_decision <= 20:
            frequency = 'moderate'
        else:
            frequency = 'common'

        return ExperimentAwareAugmentation(frequency, **kwargs)

    @staticmethod
    def create_no_augmentation():
        """Create passthrough augmentation for validation."""
        def no_augmentation(image, mask, flows=None):
            if flows is not None:
                return image, mask, flows
            return image, mask
        return no_augmentation
