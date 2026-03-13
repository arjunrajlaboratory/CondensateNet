# training/data/tiling.py
"""Tiling strategies for condensate segmentation dataset."""

import numpy as np
from typing import Tuple, Optional


class RandomTilingStrategy:
    """
    Random tiling strategy for training.

    Samples random positions within the image, with optional
    preference for regions containing foreground objects.
    """

    def __init__(self, tile_size: int = 320, samples_per_image: int = 1,
                 foreground_bias: float = 0.7):
        """
        Args:
            tile_size: Size of square tiles to extract
            samples_per_image: Number of tiles to sample per image
            foreground_bias: Probability of sampling near foreground (0.0-1.0)
        """
        self.tile_size = tile_size
        self.samples_per_image = samples_per_image
        self.foreground_bias = foreground_bias

    def sample_tile(self, mask: np.ndarray, cache_key: Optional[str] = None) -> Tuple[int, int]:
        """
        Sample a random tile position.

        Args:
            mask: Binary mask (H, W) indicating foreground objects
            cache_key: Optional cache key (unused for random sampling)

        Returns:
            (top, left) position of tile
        """
        h, w = mask.shape

        # Calculate valid range for top-left corner
        max_top = max(0, h - self.tile_size)
        max_left = max(0, w - self.tile_size)

        # Decide whether to bias toward foreground
        use_foreground_bias = (
            np.random.random() < self.foreground_bias and
            mask.max() > 0
        )

        if use_foreground_bias:
            # Find foreground pixels
            fg_coords = np.argwhere(mask > 0)

            if len(fg_coords) > 0:
                # Sample a random foreground pixel
                idx = np.random.randint(len(fg_coords))
                cy, cx = fg_coords[idx]

                # Center tile around this pixel (with some jitter)
                jitter_y = np.random.randint(-self.tile_size // 4, self.tile_size // 4)
                jitter_x = np.random.randint(-self.tile_size // 4, self.tile_size // 4)

                top = cy - self.tile_size // 2 + jitter_y
                left = cx - self.tile_size // 2 + jitter_x

                # Clip to valid range
                top = np.clip(top, 0, max_top)
                left = np.clip(left, 0, max_left)

                return (int(top), int(left))

        # Fallback: completely random position
        top = np.random.randint(0, max_top + 1) if max_top > 0 else 0
        left = np.random.randint(0, max_left + 1) if max_left > 0 else 0

        return (int(top), int(left))


class ValidationTilingStrategy:
    """
    Deterministic tiling strategy for validation.

    FIXED: Now avoids empty center crops by finding foreground
    center of mass when center is empty.
    """

    def __init__(self, tile_size: int = 320, min_foreground_pixels: int = 10):
        """
        Args:
            tile_size: Size of square tiles to extract
            min_foreground_pixels: Minimum foreground pixels to consider non-empty
        """
        self.tile_size = tile_size
        self.min_foreground_pixels = min_foreground_pixels
        self._cache = {}

    def sample_tile(self, mask: np.ndarray, cache_key: Optional[str] = None) -> Tuple[int, int]:
        """
        Sample a deterministic tile position.

        FIXED: Tries center first, but if empty, finds foreground center of mass.

        Args:
            mask: Binary mask (H, W) indicating foreground objects
            cache_key: Optional cache key for deterministic sampling

        Returns:
            (top, left) position of tile
        """
        # Use cache for deterministic behavior
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]

        h, w = mask.shape

        # Calculate valid range for top-left corner
        max_top = max(0, h - self.tile_size)
        max_left = max(0, w - self.tile_size)

        # Default: center crop
        top = max_top // 2
        left = max_left // 2

        # Check if center crop has sufficient foreground
        center_tile = mask[top:top+self.tile_size, left:left+self.tile_size]
        center_fg_count = (center_tile > 0).sum()

        # If center is mostly empty AND mask has foreground elsewhere
        if center_fg_count < self.min_foreground_pixels and mask.max() > 0:
            # Find center of mass of all foreground pixels
            fg_coords = np.argwhere(mask > 0)

            if len(fg_coords) > 0:
                # Calculate center of mass
                cy, cx = fg_coords.mean(axis=0).astype(int)

                # Center tile on foreground center of mass
                top = cy - self.tile_size // 2
                left = cx - self.tile_size // 2

                # Clip to valid range
                top = np.clip(top, 0, max_top)
                left = np.clip(left, 0, max_left)

        # Cache the position for deterministic behavior
        if cache_key is not None:
            self._cache[cache_key] = (int(top), int(left))

        return (int(top), int(left))

    def clear_cache(self):
        """Clear the position cache."""
        self._cache.clear()


def extract_tile(image: np.ndarray, mask: np.ndarray,
                 top: int, left: int, tile_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a tile from image and mask at specified position.

    Args:
        image: Input image (H, W) or (H, W, C)
        mask: Binary mask (H, W)
        top: Top coordinate of tile
        left: Left coordinate of tile
        tile_size: Size of square tile

    Returns:
        (image_tile, mask_tile)
    """
    image_tile = image[top:top+tile_size, left:left+tile_size]
    mask_tile = mask[top:top+tile_size, left:left+tile_size]

    return image_tile.copy(), mask_tile.copy()


def extract_flow_tile(flows: np.ndarray, top: int, left: int,
                      tile_size: int) -> np.ndarray:
    """
    Extract a tile from flow field at specified position.

    Args:
        flows: Flow field (2, H, W)
        top: Top coordinate of tile
        left: Left coordinate of tile
        tile_size: Size of square tile

    Returns:
        flows_tile (2, tile_size, tile_size)
    """
    flows_tile = flows[:, top:top+tile_size, left:left+tile_size]
    return flows_tile.copy()
