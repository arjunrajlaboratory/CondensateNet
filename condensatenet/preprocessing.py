"""Image preprocessing for CondensateNet."""

import numpy as np
from typing import Tuple


def normalize_image(
    image: np.ndarray,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
    clip: bool = True
) -> np.ndarray:
    """
    Normalize image using percentile-based scaling to [0, 1].
    
    This is the standard preprocessing for CondensateNet, matching
    the normalization used during training.
    
    Args:
        image: Input image as numpy array (any dtype)
        low_percentile: Lower percentile for normalization (default: 0.5)
        high_percentile: Upper percentile for normalization (default: 99.5)
        clip: Whether to clip values to [0, 1] range (default: True)
    
    Returns:
        Normalized image as float32 array with values in [0, 1]
    
    Example:
        >>> from condensatenet import normalize_image
        >>> normalized = normalize_image(raw_image)
    """
    img_float = image.astype(np.float32)
    low, high = np.percentile(img_float, [low_percentile, high_percentile])
    
    normalized = (img_float - low) / (high - low + 1e-8)
    
    if clip:
        normalized = np.clip(normalized, 0, 1)
    
    return normalized


def prepare_tensor(
    image: np.ndarray,
    device: str = 'cpu'
) -> 'torch.Tensor':
    """
    Convert normalized image to model-ready tensor.
    
    Args:
        image: Normalized image array of shape (H, W) or (H, W, 1)
        device: Device to place tensor on
    
    Returns:
        Tensor of shape (1, 1, H, W) ready for model input
    """
    import torch
    
    # Ensure 2D
    if image.ndim == 3:
        image = np.squeeze(image)
    
    # Convert to tensor: (H, W) -> (1, 1, H, W)
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    
    return tensor.to(device)
