"""Utility functions for CondensateNet."""

import torch
from typing import Optional


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the torch device to use.
    
    Args:
        device: Explicit device string ('cuda', 'cpu', 'mps'). 
                If None, auto-detects best available device.
    
    Returns:
        torch.device object
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_cache_dir() -> str:
    """Get the default cache directory for CondensateNet models."""
    import os
    
    # Use HuggingFace cache convention
    cache_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    return os.path.join(cache_home, 'condensatenet')
