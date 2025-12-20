"""Model loading and downloading for CondensateNet."""

import os
from typing import Optional, Union
from pathlib import Path

# Default HuggingFace repository
DEFAULT_REPO_ID = "rajlab/condensatenet"


def download_model(
    output_dir: Optional[str] = None,
    repo_id: str = DEFAULT_REPO_ID,
    force: bool = False
) -> str:
    """
    Download CondensateNet model from HuggingFace to a local directory.
    
    This is useful for Docker builds where you want to bake the model
    into the image rather than downloading at runtime.
    
    Args:
        output_dir: Directory to save model files. If None, uses default cache.
        repo_id: HuggingFace repository ID (default: rajlab/condensatenet)
        force: If True, re-download even if files exist (default: False)
    
    Returns:
        Path to the directory containing model files
    
    Example:
        >>> # Download to default cache
        >>> path = download_model()
        
        >>> # Download to specific directory (for Docker)
        >>> path = download_model(output_dir="/models/condensatenet")
    
    CLI usage:
        python -m condensatenet download --output /models/condensatenet
    """
    from huggingface_hub import snapshot_download
    from .utils import get_cache_dir
    
    if output_dir is None:
        output_dir = get_cache_dir()
    
    output_dir = os.path.abspath(output_dir)
    
    # Check if already downloaded
    required_files = ['config.json', 'model.safetensors']
    if not force and all(os.path.exists(os.path.join(output_dir, f)) for f in required_files):
        print(f"Model already exists at {output_dir}")
        return output_dir
    
    print(f"Downloading CondensateNet from {repo_id}...")
    print(f"Destination: {output_dir}")
    
    # Download all files from the repo
    snapshot_download(
        repo_id=repo_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False  # Copy files, don't symlink
    )
    
    print(f"✅ Model downloaded to {output_dir}")
    return output_dir


def load_model(
    source: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs
):
    """
    Load CondensateNet model.
    
    Args:
        source: Either:
            - None: Load from HuggingFace (rajlab/condensatenet)
            - HuggingFace repo ID: e.g., "rajlab/condensatenet"
            - Local path: e.g., "/models/condensatenet"
        device: Device to load model on ('cuda', 'cpu', 'mps'). 
                Auto-detects if None.
        **kwargs: Additional arguments passed to from_pretrained()
    
    Returns:
        Loaded CondensateNet model in eval mode
    
    Example:
        >>> # From HuggingFace
        >>> model = load_model()
        
        >>> # From local directory
        >>> model = load_model("/models/condensatenet")
        
        >>> # Explicit device
        >>> model = load_model(device="cuda")
    """
    from transformers import AutoModel
    from .utils import get_device
    
    device = get_device(device)
    
    # Determine source
    if source is None:
        source = DEFAULT_REPO_ID
    
    # Check if it's a local path
    is_local = os.path.isdir(source) if source != DEFAULT_REPO_ID else False
    
    if is_local:
        print(f"Loading CondensateNet from local: {source}")
    else:
        print(f"Loading CondensateNet from HuggingFace: {source}")
    
    # Load model
    model = AutoModel.from_pretrained(
        source,
        trust_remote_code=True,
        **kwargs
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    return model


def get_model_info(source: Optional[str] = None) -> dict:
    """
    Get information about the CondensateNet model.
    
    Args:
        source: Model source (HuggingFace repo or local path)
    
    Returns:
        Dict with model information
    """
    import json
    from huggingface_hub import hf_hub_download
    
    if source is None:
        source = DEFAULT_REPO_ID
    
    # Load config
    if os.path.isdir(source):
        config_path = os.path.join(source, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config_path = hf_hub_download(repo_id=source, filename='config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    return {
        'repo_id': source,
        'model_type': config.get('model_type', 'unknown'),
        'encoder_variant': config.get('encoder_variant', 'unknown'),
        'pyramid_channels': config.get('pyramid_channels', []),
        'use_spatial_attention': config.get('use_spatial_attention', False),
        'dropout_rate': config.get('dropout_rate', 0.0)
    }
