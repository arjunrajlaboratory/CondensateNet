"""
CondensateNet: Deep learning for biomolecular condensate segmentation.

A package for detecting and segmenting biomolecular condensates in
microscopy images using the CondensateNet deep learning model.

Quick Start:
    >>> from condensatenet import CondensateNetPipeline
    >>> 
    >>> # Load the model
    >>> pipeline = CondensateNetPipeline.from_pretrained()
    >>> 
    >>> # Segment an image
    >>> instances = pipeline.segment(image)
    >>> print(f"Found {instances.max()} condensates")

For Docker deployments:
    # Download model at build time
    $ python -m condensatenet download --output /models/condensatenet
    
    # Load from local path at runtime
    >>> pipeline = CondensateNetPipeline.from_local("/models/condensatenet")
"""

__version__ = "0.1.0"

from .pipeline import CondensateNetPipeline
from .preprocessing import normalize_image
from .postprocessing import flow_to_instances, extract_instance_properties
from .model import download_model, load_model, get_model_info
from .utils import get_device, get_cache_dir

__all__ = [
    # Main pipeline
    "CondensateNetPipeline",
    
    # Preprocessing
    "normalize_image",
    
    # Postprocessing
    "flow_to_instances",
    "extract_instance_properties",
    
    # Model loading
    "download_model",
    "load_model",
    "get_model_info",
    
    # Utilities
    "get_device",
    "get_cache_dir",
]
