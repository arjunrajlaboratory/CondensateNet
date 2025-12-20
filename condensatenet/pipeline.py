"""High-level pipeline for CondensateNet inference."""

import numpy as np
import torch
from typing import Optional, Dict, Any, Union, Tuple

from .preprocessing import normalize_image, prepare_tensor
from .postprocessing import flow_to_instances, extract_instance_properties
from .model import load_model, download_model, DEFAULT_REPO_ID
from .utils import get_device


class CondensateNetPipeline:
    """
    High-level pipeline for biomolecular condensate segmentation.
    
    This class provides a simple interface for:
    - Loading the CondensateNet model
    - Preprocessing images
    - Running inference
    - Postprocessing to get instance segmentations
    
    Example:
        >>> from condensatenet import CondensateNetPipeline
        >>> 
        >>> # Load from HuggingFace
        >>> pipeline = CondensateNetPipeline.from_pretrained()
        >>> 
        >>> # Segment an image
        >>> instances = pipeline.segment(image)
        >>> print(f"Found {instances.max()} condensates")
        >>> 
        >>> # Get full output
        >>> result = pipeline.segment(image, full_output=True)
        >>> print(result['num_instances'])
    """
    
    def __init__(
        self,
        model,
        device: torch.device,
        # Preprocessing defaults
        low_percentile: float = 0.5,
        high_percentile: float = 99.5,
        # Postprocessing defaults
        prob_threshold: float = 0.15,
        min_size: int = 15,
        max_size: int = 600,
        flow_steps: int = 200,
        step_scale: float = 0.1
    ):
        """
        Initialize pipeline with a loaded model.
        
        Use from_pretrained() or from_local() class methods instead of
        calling this directly.
        """
        self.model = model
        self.device = device
        
        # Store defaults
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.prob_threshold = prob_threshold
        self.min_size = min_size
        self.max_size = max_size
        self.flow_steps = flow_steps
        self.step_scale = step_scale
    
    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        device: Optional[str] = None,
        **kwargs
    ) -> 'CondensateNetPipeline':
        """
        Load pipeline from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID (default: rajlab/condensatenet)
            device: Device to run on ('cuda', 'cpu', 'mps'). Auto-detects if None.
            **kwargs: Preprocessing/postprocessing parameters to override defaults:
                - low_percentile (float): Lower percentile for normalization
                - high_percentile (float): Upper percentile for normalization
                - prob_threshold (float): Probability threshold for foreground
                - min_size (int): Minimum instance size in pixels
                - max_size (int): Maximum instance size in pixels
                - flow_steps (int): Number of flow integration steps
                - step_scale (float): Base step size for flow integration
        
        Returns:
            CondensateNetPipeline instance
        
        Example:
            >>> pipeline = CondensateNetPipeline.from_pretrained()
            >>> pipeline = CondensateNetPipeline.from_pretrained(device="cuda")
            >>> pipeline = CondensateNetPipeline.from_pretrained(prob_threshold=0.2)
        """
        device_obj = get_device(device)
        model = load_model(repo_id, device=str(device_obj))
        
        # Separate model kwargs from pipeline kwargs
        pipeline_kwargs = {
            k: kwargs.pop(k) for k in list(kwargs.keys())
            if k in ['low_percentile', 'high_percentile', 'prob_threshold',
                     'min_size', 'max_size', 'flow_steps', 'step_scale']
        }
        
        return cls(model=model, device=device_obj, **pipeline_kwargs)
    
    @classmethod
    def from_local(
        cls,
        model_path: str,
        device: Optional[str] = None,
        **kwargs
    ) -> 'CondensateNetPipeline':
        """
        Load pipeline from a local directory.
        
        Args:
            model_path: Path to directory containing model files
            device: Device to run on ('cuda', 'cpu', 'mps'). Auto-detects if None.
            **kwargs: Preprocessing/postprocessing parameters
        
        Returns:
            CondensateNetPipeline instance
        
        Example:
            >>> # For Docker deployments
            >>> pipeline = CondensateNetPipeline.from_local("/models/condensatenet")
        """
        device_obj = get_device(device)
        model = load_model(model_path, device=str(device_obj))
        
        pipeline_kwargs = {
            k: kwargs.pop(k) for k in list(kwargs.keys())
            if k in ['low_percentile', 'high_percentile', 'prob_threshold',
                     'min_size', 'max_size', 'flow_steps', 'step_scale']
        }
        
        return cls(model=model, device=device_obj, **pipeline_kwargs)
    
    def preprocess(
        self,
        image: np.ndarray,
        low_percentile: Optional[float] = None,
        high_percentile: Optional[float] = None
    ) -> np.ndarray:
        """
        Normalize an image for model input.
        
        Args:
            image: Raw image array (any dtype)
            low_percentile: Override default low percentile
            high_percentile: Override default high percentile
        
        Returns:
            Normalized image as float32 array in [0, 1]
        """
        return normalize_image(
            image,
            low_percentile=low_percentile or self.low_percentile,
            high_percentile=high_percentile or self.high_percentile
        )
    
    def predict(
        self,
        image: np.ndarray,
        skip_preprocessing: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run model inference to get probability mask and flow vectors.
        
        Args:
            image: Input image (raw or pre-normalized)
            skip_preprocessing: If True, assume image is already normalized
        
        Returns:
            Tuple of (mask_probs, flows):
                - mask_probs: Probability map of shape (H, W) in [0, 1]
                - flows: Flow vectors of shape (2, H, W)
        """
        # Preprocess if needed
        if not skip_preprocessing:
            image = self.preprocess(image)
        
        # Prepare tensor
        x = prepare_tensor(image, device=str(self.device))
        
        # Run inference
        with torch.no_grad():
            out = self.model(x)
            mask_probs = torch.sigmoid(out['mask']).cpu().numpy()[0, 0]
            flows = out['flows'].cpu().numpy()[0]
        
        return mask_probs, flows
    
    def postprocess(
        self,
        mask_probs: np.ndarray,
        flows: np.ndarray,
        prob_threshold: Optional[float] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        flow_steps: Optional[int] = None,
        step_scale: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert model outputs to instance segmentation.
        
        Args:
            mask_probs: Probability map from predict()
            flows: Flow vectors from predict()
            prob_threshold: Override default probability threshold
            min_size: Override default minimum instance size
            max_size: Override default maximum instance size
            flow_steps: Override default number of integration steps
            step_scale: Override default step scale
        
        Returns:
            Instance segmentation map of shape (H, W)
        """
        return flow_to_instances(
            mask_probs,
            flows,
            prob_threshold=prob_threshold if prob_threshold is not None else self.prob_threshold,
            min_size=min_size if min_size is not None else self.min_size,
            max_size=max_size if max_size is not None else self.max_size,
            flow_steps=flow_steps if flow_steps is not None else self.flow_steps,
            step_scale=step_scale if step_scale is not None else self.step_scale
        )
    
    def segment(
        self,
        image: np.ndarray,
        full_output: bool = False,
        skip_preprocessing: bool = False,
        # Override parameters
        low_percentile: Optional[float] = None,
        high_percentile: Optional[float] = None,
        prob_threshold: Optional[float] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        flow_steps: Optional[int] = None,
        step_scale: Optional[float] = None
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Segment condensates in an image.
        
        This is the main entry point for most users. It runs the full
        pipeline: preprocessing → inference → postprocessing.
        
        Args:
            image: Input image as numpy array
            full_output: If True, return dict with all outputs. If False, return
                        just the instance segmentation array.
            skip_preprocessing: If True, assume image is already normalized
            
            # Override parameters (use pipeline defaults if not specified):
            low_percentile: Lower percentile for normalization
            high_percentile: Upper percentile for normalization
            prob_threshold: Probability threshold for foreground detection
            min_size: Minimum instance size in pixels
            max_size: Maximum instance size in pixels
            flow_steps: Number of flow integration steps
            step_scale: Base step size for flow integration
        
        Returns:
            If full_output=False (default):
                np.ndarray of shape (H, W) with instance labels (0=background)
            
            If full_output=True:
                Dict with keys:
                    - 'instances': Instance segmentation array
                    - 'mask_probs': Probability map
                    - 'flows': Flow vectors (2, H, W)
                    - 'num_instances': Number of detected instances
        
        Example:
            >>> # Simple usage
            >>> instances = pipeline.segment(image)
            >>> print(f"Found {instances.max()} condensates")
            
            >>> # Full output
            >>> result = pipeline.segment(image, full_output=True)
            >>> plt.imshow(result['mask_probs'], cmap='hot')
            
            >>> # Override parameters
            >>> instances = pipeline.segment(image, prob_threshold=0.2, min_size=20)
        """
        # Preprocess
        if not skip_preprocessing:
            normalized = normalize_image(
                image,
                low_percentile=low_percentile or self.low_percentile,
                high_percentile=high_percentile or self.high_percentile
            )
        else:
            normalized = image
        
        # Predict
        mask_probs, flows = self.predict(normalized, skip_preprocessing=True)
        
        # Postprocess
        instances = self.postprocess(
            mask_probs,
            flows,
            prob_threshold=prob_threshold,
            min_size=min_size,
            max_size=max_size,
            flow_steps=flow_steps,
            step_scale=step_scale
        )
        
        if full_output:
            return {
                'instances': instances,
                'mask_probs': mask_probs,
                'flows': flows,
                'num_instances': int(instances.max())
            }
        
        return instances
    
    def __repr__(self) -> str:
        return (
            f"CondensateNetPipeline(\n"
            f"  device={self.device},\n"
            f"  prob_threshold={self.prob_threshold},\n"
            f"  min_size={self.min_size},\n"
            f"  max_size={self.max_size},\n"
            f"  flow_steps={self.flow_steps}\n"
            f")"
        )
