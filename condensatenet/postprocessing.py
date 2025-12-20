"""Postprocessing for CondensateNet outputs."""

import numpy as np
from typing import Tuple, Dict, Any


def flow_to_instances(
    mask_probs: np.ndarray,
    flows: np.ndarray,
    prob_threshold: float = 0.15,
    min_size: int = 15,
    max_size: int = 600,
    flow_steps: int = 200,
    step_scale: float = 0.1
) -> np.ndarray:
    """
    Convert probability mask and flow vectors to instance segmentation.
    
    Uses flow integration to cluster pixels into instances. Each pixel
    follows the flow vectors to converge at instance centers.
    
    Args:
        mask_probs: Probability map of shape (H, W) with values in [0, 1]
        flows: Flow vectors of shape (2, H, W) pointing toward instance centers
        prob_threshold: Minimum probability to consider as foreground (default: 0.15)
        min_size: Minimum instance size in pixels (default: 15)
        max_size: Maximum instance size in pixels (default: 600)
        flow_steps: Number of integration steps (default: 200)
        step_scale: Base step size for integration (default: 0.1)
    
    Returns:
        Instance segmentation map of shape (H, W) with integer labels.
        Background is 0, instances are labeled 1, 2, 3, ...
    
    Example:
        >>> instances = flow_to_instances(mask_probs, flows, prob_threshold=0.2)
        >>> num_condensates = instances.max()
    """
    from skimage.morphology import remove_small_objects
    
    H, W = mask_probs.shape
    
    # Get foreground pixels
    binary_mask = mask_probs > prob_threshold
    yy, xx = np.nonzero(binary_mask)
    
    if len(yy) == 0:
        return np.zeros((H, W), dtype=np.int32)
    
    # Flow integration
    positions = np.stack([yy, xx], axis=1).astype(np.float32)
    coords = positions.copy()
    
    for step in range(flow_steps):
        y_int = np.clip(positions[:, 0].astype(int), 0, H - 1)
        x_int = np.clip(positions[:, 1].astype(int), 0, W - 1)
        
        dy = flows[0, y_int, x_int]
        dx = flows[1, y_int, x_int]
        
        # Adaptive step size based on flow magnitude
        flow_mag = np.sqrt(dy**2 + dx**2)
        step_size = np.clip(step_scale / (flow_mag + 1e-6), 0.1, 1.0)
        
        positions[:, 0] += dy * step_size
        positions[:, 1] += dx * step_size
        positions[:, 0] = np.clip(positions[:, 0], 0, H - 1)
        positions[:, 1] = np.clip(positions[:, 1], 0, W - 1)
    
    # Cluster by endpoint
    endpoints = np.round(positions).astype(int)
    instances = np.zeros((H, W), dtype=np.int32)
    cluster_ids = {}
    inst_id = 1
    
    for (orig_y, orig_x), (end_y, end_x) in zip(coords.astype(int), endpoints):
        key = (int(end_y), int(end_x))
        if key not in cluster_ids:
            cluster_ids[key] = inst_id
            inst_id += 1
        instances[orig_y, orig_x] = cluster_ids[key]
    
    # Remove small objects
    instances = remove_small_objects(instances, min_size=min_size)
    
    # Relabel and filter by size
    unique_ids = np.unique(instances)
    unique_ids = unique_ids[unique_ids > 0]
    final_instances = np.zeros_like(instances)
    new_id = 1
    
    for old_id in unique_ids:
        mask = instances == old_id
        size = mask.sum()
        if min_size <= size <= max_size:
            final_instances[mask] = new_id
            new_id += 1
    
    return final_instances


def extract_instance_properties(
    instances: np.ndarray,
    image: np.ndarray = None,
    mask_probs: np.ndarray = None
) -> list:
    """
    Extract properties for each detected instance.
    
    Args:
        instances: Instance segmentation map of shape (H, W)
        image: Optional original image for intensity measurements
        mask_probs: Optional probability map for confidence scores
    
    Returns:
        List of dicts, one per instance, with properties:
            - id: Instance label
            - area: Number of pixels
            - centroid: (y, x) center of mass
            - bbox: (min_row, min_col, max_row, max_col)
            - mean_intensity: Mean intensity in original image (if provided)
            - mean_prob: Mean probability score (if provided)
    """
    from skimage import measure
    
    props = []
    
    for inst_id in range(1, instances.max() + 1):
        mask = instances == inst_id
        
        if not mask.any():
            continue
        
        ys, xs = np.where(mask)
        
        prop = {
            'id': inst_id,
            'area': int(mask.sum()),
            'centroid': (float(ys.mean()), float(xs.mean())),
            'bbox': (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
        }
        
        if image is not None:
            prop['mean_intensity'] = float(image[mask].mean())
        
        if mask_probs is not None:
            prop['mean_prob'] = float(mask_probs[mask].mean())
        
        props.append(prop)
    
    return props
