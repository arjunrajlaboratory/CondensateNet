# training/data/flows.py
"""Cellpose-style flow field generation via heat diffusion."""

import numpy as np
import warnings
from typing import Tuple
from scipy.ndimage import find_objects

import torch
import torch.nn.functional as F


class FlowGenerator:
    """Generate Cellpose-style flow fields via heat diffusion from instance centers."""

    def __init__(self, device: str = 'auto', niter: int = 200,
                 use_bbox: bool = True, min_size: int = 15):
        self.niter = niter
        self.use_bbox = use_bbox
        self.min_size = min_size

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def generate(self, mask: np.ndarray) -> np.ndarray:
        """Generate flow fields from instance mask."""
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got {mask.shape}")

        if mask.max() == 0:
            return np.zeros((2, *mask.shape), dtype=np.float32)

        if self.use_bbox:
            return self._generate_with_bbox(mask)
        else:
            return self._generate_full(mask)

    def _generate_full(self, mask: np.ndarray) -> np.ndarray:
        """Generate flows for entire mask using heat diffusion."""
        Ly, Lx = mask.shape

        # Pad mask
        masks_padded = torch.from_numpy(mask.astype("int64")).to(self.device)
        masks_padded = F.pad(masks_padded, (1, 1, 1, 1))
        shape = masks_padded.shape

        # Get mask pixel neighbors
        y, x = torch.nonzero(masks_padded, as_tuple=True)
        y, x = y.int(), x.int()

        neighbors = torch.zeros((2, 9, y.shape[0]), dtype=torch.int, device=self.device)
        yxi = [[0, -1, 1, 0, 0, -1, -1, 1, 1], [0, 0, 0, -1, 1, -1, 1, -1, 1]]
        for i in range(9):
            neighbors[0, i] = y + yxi[0][i]
            neighbors[1, i] = x + yxi[1][i]

        # Check valid neighbors (same instance)
        isneighbor = torch.ones((9, y.shape[0]), dtype=torch.bool, device=self.device)
        m0 = masks_padded[neighbors[0, 0], neighbors[1, 0]]
        for i in range(1, 9):
            isneighbor[i] = masks_padded[neighbors[0, i], neighbors[1, i]] == m0

        # Get centers
        slices = find_objects(mask)
        centers = self._get_centers(mask, slices)
        meds_p = torch.from_numpy(centers).to(self.device).long() + 1  # padding

        # Compute number of iterations based on max extent
        exts = np.array([(slc[0].stop - slc[0].start) + (slc[1].stop - slc[1].start) + 2
                        for slc in slices])
        n_iter = 2 * exts.max() if self.niter is None else self.niter

        # Heat diffusion
        mu = self._extend_centers_gpu(neighbors, meds_p, isneighbor, shape, n_iter)

        # Normalize
        mu = mu.astype("float64")
        mu /= (1e-60 + (mu**2).sum(axis=0)**0.5)

        # Remove padding
        mu0 = np.zeros((2, Ly, Lx), dtype=np.float32)
        mu0[:, y.cpu().numpy() - 1, x.cpu().numpy() - 1] = mu

        return mu0

    def _generate_with_bbox(self, mask: np.ndarray) -> np.ndarray:
        """Generate flows using bounding box optimization."""
        h, w = mask.shape
        flows = np.zeros((2, h, w), dtype=np.float32)

        labels = np.unique(mask)
        labels = labels[labels > 0]

        if len(labels) == 0:
            return flows

        slices = find_objects(mask)

        for label in labels:
            label_idx = int(label) - 1

            if label_idx >= len(slices) or slices[label_idx] is None:
                continue

            bbox = slices[label_idx]
            mask_region = mask[bbox]
            instance_mask = (mask_region == label)

            if instance_mask.sum() < self.min_size:
                continue

            # Process region with heat diffusion
            region_flows = self._compute_instance_flows_diffusion(instance_mask)

            # Assign to output
            flows[0][bbox][instance_mask] = region_flows[0][instance_mask]
            flows[1][bbox][instance_mask] = region_flows[1][instance_mask]

        return flows

    def _compute_instance_flows_diffusion(self, instance_mask: np.ndarray) -> np.ndarray:
        """Compute flows for single instance using heat diffusion."""
        h, w = instance_mask.shape

        # Pad
        mask_padded = torch.from_numpy(instance_mask.astype("int64")).to(self.device)
        mask_padded = F.pad(mask_padded, (1, 1, 1, 1))
        shape = mask_padded.shape

        # Get pixels
        y, x = torch.nonzero(mask_padded, as_tuple=True)
        y, x = y.int(), x.int()

        if len(y) == 0:
            return np.zeros((2, h, w), dtype=np.float32)

        # Get neighbors
        neighbors = torch.zeros((2, 9, y.shape[0]), dtype=torch.int, device=self.device)
        yxi = [[0, -1, 1, 0, 0, -1, -1, 1, 1], [0, 0, 0, -1, 1, -1, 1, -1, 1]]
        for i in range(9):
            neighbors[0, i] = y + yxi[0][i]
            neighbors[1, i] = x + yxi[1][i]

        isneighbor = torch.ones((9, y.shape[0]), dtype=torch.bool, device=self.device)
        m0 = mask_padded[neighbors[0, 0], neighbors[1, 0]]
        for i in range(1, 9):
            isneighbor[i] = mask_padded[neighbors[0, i], neighbors[1, i]] == m0

        # Get center
        yi, xi = np.nonzero(instance_mask)
        center = np.array([int(np.round(yi.mean())), int(np.round(xi.mean()))])

        # Check if center is in mask
        if not instance_mask[center[0], center[1]]:
            imin = ((xi - center[1])**2 + (yi - center[0])**2).argmin()
            center = np.array([yi[imin], xi[imin]])

        meds_p = torch.from_numpy(center).unsqueeze(0).to(self.device).long() + 1

        # Heat diffusion
        n_iter = min(2 * max(h, w), self.niter)
        mu = self._extend_centers_gpu(neighbors, meds_p, isneighbor, shape, n_iter)

        # Normalize
        mu = mu.astype("float64")
        mu /= (1e-60 + (mu**2).sum(axis=0)**0.5)

        # Remove padding
        mu0 = np.zeros((2, h, w), dtype=np.float32)
        mu0[:, y.cpu().numpy() - 1, x.cpu().numpy() - 1] = mu

        return mu0

    def _extend_centers_gpu(self, neighbors, meds, isneighbor, shape, n_iter):
        """Heat diffusion from centers to generate flows (Cellpose algorithm)."""
        T = torch.zeros(shape, dtype=torch.float32, device=self.device)

        # Diffusion iterations
        for i in range(n_iter):
            T[tuple(meds.T)] += 1
            Tneigh = T[tuple(neighbors)]
            Tneigh *= isneighbor
            T[tuple(neighbors[:, 0])] = Tneigh.mean(axis=0)

        # Compute gradients (finite differences)
        grads = T[neighbors[0, [2, 1, 4, 3]], neighbors[1, [2, 1, 4, 3]]]
        dy = grads[0] - grads[1]
        dx = grads[2] - grads[3]

        mu = np.stack((dy.cpu().numpy(), dx.cpu().numpy()), axis=0)
        return mu

    def _get_centers(self, mask: np.ndarray, slices) -> np.ndarray:
        """Get centers of mass for all instances."""
        centers = []
        for i, slc in enumerate(slices):
            if slc is None:
                continue
            yi, xi = np.nonzero(mask[slc] == (i + 1))
            if len(yi) == 0:
                continue
            ymean = int(np.round(yi.mean()))
            xmean = int(np.round(xi.mean()))

            # Check if center is in mask
            if not ((yi == ymean) & (xi == xmean)).any():
                imin = ((xi - xmean)**2 + (yi - ymean)**2).argmin()
                ymean, xmean = yi[imin], xi[imin]

            centers.append([ymean + slc[0].start, xmean + slc[1].start])

        return np.array(centers) if centers else np.zeros((0, 2))
