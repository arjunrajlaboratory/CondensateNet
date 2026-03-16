"""Combined loss function for condensate segmentation."""

import torch
import torch.nn as nn


class CondensateLoss(nn.Module):
    """
    Combined focal + Tversky + flow MSE loss for condensate segmentation.

    Default parameters are tuned for sparse, small, low-contrast condensates.
    flow_scale scales the loss contribution, NOT the target values.

    Optional calibration loss penalizes the gap between predicted confidence
    and actual accuracy per probability bin (differentiable ECE).
    """

    def __init__(self,
                 flow_scale=5.0,
                 alpha=0.90,
                 gamma=2.5,
                 focal_weight=0.4,
                 dice_weight=0.6,
                 flow_weight=1.0,
                 calibration_weight=0.0,
                 calibration_bins=10,
                 dice_mode='soft',
                 mask_flows='soft',
                 tversky_alpha=0.5,
                 tversky_beta=0.5,
                 smooth=1e-6):
        super().__init__()
        self.flow_scale = flow_scale
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.flow_weight = flow_weight
        self.calibration_weight = calibration_weight
        self.calibration_bins = calibration_bins
        self.dice_mode = dice_mode.lower()
        self.mask_flows = mask_flows.lower()
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.smooth = smooth

    def forward(self, preds, batch):
        y_mask = preds["mask"]    # (B,1,H,W)
        y_flows = preds["flows"]  # (B,2,H,W)
        lbl_mask = batch["mask"]  # (B,1,H,W)
        lbl_flows = batch["flows"]# (B,2,H,W)

        # --- numerically stable sigmoid ---
        probs = torch.sigmoid(torch.clamp(y_mask, -20, 20))
        probs = torch.clamp(probs, self.smooth, 1.0 - self.smooth)

        # === 1. FOCAL LOSS ===
        focal_pos = -self.alpha * lbl_mask * (1 - probs)**self.gamma * torch.log(probs)
        focal_neg = -(1 - self.alpha) * (1 - lbl_mask) * probs**self.gamma * torch.log(1 - probs)
        focal_loss = torch.nanmean(focal_pos + focal_neg)

        # === 2. DICE OR SOFT DICE (TVERSKY) ===
        if self.dice_mode == "soft":
            tp = (probs * lbl_mask).sum()
            fp = (probs * (1 - lbl_mask)).sum()
            fn = ((1 - probs) * lbl_mask).sum()
            denom = tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.smooth
            dice_loss = 1 - (tp + self.smooth) / denom
        else:
            inter = (probs * lbl_mask).sum()
            denom = probs.sum() + lbl_mask.sum() + self.smooth
            dice_loss = 1 - (2 * inter + self.smooth) / denom

        # === 3. FLOW MSE LOSS (OPTIONALLY MASKED) ===
        # FIXED: Compare predictions to ground truth at their natural scale
        # Both y_flows and lbl_flows should have magnitude ~1.0
        flow_diff = (y_flows - lbl_flows) ** 2  # Changed from: (y_flows - self.flow_scale * lbl_flows) ** 2

        if self.mask_flows == "soft":
            mask_w = probs
        elif self.mask_flows == "hard":
            mask_w = lbl_mask
        else:
            mask_w = 1.0

        valid = mask_w.sum() > 0
        if valid:
            # FIXED: Apply flow_scale to the loss contribution, not to the targets
            flow_loss = (flow_diff * mask_w).mean() * self.flow_weight * self.flow_scale
        else:
            flow_loss = torch.tensor(0.0, device=y_mask.device)

        # === 4. CALIBRATION LOSS ===
        # Differentiable ECE: for each probability bin, penalize the gap between
        # mean predicted confidence and actual positive fraction.
        # Uses soft bin assignment so gradients flow through.
        if self.calibration_weight > 0:
            cal_loss = self._calibration_loss(probs, lbl_mask)
        else:
            cal_loss = torch.tensor(0.0, device=y_mask.device)

        # === 5. COMBINE AND CLEAN ===
        total_loss = (
            self.focal_weight * focal_loss +
            self.dice_weight * dice_loss +
            flow_loss +
            self.calibration_weight * cal_loss
        )
        total_loss = torch.nan_to_num(total_loss)

        return total_loss, {
            "focal": float(torch.nan_to_num(focal_loss).detach()),
            "dice": float(torch.nan_to_num(dice_loss).detach()),
            "flow": float(torch.nan_to_num(flow_loss).detach()),
            "calibration": float(torch.nan_to_num(cal_loss).detach()),
            "total": float(torch.nan_to_num(total_loss).detach())
        }

    def _calibration_loss(self, probs, labels):
        """Differentiable Expected Calibration Error (ECE).

        Bins predicted probabilities and penalizes the squared difference
        between mean confidence and actual accuracy in each bin.
        Uses soft bin assignment (triangular kernels) so gradients flow.
        """
        probs_flat = probs.view(-1)
        labels_flat = labels.view(-1)
        n_bins = self.calibration_bins

        # Bin centers: e.g. for 10 bins -> [0.05, 0.15, ..., 0.95]
        bin_width = 1.0 / n_bins
        bin_centers = torch.linspace(
            bin_width / 2, 1.0 - bin_width / 2, n_bins,
            device=probs.device
        )

        # Soft bin assignment using triangular kernel
        # Each pixel contributes to the nearest bin(s) proportionally
        # Shape: (n_pixels, n_bins)
        distances = (probs_flat.unsqueeze(1) - bin_centers.unsqueeze(0)).abs()
        weights = torch.clamp(1.0 - distances / bin_width, min=0.0)

        # Weighted mean confidence and accuracy per bin
        bin_counts = weights.sum(dim=0) + self.smooth
        bin_confidence = (weights * probs_flat.unsqueeze(1)).sum(dim=0) / bin_counts
        bin_accuracy = (weights * labels_flat.unsqueeze(1)).sum(dim=0) / bin_counts

        # ECE: weighted average of |confidence - accuracy| per bin
        # Use squared error for smoother gradients
        bin_fractions = bin_counts / (probs_flat.numel() + self.smooth)
        cal_error = (bin_fractions * (bin_confidence - bin_accuracy) ** 2).sum()

        return cal_error
