"""Combined loss function for condensate segmentation."""

import torch
import torch.nn as nn


class CondensateLoss(nn.Module):
    """
    Combined focal + Tversky + flow MSE loss for condensate segmentation.

    Default parameters are tuned for sparse, small, low-contrast condensates.
    flow_scale scales the loss contribution, NOT the target values.
    """

    def __init__(self,
                 flow_scale=5.0,
                 alpha=0.90,
                 gamma=2.5,
                 focal_weight=0.4,
                 dice_weight=0.6,
                 flow_weight=1.0,
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

        # === 4. COMBINE AND CLEAN ===
        total_loss = (
            self.focal_weight * focal_loss +
            self.dice_weight * dice_loss +
            flow_loss
        )
        total_loss = torch.nan_to_num(total_loss)

        return total_loss, {
            "focal": float(torch.nan_to_num(focal_loss)),
            "dice": float(torch.nan_to_num(dice_loss)),
            "flow": float(torch.nan_to_num(flow_loss)),
            "total": float(torch.nan_to_num(total_loss))
        }
