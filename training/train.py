# training/train.py
"""Training loop for condensate segmentation model."""

import time
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from pathlib import Path

from .config import PipelineConfig
from .model import create_condensate_model
from .loss import CondensateLoss
from .data.registry import DataRegistry, quick_split
from .data.dataset import create_dataloaders


# ============================================================
#   METRICS
# ============================================================

def compute_condensate_metrics(pred_mask_logits, pred_flows, gt_mask, gt_flows, thr=0.5):
    """
    Compute segmentation and flow metrics at a fixed threshold.

    FIXED: Removed magnitude error computation (Issue 8).
    MSE loss on unit-normalized flows already handles magnitude implicitly.
    """
    with torch.no_grad():
        eps = 1e-8
        probs = torch.sigmoid(pred_mask_logits)
        pred_binary = (probs > thr).float()

        # Segmentation metrics
        tp = (pred_binary * gt_mask).sum()
        fp = (pred_binary * (1 - gt_mask)).sum()
        fn = ((1 - pred_binary) * gt_mask).sum()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * tp / (2 * tp + fp + fn + eps)
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = tp / (tp + fp + fn + eps)

        seg = {
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
            "dice": dice.item(),
            "iou": iou.item(),
        }

        # --- Flow metrics (only over FG) ---
        fg_mask = (gt_mask > 0).float()
        if fg_mask.sum() > 0:
            flow_diff = pred_flows - gt_flows
            flow_l2 = torch.sqrt((flow_diff ** 2).sum(dim=1) + eps)
            masked_flow_error = (flow_l2 * fg_mask.squeeze(1)).sum() / (fg_mask.sum() + eps)

            flow = {
                "flow_l2_error": masked_flow_error.item(),
            }
        else:
            flow = {"flow_l2_error": 0.0}

        return {**seg, **flow}


def adaptive_threshold_search(pred_mask_logits, gt_mask, thresholds=None):
    """Search for the best threshold maximizing F1 for current batch."""
    if thresholds is None:
        thresholds = torch.linspace(0.1, 0.9, steps=17).to(pred_mask_logits.device)

    best_thr, best_f1 = 0.5, 0.0
    for thr in thresholds:
        # Only need mask metrics for threshold search
        metrics = compute_condensate_metrics(
            pred_mask_logits,
            torch.zeros_like(pred_mask_logits),
            gt_mask,
            torch.zeros_like(gt_mask),
            thr=thr.item()
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thr = thr.item()
    return best_thr, best_f1


# ============================================================
#   TRAINING LOOP
# ============================================================

def train_condensate_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    config,
    num_epochs=80,
    lr=2e-4,
    weight_decay=1e-4,
    save_dir=None,
    device=None,
    mixed_precision=True,
    gradient_clip_value=5.0,
):
    """
    Train condensate segmentation model with adaptive thresholding and precision logging.

    Uses AdamW optimizer and ReduceLROnPlateau scheduler. Always monitors gradient
    norms and logs detailed training diagnostics.

    Args:
        model: The segmentation model to train.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        loss_fn: Loss function (e.g. CondensateLoss).
        config: PipelineConfig instance.
        num_epochs: Number of training epochs (default: 80).
        lr: Learning rate (default: 2e-4).
        weight_decay: Weight decay for AdamW (default: 1e-4).
        save_dir: Directory for checkpoints (default: config.output_dir/checkpoints).
        device: Torch device (default: auto-detect).
        mixed_precision: Whether to use AMP (default: True).
        gradient_clip_value: Max gradient norm (default: 5.0).

    Returns:
        Tuple of (model, history) where history is a dict of training metrics.
    """

    save_dir = Path(save_dir or config.output_dir / "checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler = GradScaler(device.type if mixed_precision else "cpu", enabled=mixed_precision)

    # === Optimizer (always AdamW) ===
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # === Scheduler (always ReduceLROnPlateau) ===
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, cooldown=5, min_lr=1e-6
    )

    # === History tracking ===
    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "val_dice": [], "val_iou": [], "val_recall": [],
        "val_precision": [], "lr": [],
        "train_focal": [], "train_dice_loss": [], "train_flow": [],
        "val_focal": [], "val_dice_loss": [], "val_flow": [],
        "grad_norms": []
    }

    print(f"\nTraining on {device} using AdamW | Loss: {loss_fn.__class__.__name__}")
    print(f"   Gradient clipping: {gradient_clip_value}")
    print(f"   Mixed precision: {mixed_precision}")
    print("-" * 80)

    best_val_dice, best_epoch = 0.0, 0

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()

        # Training metrics
        train_loss, train_f1 = 0.0, 0.0
        train_focal, train_dice_loss, train_flow_loss = 0.0, 0.0, 0.0
        grad_norms = []
        n_train = 0

        for batch_idx, batch in enumerate(train_loader):
            imgs = batch["image"].to(device)
            gt_m = batch["mask"].to(device)
            gt_f = batch["flows"].to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            with autocast(device_type=device.type, enabled=mixed_precision):
                preds = model(imgs)
                loss, loss_dict = loss_fn(preds, {"mask": gt_m, "flows": gt_f})

            # Backward pass
            scaler.scale(loss).backward()

            # Monitor gradient norm BEFORE clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float('inf')
            ).item()
            grad_norms.append(grad_norm)

            # Warn if gradients are exploding
            if grad_norm > 20.0:
                print(f"\n  WARNING: Large gradient norm {grad_norm:.1f} at epoch {epoch+1}, batch {batch_idx}")
                print(f"      Loss components: focal={loss_dict['focal']:.4f}, "
                      f"dice={loss_dict['dice']:.4f}, flow={loss_dict['flow']:.4f}")

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)

            scaler.step(optimizer)
            scaler.update()

            # Track metrics
            metrics = compute_condensate_metrics(preds["mask"], preds["flows"], gt_m, gt_f)
            train_loss += loss.item()
            train_f1 += metrics["f1"]
            train_focal += loss_dict["focal"]
            train_dice_loss += loss_dict["dice"]
            train_flow_loss += loss_dict["flow"]
            n_train += 1

            # Detect loss component imbalance
            if batch_idx % 10 == 0 and batch_idx > 0:
                focal_avg = train_focal / n_train
                dice_avg = train_dice_loss / n_train
                flow_avg = train_flow_loss / n_train

                # Warn if flow loss is dominating
                if flow_avg > 10 * (focal_avg + dice_avg):
                    print(f"\n  WARNING: Flow loss dominating at epoch {epoch+1}, batch {batch_idx}")
                    print(f"      focal={focal_avg:.4f}, dice={dice_avg:.4f}, flow={flow_avg:.4f}")
                    print(f"      Consider reducing flow_scale parameter")

                # Detailed logging every 10 batches
                if len(grad_norms) > 0:
                    recent_grad_norm = np.mean(grad_norms[-10:])
                    print(f"  [Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f}, F1: {metrics['f1']:.3f}, "
                          f"Grad: {recent_grad_norm:.2f}")

        # Average training metrics
        avg_train_loss = train_loss / max(1, n_train)
        avg_train_f1 = train_f1 / max(1, n_train)
        avg_train_focal = train_focal / max(1, n_train)
        avg_train_dice = train_dice_loss / max(1, n_train)
        avg_train_flow = train_flow_loss / max(1, n_train)
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0

        # === Validation ===
        model.eval()
        val_loss, val_f1, val_dice, val_iou, val_recall, val_precision = 0, 0, 0, 0, 0, 0
        val_focal, val_dice_loss, val_flow_loss = 0.0, 0.0, 0.0
        n_val = 0
        best_thresholds = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                gt_m = batch["mask"].to(device)
                gt_f = batch["flows"].to(device)

                preds = model(imgs)
                loss, loss_dict = loss_fn(preds, {"mask": gt_m, "flows": gt_f})

                # --- Adaptive threshold search ---
                best_thr, _ = adaptive_threshold_search(preds["mask"], gt_m)
                best_thresholds.append(best_thr)

                # Compute metrics at best threshold
                metrics = compute_condensate_metrics(
                    preds["mask"], preds["flows"], gt_m, gt_f, thr=best_thr
                )

                val_loss += loss.item()
                val_f1 += metrics["f1"]
                val_dice += metrics["dice"]
                val_iou += metrics["iou"]
                val_recall += metrics["recall"]
                val_precision += metrics["precision"]
                val_focal += loss_dict["focal"]
                val_dice_loss += loss_dict["dice"]
                val_flow_loss += loss_dict["flow"]
                n_val += 1

        # Average validation metrics
        avg_val_loss = val_loss / max(1, n_val)
        avg_val_f1 = val_f1 / max(1, n_val)
        avg_val_dice = val_dice / max(1, n_val)
        avg_val_iou = val_iou / max(1, n_val)
        avg_val_recall = val_recall / max(1, n_val)
        avg_val_precision = val_precision / max(1, n_val)
        avg_val_focal = val_focal / max(1, n_val)
        avg_val_dice_loss = val_dice_loss / max(1, n_val)
        avg_val_flow = val_flow_loss / max(1, n_val)
        mean_best_thr = np.mean(best_thresholds) if best_thresholds else 0.5

        # Learning rate
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_dice)

        # Timing
        dt = time.time() - t0

        # === Logging ===
        print(f"Epoch {epoch+1:03d}/{num_epochs} | "
              f"Loss {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
              f"F1 {avg_train_f1:.3f}/{avg_val_f1:.3f} | "
              f"Dice {avg_val_dice:.3f} | IoU {avg_val_iou:.3f} | "
              f"P {avg_val_precision:.3f} | R {avg_val_recall:.3f} | "
              f"Thr {mean_best_thr:.2f} | LR {lr_now:.1e} | {dt:.1f}s")

        print(f"       Grad: {avg_grad_norm:.2f} | "
              f"Train [F:{avg_train_focal:.3f} D:{avg_train_dice:.3f} Fl:{avg_train_flow:.3f}] | "
              f"Val [F:{avg_val_focal:.3f} D:{avg_val_dice_loss:.3f} Fl:{avg_val_flow:.3f}]")

        # Store history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_f1"].append(avg_train_f1)
        history["val_f1"].append(avg_val_f1)
        history["val_dice"].append(avg_val_dice)
        history["val_iou"].append(avg_val_iou)
        history["val_recall"].append(avg_val_recall)
        history["val_precision"].append(avg_val_precision)
        history["lr"].append(lr_now)
        history["train_focal"].append(avg_train_focal)
        history["train_dice_loss"].append(avg_train_dice)
        history["train_flow"].append(avg_train_flow)
        history["val_focal"].append(avg_val_focal)
        history["val_dice_loss"].append(avg_val_dice_loss)
        history["val_flow"].append(avg_val_flow)
        history["grad_norms"].append(avg_grad_norm)

        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': avg_val_dice,
                'val_f1': avg_val_f1,
                'val_iou': avg_val_iou,
                'config': config,
            }, save_dir / "best_model.pt")
            print(f"  New best Dice: {best_val_dice:.3f} at epoch {epoch+1}")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, save_dir / f"checkpoint_epoch_{epoch+1}.pt")

    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Best Dice: {best_val_dice:.3f} at epoch {best_epoch+1}")
    print(f"Model saved to: {save_dir / 'best_model.pt'}")
    print(f"{'='*80}\n")

    return model, history


if __name__ == "__main__":
    config = PipelineConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data pipeline
    print("Scanning for image-mask pairs...")
    registry = DataRegistry(config.images_dir, config.masks_dir)
    manifest_df = registry.scan_and_validate()

    print("\nCreating train/val splits...")
    train_df, val_df = quick_split(
        manifest_df,
        experiment_caps=config.experiment_caps,
        val_ratio=config.val_ratio,
        output_dir=config.output_dir,
        random_seed=config.random_seed
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # 2. Dataloaders
    train_loader, val_loader = create_dataloaders(config)

    # 3. Model + loss
    model = create_condensate_model()
    loss_fn = CondensateLoss()

    # 4. Train
    model, history = train_condensate_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
    )
