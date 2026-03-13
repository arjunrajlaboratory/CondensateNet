# training/inference.py
"""Inference pipeline for condensate segmentation model."""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from skimage import measure
from typing import Dict, List, Tuple, Optional
import warnings

from .config import PipelineConfig
from .model import create_condensate_model
from .train import compute_condensate_metrics


class CondensateInference:
    """
    Inference pipeline for condensate segmentation model evaluation.
    """

    def __init__(self,
                 model_path: str,
                 device: str = None,
                 threshold: float = 0.5,
                 adaptive_threshold: bool = True):
        """
        Initialize inference pipeline.
        """
        self.model_path = Path(model_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.adaptive_threshold = adaptive_threshold

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Results storage
        self.results = {
            'metrics': [],
            'predictions': [],
            'ground_truths': [],
            'sample_info': []
        }

    def _load_model(self):
        """Load trained model from checkpoint - FIXED to handle different formats."""
        print(f"Loading model from {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Debug: print checkpoint keys
        print(f"Checkpoint keys: {checkpoint.keys()}")

        # Create model instance
        model = create_condensate_model()

        # Try different key names for model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Assume the checkpoint IS the state dict
            state_dict = checkpoint

        # Load state dict
        model.load_state_dict(state_dict)
        model = model.to(self.device)

        # Print checkpoint info if available
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"  Loaded model from epoch {checkpoint['epoch'] + 1}")
            if 'val_dice' in checkpoint:
                print(f"  Validation Dice: {checkpoint['val_dice']:.4f}")
            if 'val_f1' in checkpoint:
                print(f"  Validation F1: {checkpoint['val_f1']:.4f}")

        return model

    def find_optimal_threshold(self, pred_logits: torch.Tensor, gt_mask: torch.Tensor) -> float:
        """Find optimal threshold for current batch using F1 score."""
        thresholds = np.linspace(0.1, 0.9, 17)
        best_f1 = 0
        best_threshold = 0.5

        with torch.no_grad():
            for thr in thresholds:
                pred_binary = (torch.sigmoid(pred_logits) > thr).float()

                tp = (pred_binary * gt_mask).sum()
                fp = (pred_binary * (1 - gt_mask)).sum()
                fn = ((1 - pred_binary) * gt_mask).sum()

                f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thr

        return best_threshold

    def compute_instance_metrics(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict:
        """Compute instance-level segmentation metrics."""
        # Convert to binary if needed
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)

        # Label connected components
        pred_labels = measure.label(pred_binary, connectivity=2)
        gt_labels = measure.label(gt_binary, connectivity=2)

        # Count instances
        n_pred = pred_labels.max()
        n_gt = gt_labels.max()

        # IoU matching for instance evaluation
        matched_gt = 0
        iou_threshold = 0.5

        for gt_id in range(1, n_gt + 1):
            gt_instance = (gt_labels == gt_id)
            best_iou = 0

            for pred_id in range(1, n_pred + 1):
                pred_instance = (pred_labels == pred_id)

                intersection = (gt_instance & pred_instance).sum()
                union = (gt_instance | pred_instance).sum()

                if union > 0:
                    iou = intersection / union
                    best_iou = max(best_iou, iou)

            if best_iou > iou_threshold:
                matched_gt += 1

        # Calculate instance precision/recall
        instance_precision = matched_gt / max(n_pred, 1)
        instance_recall = matched_gt / max(n_gt, 1)
        instance_f1 = 2 * instance_precision * instance_recall / (
            instance_precision + instance_recall + 1e-8
        )

        return {
            'n_pred_instances': n_pred,
            'n_gt_instances': n_gt,
            'instance_precision': instance_precision,
            'instance_recall': instance_recall,
            'instance_f1': instance_f1
        }

    def test_single_batch(self, batch: Dict) -> Tuple[Dict, Dict]:
        """Run inference on a single batch and compute metrics."""
        # Move to device
        images = batch['image'].to(self.device)
        gt_masks = batch['mask'].to(self.device)
        gt_flows = batch['flows'].to(self.device)

        # Forward pass
        with torch.no_grad():
            predictions = self.model(images)

        # Find optimal threshold if needed
        if self.adaptive_threshold:
            threshold = self.find_optimal_threshold(predictions['mask'], gt_masks)
        else:
            threshold = self.threshold

        # Compute metrics using your existing function
        metrics = compute_condensate_metrics(
            predictions['mask'],
            predictions['flows'],
            gt_masks,
            gt_flows,
            thr=threshold
        )

        # Add instance metrics for first sample in batch
        pred_mask_binary = (torch.sigmoid(predictions['mask']) > threshold).float()
        if len(pred_mask_binary) > 0:
            pred_mask_np = pred_mask_binary[0].squeeze().cpu().numpy()
            gt_mask_np = gt_masks[0].squeeze().cpu().numpy()
            instance_metrics = self.compute_instance_metrics(pred_mask_np, gt_mask_np)
            metrics.update(instance_metrics)

        metrics['threshold'] = threshold

        return predictions, metrics

    def test_dataloader(self, dataloader, save_predictions: bool = False,
                       num_visualizations: int = 5, save_dir: Path = None) -> pd.DataFrame:
        """Test model on entire dataloader and compile results."""
        print(f"\nTesting on {len(dataloader)} batches...")
        all_metrics = []
        viz_count = 0
        save_dir = save_dir or Path('inference_results')
        save_dir.mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                try:
                    # Run inference
                    predictions, metrics = self.test_single_batch(batch)

                    # Store metrics
                    metrics['batch_idx'] = batch_idx
                    all_metrics.append(metrics)

                    # Store predictions if requested
                    if save_predictions:
                        self.results['predictions'].append(predictions)
                        self.results['ground_truths'].append({
                            'mask': batch['mask'],
                            'flows': batch['flows']
                        })

                    # Visualize first few batches
                    if viz_count < num_visualizations:
                        self.visualize_batch_results(
                            batch, predictions, metrics,
                            save_path=save_dir / f"inference_batch_{batch_idx}.png"
                        )
                        viz_count += 1

                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue

        # Create DataFrame with results
        df_metrics = pd.DataFrame(all_metrics)
        self.results['metrics'] = df_metrics

        # Print summary statistics
        self.print_summary(df_metrics)

        return df_metrics

    def print_summary(self, df: pd.DataFrame):
        """Print comprehensive summary of test results."""
        print("\n" + "="*80)
        print("VALIDATION SET INFERENCE RESULTS")
        print("="*80)

        # Pixel-level metrics
        print("\nPixel-Level Metrics (mean +/- std):")
        print("-"*40)
        for metric in ['dice', 'iou', 'f1', 'precision', 'recall']:
            if metric in df.columns:
                mean = df[metric].mean()
                std = df[metric].std()
                print(f"  {metric.upper():12s}: {mean:.4f} +/- {std:.4f}")

        # Flow metrics
        print("\nFlow Field Metrics:")
        print("-"*40)
        if 'flow_l2_error' in df.columns:
            mean = df['flow_l2_error'].mean()
            std = df['flow_l2_error'].std()
            print(f"  Flow L2 Error: {mean:.4f} +/- {std:.4f}")

        # Best and worst performing samples
        if len(df) > 0 and 'dice' in df.columns:
            print("\nPerformance Distribution:")
            print("-"*40)
            print(f"  Best Dice Score:  {df['dice'].max():.4f} (Batch {df['dice'].idxmax()})")
            print(f"  Worst Dice Score: {df['dice'].min():.4f} (Batch {df['dice'].idxmin()})")
            print(f"  Median Dice:      {df['dice'].median():.4f}")

            # Threshold statistics if adaptive
            if 'threshold' in df.columns:
                print(f"\n  Optimal Threshold (mean): {df['threshold'].mean():.3f}")
                print(f"  Threshold Range: [{df['threshold'].min():.3f}, {df['threshold'].max():.3f}]")

        print("\n" + "="*80)

    def visualize_batch_results(self, batch: Dict, predictions: Dict,
                               metrics: Dict, save_path: Path = None):
        """Visualize predictions vs ground truth for a batch."""
        # Take first sample from batch
        image = batch['image'][0].cpu()
        gt_mask = batch['mask'][0].cpu()
        gt_flows = batch['flows'][0].cpu()

        pred_mask_logits = predictions['mask'][0].cpu()
        pred_flows = predictions['flows'][0].cpu()

        # Apply threshold
        threshold = metrics.get('threshold', self.threshold)
        pred_mask = (torch.sigmoid(pred_mask_logits) > threshold).float()

        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Row 1: Masks
        # Original image
        ax = axes[0, 0]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title('Input Image')
        ax.axis('off')

        # Ground truth mask
        ax = axes[0, 1]
        ax.imshow(gt_mask.squeeze(), cmap='hot')
        ax.set_title('Ground Truth Mask')
        ax.axis('off')

        # Predicted mask
        ax = axes[0, 2]
        ax.imshow(pred_mask.squeeze(), cmap='hot')
        ax.set_title(f'Predicted Mask (thr={threshold:.2f})')
        ax.axis('off')

        # Overlay comparison
        ax = axes[0, 3]
        overlay = np.zeros((*gt_mask.squeeze().shape, 3))
        overlay[..., 0] = gt_mask.squeeze()  # Red for GT
        overlay[..., 1] = pred_mask.squeeze()  # Green for Pred
        ax.imshow(overlay)
        ax.set_title(f'Overlay (Dice={metrics["dice"]:.3f})')
        ax.axis('off')

        # Row 2: Flow fields
        # Ground truth flow
        ax = axes[1, 0]
        flow_magnitude_gt = torch.sqrt((gt_flows**2).sum(0))
        im = ax.imshow(flow_magnitude_gt, cmap='viridis')
        ax.set_title('GT Flow Magnitude')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Predicted flow
        ax = axes[1, 1]
        flow_magnitude_pred = torch.sqrt((pred_flows**2).sum(0))
        im = ax.imshow(flow_magnitude_pred, cmap='viridis')
        ax.set_title('Pred Flow Magnitude')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Flow error
        ax = axes[1, 2]
        flow_error = torch.abs(flow_magnitude_pred - flow_magnitude_gt)
        im = ax.imshow(flow_error, cmap='hot')
        ax.set_title(f'Flow Error')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Flow vectors
        ax = axes[1, 3]
        step = 10
        y, x = np.mgrid[0:pred_flows.shape[1]:step, 0:pred_flows.shape[2]:step]
        u = pred_flows[1, ::step, ::step].numpy()
        v = pred_flows[0, ::step, ::step].numpy()
        ax.quiver(x, y, u, v, flow_magnitude_pred[::step, ::step], cmap='coolwarm')
        ax.set_title('Predicted Flow Vectors')
        ax.set_aspect('equal')
        ax.invert_yaxis()

        plt.suptitle(f"Batch {metrics['batch_idx']} | Dice: {metrics['dice']:.3f}",
                     fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run condensate segmentation inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--val-manifest", default=None, help="Path to validation manifest CSV")
    parser.add_argument("--output-dir", default="inference_results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-viz", type=int, default=5, help="Number of visualizations")
    args = parser.parse_args()

    config = PipelineConfig()
    val_manifest = args.val_manifest or str(config.output_dir / "val_manifest.csv")

    from .data.dataset import CondensateDataset
    from torch.utils.data import DataLoader

    val_dataset = CondensateDataset(val_manifest, split="val", config=config)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    inferencer = CondensateInference(
        model_path=args.checkpoint,
        device=args.device,
        adaptive_threshold=True
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_metrics = inferencer.test_dataloader(
        val_loader,
        num_visualizations=args.num_viz,
        save_dir=output_dir
    )

    if len(df_metrics) > 0:
        df_metrics.to_csv(output_dir / "metrics.csv", index=False)
        print(f"\nResults saved to {output_dir}/")
