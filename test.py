"""
Standalone test script for the Physics-Aware Attention UNet
on the Fish Dataset.

ENHANCED VERSION:
- Adds '--save_mode both' to save both grid and per-image visualizations.
- Automatically creates timestamped folders for every test run.
- Saves metrics in both .txt and .csv formats.
- Computes and prints total and trainable model parameters.
"""

import os
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from datetime import datetime
import pandas as pd

# Import project-specific functions
from datasets import get_fish_data_loaders
from model import PhysicsAwareAttentionUNet
from metricsEvaluations import compute_segmentation_metrics
from visualization import save_visual_results, save_individual_comparison_image


@torch.inference_mode()
def test_fish_dataset(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_dir: str = "results",
    save_mode: str = "grid",
    num_classes: int = 2
):
    """
    Runs evaluation on the Fish test loader, computes metrics,
    and saves visualizations based on save_mode.
    """
    print("Testing on Fish Dataset...")
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Compute Full Dataset Metrics ---
    print("Computing metrics...")
    seg_metrics = compute_segmentation_metrics(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=num_classes
    )

    print("\n--- Test Results (Fish Dataset) ---")
    print(f"  {'mIoU':<16s}: {seg_metrics.get('mIoU', 0.0):.4f}")
    print(f"  {'Dice':<16s}: {seg_metrics.get('Dice', 0.0):.4f}")
    print(f"  {'BoundaryIoU':<16s}: {seg_metrics.get('BoundaryIoU', 0.0):.4f}")
    print(f"  {'Precision':<16s}: {seg_metrics.get('Precision', 0.0):.4f}")
    print(f"  {'Recall':<16s}: {seg_metrics.get('Recall', 0.0):.4f}")
    print(f"  {'pixel_accuracy':<16s}: {seg_metrics.get('pixel_accuracy', 0.0):.4f}")
    print("-----------------------------------")

    # --- 2. Save Metrics to File + CSV (timestamped folder) ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(save_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    txt_path = os.path.join(run_dir, "test_metrics_final.txt")
    csv_path = os.path.join(run_dir, "test_metrics_final.csv")

    # Write human-readable text file
    with open(txt_path, "w") as f:
        f.write("Test Results (Fish Dataset):\n")
        for k, v in seg_metrics.items():
            f.write(f"  {k}: {v}\n")

    # Write structured CSV
    try:
        df = pd.DataFrame(list(seg_metrics.items()), columns=["Metric", "Value"])
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error saving metrics CSV: {e}")

    print(f"\n✓ Test metrics saved to:\n  {txt_path}\n  {csv_path}")

    # --- 3. Save Visualizations ---
    print(f"Saving visualizations (mode: {save_mode}) to {run_dir}...")

    # Always generate grid visualization
    try:
        batch = next(iter(test_loader))
        images = batch["image"].to(device)
        true_masks = batch["mask"].to(device)
        model.eval()
        logits, t_pred, b_pred, j_pred = model(images)
        pred_masks = torch.argmax(logits, dim=1)

        save_visual_results(
            images, true_masks, pred_masks,
            run_dir, "test_run_visuals",
            max_images=batch["image"].size(0)
        )
        print("Saved grid visualization.")
    except Exception as e:
        print(f"Error during grid visualization: {e}")

    # Optionally also generate individual results if requested
    if save_mode in ["all_individual", "both"]:
        individual_save_dir = os.path.join(run_dir, "individual_results")
        os.makedirs(individual_save_dir, exist_ok=True)
        print(f"Saving individual images to: {individual_save_dir}")

        image_index = 0
        for batch in tqdm(test_loader, desc="Saving individual images"):
            images = batch["image"].to(device)
            true_masks = batch["mask"].to(device)

            model.eval()
            logits, t_pred, b_pred, j_pred = model(images)
            pred_masks = torch.argmax(logits, dim=1)

            for i in range(images.size(0)):
                save_path = os.path.join(individual_save_dir, f"test_image_{image_index:04d}.png")
                try:
                    save_individual_comparison_image(
                        image=images[i],
                        true_mask=true_masks[i],
                        pred_mask=pred_masks[i],
                        save_path=save_path,
                        num_classes=num_classes
                    )
                except Exception as e:
                    print(f"Error saving image {image_index}: {e}")
                image_index += 1
        print(f"Saved {image_index} individual comparison images.")

    return seg_metrics


# ============================================================
# RUN SCRIPT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Physics-Aware UNet on Fish Dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--dataset_root", type=str, default="./Fish Dataset",
                        help="Root directory of the Fish Dataset.")
    parser.add_argument("--save_dir", type=str, default="results_test",
                        help="Directory to save test metrics and visualizations.")
    parser.add_argument("--save_mode", type=str,
                        choices=["grid", "all_individual", "both"],
                        default="both",
                        help="grid: save one batch. all_individual: save all test images separately. both: save both.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size to use for testing.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of dataloader workers")
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 2
    IMAGE_SIZE = (256, 256)

    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Loading model from {args.checkpoint}")
    model = PhysicsAwareAttentionUNet(in_ch=3, out_ch=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        exit(1)
    except RuntimeError as e:
        print(f"Error loading model state: {e}")
        exit(1)

    model.to(device)
    model.eval()

    # --- Model Parameters ---
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n--- Model Parameters ---")
        print(f"  Total Parameters:     {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print("------------------------")
    except Exception as e:
        print(f"Could not calculate model parameters: {e}")

    # --- Load Data ---
    print("\nLoading test data...")
    _, _, test_loader = get_fish_data_loaders(
        root_dir=args.dataset_root,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # --- Run Test ---
    test_fish_dataset(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir,
        save_mode=args.save_mode,
        num_classes=NUM_CLASSES
    )

    print("\nTest complete. Results saved to", args.save_dir)
