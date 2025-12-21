# main.py
import argparse
import torch
import os
import torch.optim as optim

from model import PhysicsAwareAttentionUNet
from datasets import get_fish_data_loaders
from train import train_model
from tune import tune_hyperparameters
from metricsEvaluations import compute_segmentation_metrics

def run_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loading  ---
    print("Loading Fish Dataset...")
    train_loader, val_loader, _ = get_fish_data_loaders(
        root_dir=args.dataset_root,
        image_size=(256, 256), 
        batch_size=args.batch_size,
        num_workers=0
    )
    print("Fish DataLoaders ready.")

    # --- Model Setup ---
    print(f"Initializing model with {args.num_classes} classes.")
    model = PhysicsAwareAttentionUNet(in_ch=3, out_ch=args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # type: ignore

    # --- Training ---
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir,
        lambda_seg=args.lambda_seg,
        lambda_phys=args.lambda_phys,
        lambda_smooth=args.lambda_smooth,
        lambda_B_sup=args.lambda_B_sup,
        lambda_T_sup=args.lambda_T_sup,
        num_classes=args.num_classes,
    )

def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {args.checkpoint} for Fish dataset.")
    model = PhysicsAwareAttentionUNet(in_ch=3, out_ch=args.num_classes)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print("Testing on Fish Dataset...")
    _, _, test_loader = get_fish_data_loaders(
        root_dir=args.dataset_root,
        image_size=(256, 256), # Or make this an arg
        batch_size=args.batch_size,
        num_workers=0
        )
    
    print("Computing metrics...")
    seg_metrics = compute_segmentation_metrics(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=args.num_classes
    )
    
    print("\nTest Results (Fish Dataset):")
    for k, v in seg_metrics.items():
        if isinstance(v, list):
            print(f"  {k}: {v}")
        else:
            print(f"  {k:15s}: {v:.4f}")
    
    results_path = os.path.join(args.save_dir, "test_metrics.txt")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        f.write("Test Results (Fish Dataset):\n")
        for k, v in seg_metrics.items():
            f.write(f"  {k}: {v}\n")
    print(f"\nTest metrics saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "tune"], default="train",
                        help="Action to perform.")
    parser.add_argument("--dataset_root", default="./Fish Dataset",
                        help="Root directory for the Fish dataset.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth",
                        help="Path to model checkpoint for testing.")
    parser.add_argument("--save_dir", default="results",
                        help="Directory to save test results, plots, and visuals.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    
    # Loss weights
    parser.add_argument("--lambda_seg", type=float, default=1.0)
    parser.add_argument("--lambda_phys", type=float, default=0.3)
    parser.add_argument("--lambda_smooth", type=float, default=0.05)
    parser.add_argument("--lambda_B_sup", type=float, default=0.02)
    parser.add_argument("--lambda_T_sup", type=float, default=0.05)
    
    args = parser.parse_args()

    args.num_classes = 2 # Always 2 for Fish Dataset
    
    if args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    elif args.mode == "tune":
        print("Running hyperparameter tuning (configured in tune.py for Fish Dataset)...")
        # Note: tune.py might have its own num_workers setting.
        # This is the *real* fix (see tune.py update)
        tune_hyperparameters()

