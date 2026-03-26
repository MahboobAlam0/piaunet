import argparse
import os
import torch
import torch.optim as optim

# Disable CUDA expandable segments warning (not supported on Windows)
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

from dataset.datasets import get_data_loaders
from model.model import PhysicsInformedAttentionUNet
from train.train import train_model
from train.tune import tune_hyperparameters
from testing.test import test_model # type: ignore


def partial_load_checkpoint(model, checkpoint_state):
    """
    Load checkpoint with architecture mismatch handling.
    Load only compatible layers, skip new layers.
    """
    model_state = model.state_dict()
    loaded_keys = set()
    skipped_keys = set()
    
    for key, value in checkpoint_state.items():
        if key in model_state:
            # Check shape compatibility
            if model_state[key].shape == value.shape:
                model_state[key] = value
                loaded_keys.add(key)
            else:
                skipped_keys.add(f"{key} (shape mismatch)")
        else:
            skipped_keys.add(f"{key} (not in new model)")
    
    model.load_state_dict(model_state)
    
    print(f"Loaded {len(loaded_keys)} compatible layers")
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} incompatible layers")
    
    return model


def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    _, test_loader = get_data_loaders(
        root_dir=args.dataset_root,
        image_size=(256, 256),
        batch_size=args.batch_size,
    )
    
    # Load model
    model = PhysicsInformedAttentionUNet(3, args.num_classes).to(device)
    
    if not args.checkpoint:
        print("Please provide --checkpoint for testing")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint {args.checkpoint} not found")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Run test
    test_model(model, test_loader, device, save_dir=args.save_dir, num_classes=args.num_classes)
    print(f"Test results saved to {args.save_dir}/")


def run_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_data_loaders(
        root_dir=args.dataset_root,
        image_size=(256, 256),
        batch_size=args.batch_size,
    )

    model = PhysicsInformedAttentionUNet(3, args.num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            
            # Handle both old and new checkpoint formats
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                # New format with full checkpoint
                try:
                    model.load_state_dict(checkpoint["model_state"])
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                    scheduler.load_state_dict(checkpoint["scheduler_state"])
                except RuntimeError:
                    # Architecture mismatch - do partial loading
                    print("Architecture changed. Doing partial checkpoint loading...")
                    partial_load_checkpoint(model, checkpoint["model_state"])
                    # Reset optimizer and scheduler (they may not be compatible)
                    print("ℹOptimizer and scheduler reset (architecture changed)")
                
                start_epoch = checkpoint.get("epoch", 0)
            else:
                # Old format - checkpoint is just model weights
                try:
                    model.load_state_dict(checkpoint)
                except RuntimeError:
                    # Architecture mismatch - do partial loading
                    print("Architecture changed. Doing partial checkpoint loading...")
                    partial_load_checkpoint(model, checkpoint)
                    print("Optimizer and scheduler initialized fresh")
                start_epoch = 0
            
            print(f"Loaded checkpoint from {args.checkpoint} at epoch {start_epoch}")
        else:
            print(f"Checkpoint {args.checkpoint} not found. Starting from scratch.")

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        save_dir=args.save_dir,
        start_epoch=start_epoch,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test", "tune"], default="train")
    parser.add_argument("--dataset", default="aqua")
    parser.add_argument("--dataset_root", default="./AquaOV255")
    parser.add_argument("--save_dir", default="results")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for resuming training")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)

    parser.add_argument("--lambda_smooth", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=2)

    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    elif args.mode == "tune":
        tune_hyperparameters(args.dataset, args.dataset_root)


if __name__ == "__main__":
    main()