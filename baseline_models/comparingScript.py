# comparingScript.py

import torch
from torch.utils.data import DataLoader
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- IMPORTS from your project's files ---
from dataset.datasets import get_data_loaders  # type: ignore
# ------------------------------------------------
from baseline_models.comparisonModels import (
    UNet, AttentionUNet, DeepLabV3Plus, 
    train_one_model, compute_segmentation_metrics_baseline,
    set_seed, safe_resize_tensor, METRIC_KEYS
)
import torch.nn.functional as F
import os
from typing import Dict, Any, cast
import traceback
from torch.amp.autocast_mode import autocast
from torchvision.utils import save_image



def get_three_loaders(root_dir: str, image_size=(256, 256), batch_size=4, num_workers=0):
    """
    Get train, validation, and test loaders from AquaOV255 dataset.
    Since get_data_loaders returns train/val, we use val as test.
    """
    train_loader, val_loader = get_data_loaders(
        root_dir=root_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers
    )
    # For now, use val_loader as test_loader
    return train_loader, val_loader, val_loader


# CONFIGURATION

# Choose dataset: "AQUA" or "FISH"
DATASET = "AQUA"  # Change to "FISH" to use Fish Dataset

if DATASET == "AQUA":
    DATASET_ROOT = "./AquaOV255"
    BATCH_SIZE = 4  # Smaller batch size for AQUA
    EPOCHS = 50
    VISUALIZE_DIR = "./test_visuals_baselines_aqua_1"
else:
    DATASET_ROOT = "./Fish Dataset"
    BATCH_SIZE = 8
    EPOCHS = 2
    VISUALIZE_DIR = "./test_visuals_baselines_2"

LEARNING_RATE = 1e-4
NUM_WORKERS = 0
IMAGE_SIZE = (256, 256)
NUM_CLASSES = 2 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

set_seed(42)


# VISUALIZATION HELPERS

def denormalize_image(tensor: torch.Tensor, 
                      mean=(0.485, 0.456, 0.406), 
                      std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    """Denormalizes a tensor image."""
    mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp((tensor * std) + mean, 0, 1)

def mask_to_rgb(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Converts a [B, H, W] integer mask to a [B, 3, H, W] B&W image.
    """
    b, h, w = mask.shape
    
    if num_classes == 2: # Specific for Fish Dataset
        norm_mask = (mask == 1).float() # 0.0 for BG, 1.0 for Fish
    else:
        norm_mask = (mask.float() / (num_classes - 1)) if num_classes > 1 else mask.float()
        
    color_mask = norm_mask.unsqueeze(1).repeat(1, 3, 1, 1) # [B, 3, H, W]
    return color_mask


# MODEL EVALUATION 

def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, 
                   visualize_dir: str, model_name: str, 
                   num_classes: int) -> Dict[str, Any]: # <-- Return Any for lists
    """
    Evaluate model, compute metrics using compute_segmentation_metrics,
    and save 3-column visualizations.
    """
    model.eval()
    
    # --- 1. Compute Full Dataset Metrics ---
    print(f"Computing full test metrics for {model_name}...")
    avg_metrics = compute_segmentation_metrics_baseline(
        model, dataloader, device, num_classes
    )
    
    # --- 2. Save Visualizations ---
    print(f"Saving visualizations for {model_name}...")
    os.makedirs(visualize_dir, exist_ok=True)
    
    try:
        batch = next(iter(dataloader))
        imgs = batch["image"].to(device)
        true_masks = batch["mask"].to(device) # [B, H, W] Long

        imgs_resized = safe_resize_tensor(imgs, size=IMAGE_SIZE, is_mask=False)
        true_masks_resized = safe_resize_tensor(true_masks, size=IMAGE_SIZE, is_mask=True)

        with torch.no_grad():
            with autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(imgs_resized)
        
        if isinstance(outputs, dict) and "out" in outputs: # DeepLab
            logits = outputs["out"]
        elif isinstance(outputs, (tuple, list)): # Physics-Aware
            logits = outputs[0]
        else: # UNet, AttnUNet
            logits = cast(torch.Tensor, outputs)

        if logits.shape[-2:] != IMAGE_SIZE:
            logits = F.interpolate(logits, size=IMAGE_SIZE, mode='bilinear', align_corners=False)

        preds = torch.argmax(logits, dim=1) # [B, H, W] Long

        images_denorm = denormalize_image(imgs_resized).cpu()
        true_masks_rgb = mask_to_rgb(true_masks_resized.cpu(), num_classes)
        pred_masks_rgb = mask_to_rgb(preds.cpu(), num_classes)

        grid = torch.cat([
            images_denorm, 
            true_masks_rgb, 
            pred_masks_rgb
        ], dim=0)
        
        save_path = os.path.join(visualize_dir, f"{model_name}_comparison_grid.png")
        save_image(grid, save_path, nrow=imgs.size(0)) 
        print(f"Saved 3-row visual comparison grid to {save_path}")

    except Exception as e:
        print(f"[WARN] Failed to save visualization for {model_name}: {e}")
        traceback.print_exc()

    return avg_metrics


# MAIN COMPARISON LOGIC 
def main():
    if DEVICE.type == 'cuda':
        print(f" Using device: {DEVICE.type.upper()} ({torch.cuda.get_device_name(0)})")
    else:
        print(f" Using device: {DEVICE.type.upper()} (CPU)")

    os.makedirs(VISUALIZE_DIR, exist_ok=True)
    print(f"[INFO] Test visualizations will be saved to: {VISUALIZE_DIR}")

    # --- Load Dataset (AquaOV255 or Fish) ---
    print(f"\n[INFO] Loading {DATASET} dataset from: {DATASET_ROOT}")
    train_loader, val_loader, test_loader = get_three_loaders(
        root_dir=DATASET_ROOT,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    print("[INFO] Dataset loaded successfully!")

    # --- Instantiate models ---
    models_dict = {
        "U-Net": UNet(in_ch=3, out_ch=NUM_CLASSES),
        "AttentionUNet": AttentionUNet(in_ch=3, out_ch=NUM_CLASSES),
        "DeepLabV3Plus": DeepLabV3Plus(pretrained=True, out_ch=NUM_CLASSES)
    }

    os.makedirs("./checkpoints_baselines_1", exist_ok=True)
    results_test: Dict[str, Dict[str, Any]] = {}
    results_best_epoch: Dict[str, Any] = {}

    for model_name, model in models_dict.items():
        print(f"\n--- Training {model_name} ---")

        # --- NEW: Compute Model Parameters ---
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("\n--- Model Parameters ---")
            print(f"  Total Parameters:     {total_params:,}")
            print(f"  Trainable Parameters: {trainable_params:,}")
            print("------------------------")
        except Exception as e:
            print(f"Could not calculate model parameters: {e}")
        # -----------------------------------

        try:
            trained_model, metrics_history, best_epoch_idx = train_one_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=DEVICE,
                num_classes=NUM_CLASSES, 
                num_epochs=EPOCHS,
                lr=LEARNING_RATE,
                weight_decay=1e-4,
                grad_clip=1.0,
                save_dir=f"./checkpoints_baselines_2/{model_name}",
                early_stop_patience=15, 
                image_size=IMAGE_SIZE
            )

            save_path = f"./checkpoints_baselines_2/{model_name}/{model_name}_final.pth"
            torch.save(trained_model.state_dict(), save_path)
            print(f" Saved trained {model_name} at {save_path}")

            model_visualize_dir = os.path.join(VISUALIZE_DIR, model_name)
            os.makedirs(model_visualize_dir, exist_ok=True)

            print(f"\n Evaluating {model_name} on Test Set...")
            test_metrics = evaluate_model(
                model=trained_model, 
                dataloader=test_loader, 
                device=DEVICE,
                visualize_dir=model_visualize_dir,
                model_name=model_name,
                num_classes=NUM_CLASSES
            )
            
            results_test[model_name] = test_metrics
            results_best_epoch[model_name] = {"best_epoch": best_epoch_idx, "history": metrics_history}

        except Exception as e:
            print(f" Error training {model_name}: {e}")
            traceback.print_exc()

    # --- Print Final Comparison ---
    print("\n==============================================")
    print("           Final Model Comparison")
    print("==============================================")
    
    metric_order = METRIC_KEYS 
    
    for model_name, test_metrics in results_test.items():
        print(f"\n--- Model: {model_name} ---")
        
        # --- Print Test Metrics ---
        print(" \nTest Metrics:")
        test_keys = [k for k in metric_order if k in test_metrics] + \
                    [k for k in test_metrics if k not in metric_order]
        for k in test_keys:
            if k in test_metrics:
                if isinstance(test_metrics[k], (float, int)):
                    print(f"    {k:16s}: {test_metrics[k]:.4f}")
                else:
                    print(f"    {k:16s}: (see per-class scores)")

        # --- Print Train/Val Metrics from Best Epoch ---
        if model_name in results_best_epoch:
            epoch_data = results_best_epoch[model_name]
            best_epoch = epoch_data["best_epoch"]
            history = epoch_data["history"]
            
            if best_epoch != -1 and history:
                print(f"\nMetrics from Best Validation Epoch (Epoch {best_epoch + 1}):")
                
                print("\n  Validation:")
                for k in metric_order:
                    if k in history["val"] and len(history["val"]) > 0 and len(history["val"][k]) > best_epoch:
                        print(f"      {k:16s}: {history['val'][k][best_epoch]:.4f}")
                        
                print("\n  Training (from same epoch):")
                for k in metric_order:
                    if k in history["train"] and len(history["train"]) > 0 and len(history["train"][k]) > best_epoch:
                        print(f"      {k:16s}: {history['train'][k][best_epoch]:.4f}")
            else:
                print("\nTraining/Validation metrics not available (training may have failed).")
        print("----------------------------------------------")


# RUN SCRIPT

if __name__ == "__main__":
    main()