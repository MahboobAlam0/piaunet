# train.py

from __future__ import annotations
from typing import Dict, Tuple, Optional, Any, List
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lossFunction import physics_aware_loss
from metricsEvaluations import compute_segmentation_metrics
from visualization import save_metric_plots, save_visual_results, append_metrics_to_csv

# Training Loop
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    lambda_seg: float,
    lambda_phys: float,
    lambda_smooth: float,
    lambda_B_sup: float,
    lambda_T_sup: float,
    num_classes: int,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    running_metrics: Dict[str, float] = {}

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad()
        seg_logits, turbidity, backscatter, j_pred = model(images)
        loss, loss_dict = physics_aware_loss(
            seg_logits=seg_logits, seg_target=masks, j_pred=j_pred,
            t_pred=turbidity, b_pred=backscatter, images=images,
            lambda_seg=lambda_seg, lambda_phys=lambda_phys, lambda_smooth=lambda_smooth,
            lambda_B_sup=lambda_B_sup, lambda_T_sup=lambda_T_sup, num_classes=num_classes,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        for k, v in loss_dict.items():
            running_metrics[k] = running_metrics.get(k, 0.0) + v
    n_batches = len(loader)
    avg_metrics = {k: v / n_batches for k, v in running_metrics.items()}
    avg_loss = total_loss / n_batches
    return avg_loss, avg_metrics

# Validation Loop

@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    lambda_seg: float,
    lambda_phys: float,
    lambda_smooth: float,
    lambda_B_sup: float,
    lambda_T_sup: float,
    num_classes: int,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    running_metrics: Dict[str, float] = {}
    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        seg_logits, turbidity, backscatter, j_pred = model(images)
        loss, loss_dict = physics_aware_loss(
            seg_logits=seg_logits, seg_target=masks, j_pred=j_pred,
            t_pred=turbidity, b_pred=backscatter, images=images,
            lambda_seg=lambda_seg, lambda_phys=lambda_phys, lambda_smooth=lambda_smooth,
            lambda_B_sup=lambda_B_sup, lambda_T_sup=lambda_T_sup, num_classes=num_classes,
        )
        total_loss += loss.item()
        for k, v in loss_dict.items():
            running_metrics[k] = running_metrics.get(k, 0.0) + v
    n_batches = len(loader)
    avg_metrics = {k: v / n_batches for k, v in running_metrics.items()}
    avg_loss = total_loss / n_batches
    return avg_loss, avg_metrics

# Full Training Routine

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    num_epochs: int,
    device: torch.device,
    save_dir: str, 
    *,
    lambda_seg: float,
    lambda_phys: float,
    lambda_smooth: float,
    lambda_B_sup: float,
    lambda_T_sup: float,
    num_classes: int,
) -> Tuple[torch.nn.Module, Dict[str, List[float]], float]:
    
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True) 
    patience = 15
    epochs_no_improve = 0

    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []
    val_miou_hist: List[float] = []
    val_mdice_hist: List[float] = []
    val_pix_acc_hist: List[float] = []
    val_mprecision_hist: List[float] = []
    val_mrecall_hist: List[float] = []

    final_train_losses = {}
    final_val_losses = {}

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        # ---- Training ----
        train_loss, train_losses = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, device=device,
            lambda_seg=lambda_seg, lambda_phys=lambda_phys, lambda_smooth=lambda_smooth,
            lambda_B_sup=lambda_B_sup, lambda_T_sup=lambda_T_sup, num_classes=num_classes
        )
        train_loss_hist.append(train_loss)
        final_train_losses = train_losses

        # ---- Validation ----
        val_loss, val_losses = validate_one_epoch(
            model=model, loader=val_loader, device=device,
            lambda_seg=lambda_seg, lambda_phys=lambda_phys, lambda_smooth=lambda_smooth,
            lambda_B_sup=lambda_B_sup, lambda_T_sup=lambda_T_sup, num_classes=num_classes
        )
        val_loss_hist.append(val_loss)
        final_val_losses = val_losses

        # ---- Full Metrics Calculation ----
        print("Calculating validation metrics for epoch...")
        val_metrics = compute_segmentation_metrics(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=num_classes
        )
        val_miou_hist.append(val_metrics['mIoU'])
        val_mdice_hist.append(val_metrics['Dice'])
        val_pix_acc_hist.append(val_metrics['pixel_accuracy'])
        val_mprecision_hist.append(val_metrics['Precision'])
        val_mrecall_hist.append(val_metrics['Recall'])

        # ---- Scheduler ----
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val mIoU: {val_metrics['mIoU']:.4f} | Val Dice: {val_metrics['Dice']:.4f} | Val PixAcc: {val_metrics['pixel_accuracy']:.4f}")
        print(f"  Val Precision: {val_metrics['Precision']:.4f} | Val Recall: {val_metrics['Recall']:.4f}")
        
        print("  Losses (Train):", train_losses)
        print("  Losses (Val):  ", val_losses)
        
        # --- Early Stopping Logic ---
        if val_loss < best_val_loss:
            print(f"Val loss improved ({best_val_loss:.4f} -> {val_loss:.4f}). Saving model...")
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            epochs_no_improve = 0 
        else:
            print(f"Val loss did not improve from {best_val_loss:.4f}")
            epochs_no_improve += 1 
            print(f"  Patience: {epochs_no_improve}/{patience}")

        current_history = {
            "train_loss": train_loss_hist,
            "val_loss": val_loss_hist,
            "val_miou": val_miou_hist,
            "val_dice": val_mdice_hist,
            "val_pix_acc": val_pix_acc_hist,
            "val_precision": val_mprecision_hist,
            "val_recall": val_mrecall_hist,
        }
        try:
            save_metric_plots(current_history, save_dir)
            append_metrics_to_csv(current_history, save_dir)
            print(f"Epoch {epoch+1} plots and CSV log saved to {save_dir}")
        except Exception as e:
            print(f"Error saving epoch plots/CSV: {e}")
            
        # --- Check for early stopping ---
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping! No improvement in validation loss for {patience} epochs.")
            break 

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")

    history = {
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
        "val_miou": val_miou_hist,
        "val_dice": val_mdice_hist,
        "val_pix_acc": val_pix_acc_hist,
        "val_precision": val_mprecision_hist,
        "val_recall": val_mrecall_hist,
        "last_train_loss_breakdown": final_train_losses,
        "last_val_loss_breakdown": final_val_losses,
    }
    
    print(f"Saving final visualization grid to {save_dir}...")
    try:
        batch = next(iter(val_loader))
        images = batch["image"].to(device)
        true_masks = batch["mask"].to(device)
        
        model.eval() 
        with torch.no_grad():
            logits, t_pred, b_pred, j_pred = model(images)
        pred_masks = torch.argmax(logits, dim=1)

        save_visual_results(
            images, true_masks, pred_masks, 
            save_dir, "final_train_visuals"
        )
        
    except Exception as e:
        print(f"Error during final visualization saving: {e}")
    
    return model, history, best_val_loss