# tune.py
"""
Hyperparameter tuning for Physics-Aware Attention UNet.
MODIFIED: Saves plots of all metrics for *every trial*
into the 'results_tuning' folder.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from datasets import get_fish_data_loaders
from model import PhysicsAwareAttentionUNet
from train import train_model 

def save_plots(history: dict, save_dir: str):
    """Saves plots of training history to a directory."""
    
    plt.ioff()
    epochs = range(1, len(history['train_loss']) + 1)

    # --- Plot 1: Loss ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_loss.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history.get('val_miou', []), 'g-o', label='Validation mIoU')
    plt.plot(epochs, history.get('val_mdice', []), 'm-o', label='Validation mDice')
    plt.title('Validation Segmentation Metrics (IoU & Dice)')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_metrics_iou_dice.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history.get('val_pix_acc', []), 'c-o', label='Validation Pixel Accuracy')
    plt.title('Validation Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_pixel_accuracy.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history.get('val_mprecision', []), 'b-o', label='Validation mPrecision')
    plt.plot(epochs, history.get('val_mrecall', []), 'r-o', label='Validation mRecall')
    plt.title('Validation Precision & Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_metrics_prec_recall.png')) 
    plt.close()
    
    print(f"Saved plots to {save_dir}")


def run_experiment(
    model: torch.nn.Module, # <-- Pass in model
    train_loader: DataLoader, # <-- Pass in loader
    val_loader: DataLoader,   # <-- Pass in loader
    lr: float,
    lambda_phys: float,
    lambda_smooth: float,
    num_epochs: int = 5,
    num_classes: int = 2,
    trial_save_dir: str = ".",
):
    """
    Run a single training experiment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\n=== Running experiment: {os.path.basename(trial_save_dir)} ===")

    # train_model now handles all plotting and CSV logging
    model, history, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        save_dir=trial_save_dir,
        lambda_seg=1.0,
        lambda_phys=lambda_phys,
        lambda_smooth=lambda_smooth,
        lambda_B_sup=0.02,
        lambda_T_sup=0.05,
        num_classes=num_classes,
    )

    return best_val_loss


def tune_hyperparameters():
    
    ROOT_DIR = r"./Fish Dataset"
    IMAGE_SIZE = (256, 256) 
    BATCH_SIZE = 4 
    NUM_WORKERS = 0 

    print("Loading and splitting Fish Dataset (once for all trials)...")
    train_loader, val_loader, test_loader = get_fish_data_loaders(
        root_dir=ROOT_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS 
    )
    print("DataLoaders ready.")

    learning_rates = [1e-3, 5e-4]
    lambda_physics = [0.3, 0.5]
    lambda_smoothness = [0.05, 0.1]
    NUM_EPOCHS_PER_RUN = 5
    NUM_CLASSES = 2

    best_config = None
    best_loss = float("inf")

    base_save_dir = "results_tuning"
    os.makedirs(base_save_dir, exist_ok=True)

    print("\nStarting Physics-Aware UNet Hyperparameter Tuning...\n")

    for lr in learning_rates:
        for l_phys in lambda_physics:
            for l_smooth in lambda_smoothness:
                
                trial_name = f"lr_{lr}_phys_{l_phys}_smooth_{l_smooth}"
                trial_save_dir = os.path.join(base_save_dir, trial_name)
                os.makedirs(trial_save_dir, exist_ok=True)
                
                try:
                    model = PhysicsAwareAttentionUNet(in_ch=3, out_ch=NUM_CLASSES)
                    
                    val_loss = run_experiment(
                        model=model, 
                        train_loader=train_loader, 
                        val_loader=val_loader,   
                        lr=lr, 
                        lambda_phys=l_phys, 
                        lambda_smooth=l_smooth,
                        num_epochs=NUM_EPOCHS_PER_RUN,
                        trial_save_dir=trial_save_dir
                    )
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_config = (lr, l_phys, l_smooth)
                    print(f"Finished Trial: {trial_name} | Best Val Loss={val_loss:.4f}")
                
                except Exception as e:
                    print(f"Trial FAILED: {trial_name} | Error: {e}")
                    # raise e 

    if best_config is None:
        raise RuntimeError("No valid experiment completed successfully. Check dataset or training setup.")

    print("\n" + "="*50)
    print("Tuning Complete! ")
    print("\nBest Configuration Found:")
    print(f"  Learning Rate (lr): {best_config[0]}")
    print(f"  Lambda Physics (λ_phys): {best_config[1]}")
    print(f"  Lambda Smooth (λ_smooth): {best_config[2]}")
    print(f"  Best Validation Loss: {best_loss:.4f}")
    print("="*50)
    
    return best_config


if __name__ == "__main__":
    best = tune_hyperparameters()
    print(f"\nHyperparameter tuning complete. Best config: {best}")