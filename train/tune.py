import os
import torch
import torch.optim as optim
import pandas as pd

from dataset.datasets import get_data_loaders
from model.model import PhysicsInformedAttentionUNet
from train.train import train_model


def run_experiment(train_loader, val_loader, lr, smooth, epochs, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PhysicsInformedAttentionUNet(3, 2).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )

    _, _, best_miou = train_model( # type: ignore
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        save_dir=save_dir,
    )

    return best_miou


def tune_hyperparameters(dataset_name="aqua", root_dir="./AquaOV255"):

    train_loader, val_loader = get_data_loaders(
        root_dir=root_dir,
        image_size=(256, 256),
        batch_size=4,
    )

    learning_rates = [3e-4, 1e-4]
    smooth_values = [0.01, 0.02]

    results = []
    best_score = -1
    best_config = None

    for lr in learning_rates:
        for smooth in smooth_values:

            trial_dir = f"results_tuning/lr_{lr}_smooth_{smooth}"
            os.makedirs(trial_dir, exist_ok=True)

            print(f"\nRunning lr={lr}, smooth={smooth}")

            try:
                score = run_experiment(
                    train_loader,
                    val_loader,
                    lr,
                    smooth,
                    epochs=15,
                    save_dir=trial_dir
                )

                results.append({
                    "lr": lr,
                    "smooth": smooth,
                    "miou": score
                })

                if score > best_score:
                    best_score = score
                    best_config = (lr, smooth)

            except Exception as e:
                print("FAILED:", e)

    pd.DataFrame(results).to_csv("results_tuning/results.csv", index=False)

    print("\nBest config:", best_config, "mIoU:", best_score)

    return best_config