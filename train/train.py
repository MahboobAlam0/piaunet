import os
import csv
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast_mode, grad_scaler

from lossfunction.lossFunction import physics_aware_loss
from metrics.metricsEvaluations import compute_segmentation_metrics
from visualization.visualization import save_visual_results, save_enhanced_image


# SMART CURRICULUM 
def get_loss_weights(epoch):
    if epoch < 5:
        return dict(lambda_phys=0.0)
    elif epoch < 15:
        return dict(lambda_phys=0.01)
    else:
        return dict(lambda_phys=0.03)


# TRAIN ONE EPOCH

def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc=f"Train {epoch + 1}"):
        img = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_mode.autocast(device_type="cuda"):
            seg_main, seg_aux, j, t, b = model(img)

            weights = get_loss_weights(epoch)

            loss, _ = physics_aware_loss(
                seg_main,
                seg_aux,
                mask,
                j,
                t,
                b,
                img,
                lambda_seg=1.0,
                lambda_aux=0.3,
                lambda_smooth=0.01,
                lambda_weak_t=0.01,
                lambda_weak_b=0.01,
                **weights
            )

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)



# FAST VALIDATION

@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    # lightweight validation 
    total_iou = 0
    count = 0

    for batch in loader:
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)

        logits, _, _, _, _ = model(img)

       
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).long()

        intersection = (preds & mask).sum().item()
        union = (preds | mask).sum().item()

        if union > 0:
            total_iou += intersection / union
            count += 1

    return total_iou / max(count, 1)


# MAIN TRAIN

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs,
    device,
    save_dir,
    start_epoch=0,
):

    os.makedirs(save_dir, exist_ok=True)

    scaler = grad_scaler.GradScaler("cuda")
    best_miou = 0.0
    patience = 15
    patience_counter = 0

    # Initialize CSV logging 
    csv_path = os.path.join(save_dir, "metrics.csv")
    csv_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, "a" if csv_exists else "w", newline="")
    csv_writer = csv.writer(csv_file)
    
    if not csv_exists:
        csv_writer.writerow(["Epoch", "Loss", "Pixel_Accuracy", "mIoU", "Dice", "Precision", "Recall"])
    csv_file.flush()

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch
        )

        # Compute full metrics
        metrics = compute_segmentation_metrics(model, val_loader, device, num_classes=2)

        pixel_acc = metrics["Pixel_Accuracy"]
        miou = metrics["mIoU"]
        dice = metrics["Dice"]
        precision = metrics["Precision"]
        recall = metrics["Recall"]

        print(f"Loss: {train_loss:.4f} | Pixel_Acc: {pixel_acc:.4f} | mIoU: {miou:.4f} | Dice: {dice:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        # Log to CSV
        csv_writer.writerow([epoch+1, f"{train_loss:.4f}", f"{pixel_acc:.4f}", f"{miou:.4f}", f"{dice:.4f}", f"{precision:.4f}", f"{recall:.4f}"])
        csv_file.flush()

        # visualize occasionally 
        if epoch % 3 == 0:
            batch = next(iter(val_loader))
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits, seg_aux, j, t, b = model(imgs)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).long()
            
            # Reconstruct enhanced image from physics branch
            # Upsample t and b to match j resolution (j is 256x256, t/b are 32x32)
            t_up = F.interpolate(t, size=j.shape[-2:], mode='bilinear', align_corners=False)
            b_up = F.interpolate(b, size=j.shape[-2:], mode='bilinear', align_corners=False)
            enhanced = j * t_up + b_up * (1 - t_up)

            # Save both visualizations
            save_visual_results(imgs, masks, preds, save_dir, f"epoch_{epoch+1}")
            save_enhanced_image(imgs, enhanced, save_dir, f"epoch_{epoch+1}_enhanced")

        # Early stopping based on mIoU
        if miou > best_miou:
            best_miou = miou
            patience_counter = 0
            # Save full checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_miou": best_miou,
            }
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
            print(f"Saved best model (mIoU: {best_miou:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        scheduler.step()

        if patience_counter >= patience:
            print(f"Early stopping triggered. Best mIoU: {best_miou:.4f}")
            break

    csv_file.close()
    return model, best_miou