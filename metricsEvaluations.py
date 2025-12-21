# metricsEvaluations.py
from typing import Dict, Any
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Core Metric Functions

def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    
    valid_mask = (target >= 0) & (target < num_classes)
    indices = num_classes * target[valid_mask].to(torch.int64) + pred[valid_mask]
    cm = torch.bincount(indices, minlength=num_classes ** 2)
    cm = cm.reshape(num_classes, num_classes).float()
    return cm


def compute_derived_metrics(cm: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Calculates IoU, Dice, Precision, and Recall from a confusion matrix.
    Returns per-class metrics and their means.
    """
    eps = 1e-6
    tp = torch.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp

    intersection = tp
    union = cm.sum(1) + cm.sum(0) - intersection

    iou_per_class = intersection / (union + eps)
    dice_per_class = (2 * intersection) / (cm.sum(1) + cm.sum(0) + eps)
    precision_per_class = tp / (tp + fp + eps)
    recall_per_class = tp / (tp + fn + eps)

    return {
        "iou_per_class": iou_per_class,
        "dice_per_class": dice_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,

        "miou": iou_per_class.mean(),
        "dice": dice_per_class.mean(),
        "Precision": precision_per_class.mean(),
        "Recall": recall_per_class.mean(),
    }


def compute_segmentation_metrics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, Any]:
    """
    Evaluates segmentation metrics for a model on a dataloader.
    Computes mIoU, Dice, Precision, Recall, and pixel accuracy.
    """
    model.eval()
    total_confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating metrics", leave=False)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)  # [B, H, W]

            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)  # [B, H, W]

            # Update confusion matrix
            for i in range(preds.size(0)):
                cm = compute_confusion_matrix(preds[i], masks[i], num_classes)
                total_confusion += cm

            # Update pixel accuracy
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()

    # Compute final metrics
    metrics = compute_derived_metrics(total_confusion)
    pix_acc = total_correct / (total_pixels + 1e-6)

    results = {
        "mIoU": float(metrics["miou"]),
        "Dice": float(metrics["dice"]),
        "Precision": float(metrics["Precision"]),
        "Recall": float(metrics["Recall"]),
        "pixel_accuracy": float(pix_acc),

        "IoU_per_class": metrics["iou_per_class"].detach().cpu().tolist(),
        "Dice_per_class": metrics["dice_per_class"].detach().cpu().tolist(),
        "Precision_per_class": metrics["precision_per_class"].detach().cpu().tolist(),
        "Recall_per_class": metrics["recall_per_class"].detach().cpu().tolist(),
    }

    return results
