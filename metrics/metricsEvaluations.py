import torch


def compute_confusion_matrix(pred, target, num_classes):
    mask = (target >= 0) & (target < num_classes)

    pred = pred[mask]
    target = target[mask]

    idx = num_classes * target + pred
    cm = torch.bincount(idx, minlength=num_classes**2)

    return cm.view(num_classes, num_classes).float()


def compute_metrics(cm):
    eps = 1e-6

    tp = torch.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp

    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    
    # Pixel accuracy: (TP + TN) / Total
    pixel_accuracy = tp.sum() / (cm.sum() + eps)

    return {
        "Pixel_Accuracy": pixel_accuracy.item(),
        "mIoU": iou.mean().item(),
        "Dice": dice.mean().item(),
        "Precision": precision.mean().item(),
        "Recall": recall.mean().item(),
    }


@torch.no_grad()
def compute_segmentation_metrics(model, loader, device, num_classes):
    model.eval()

    total_cm = torch.zeros((num_classes, num_classes), device=device)

    for batch in loader:
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)

        logits, _, _, _, _ = model(img)
        pred = torch.argmax(logits, dim=1)

        total_cm += compute_confusion_matrix(pred, mask, num_classes)

    return compute_metrics(total_cm)