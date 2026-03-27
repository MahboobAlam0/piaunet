# comparisonModels.py
import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from tqdm import tqdm
import torch.optim as optim
import copy
import random
import numpy as np
from typing import Dict, List


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




# loss functions 

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice loss for multiclass segmentation.
    pred_logits: [B, C, H, W] logits
    target: [B, H, W] long tensor with class indices
    """
    pred = F.softmax(pred_logits, dim=1)
    
    # Convert target to one-hot [B, C, H, W]
    num_classes = pred_logits.shape[1]
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    intersection = (pred * target_one_hot).sum(dim=(2, 3))  # [B, C]
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # [B, C]
    
    dice = 1.0 - (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def compute_batch_class_weights(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute class weights based on inverse frequency in batch.
    target: [B, H, W] long tensor
    num_classes: number of classes
    """
    device = target.device
    class_counts = torch.bincount(target.reshape(-1), minlength=num_classes)
    total_pixels = target.numel()
    
    # Inverse frequency weighting
    class_weights = total_pixels / (num_classes * (class_counts.float() + 1.0))
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
    
    return class_weights.to(device)


def compute_segmentation_metrics_baseline(model: nn.Module, loader: DataLoader, 
                                          device: torch.device, num_classes: int) -> Dict:
    """
    Compute segmentation metrics for baseline models (outputs single tensor, not tuple).
    """
    model.eval()
    
    total_tp = torch.zeros(num_classes, device=device)
    total_fp = torch.zeros(num_classes, device=device)
    total_fn = torch.zeros(num_classes, device=device)
    total_pixels = 0
    correct_pixels = 0
    
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            
            # Get predictions from baseline model (single tensor output)
            logits = model(img)
            pred = torch.argmax(logits, dim=1).long()
            
            # Flatten all predictions and targets
            pred_flat = pred.reshape(-1)
            mask_flat = mask.reshape(-1)
            
            # Compute per-class metrics
            for c in range(num_classes):
                tp = ((pred_flat == c) & (mask_flat == c)).sum().float()
                fp = ((pred_flat == c) & (mask_flat != c)).sum().float()
                fn = ((pred_flat != c) & (mask_flat == c)).sum().float()
                
                total_tp[c] += tp
                total_fp[c] += fp
                total_fn[c] += fn
            
            # Pixel accuracy
            correct_pixels += (pred_flat == mask_flat).sum().item()
            total_pixels += mask_flat.numel()
    
    eps = 1e-6
    
    # Compute metrics
    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    
    return {
        "mIoU": iou.mean().item(),
        "Dice": dice.mean().item(),
        "Precision": precision.mean().item(),
        "Recall": recall.mean().item(),
        "pixel_accuracy": pixel_accuracy,
        "Loss": 0.0,  # Placeholder
    }

# Metric keys for logging
METRIC_KEYS = ["Loss", "mIoU", "Dice", "Precision", "Recall", "pixel_accuracy"]


# Helper: deterministic utilities

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


#   Helper: Safe Resize 

def safe_resize_tensor(tensor: torch.Tensor, size=(256, 256), is_mask=False) -> torch.Tensor:
    """
    Safely resize tensor to target size.
    - If is_mask=True, use nearest neighbor interpolation to preserve labels.
    - Accepts input shapes (B,C,H,W) or (B,H,W).
    """
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeezed = True
    else:
        squeezed = False

    # (for multiclass, it's usually Long, but interpolate needs float/int)
    if is_mask:
        if not tensor.is_floating_point():
            tensor = tensor.float()
        mode = "nearest"
        align_corners_mode = None
    else:
        mode = "bilinear"
        align_corners_mode = False


    out = F.interpolate(tensor, size=size, mode=mode, align_corners=align_corners_mode)

    # If mask, convert back to Long
    if is_mask:
        out = out.long()

    if squeezed:
        out = out.squeeze(0)
    return out

#   Model Definitions 

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """Baseline U-Net MODIFIED for num_classes=2 output"""
    def __init__(self, in_ch=3, out_ch=2): # <-- MODIFIED
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, out_ch, kernel_size=1) # <-- Uses out_ch

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = F.interpolate(x, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv1(torch.cat([x, x4], dim=1))
        
        x = self.up2(x)
        x = F.interpolate(x, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv2(torch.cat([x, x3], dim=1))
        
        x = self.up3(x)
        x = F.interpolate(x, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv3(torch.cat([x, x2], dim=1))
        
        x = self.up4(x)
        x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv4(torch.cat([x, x1], dim=1))
        return self.outc(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AttentionBlock(nn.Module):
    """Attention Gate"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # This implementation assumes g and x are already the same size
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """U-Net with Attention Gates MODIFIED for num_classes=2 output"""
    def __init__(self, in_ch=3, out_ch=2): # <-- MODIFIED
        super(AttentionUNet, self).__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(512, 512, 256)
        self.conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(256, 256, 128)
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(128, 128, 64)
        self.conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(64, 64, 32)
        self.conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_ch, kernel_size=1) # <-- Uses out_ch

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = F.interpolate(x, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4_att = self.att1(g=x, x=x4)
        x = self.conv1(torch.cat([x, x4_att], dim=1))
        
        x = self.up2(x)
        x = F.interpolate(x, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        x3_att = self.att2(g=x, x=x3)
        x = self.conv2(torch.cat([x, x3_att], dim=1))
        
        x = self.up3(x)
        x = F.interpolate(x, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x2_att = self.att3(g=x, x=x2)
        x = self.conv3(torch.cat([x, x2_att], dim=1))
        
        x = self.up4(x)
        x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x1_att = self.att4(g=x, x=x1)
        x = self.conv4(torch.cat([x, x1_att], dim=1))
        
        return self.outc(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ MODIFIED for num_classes=2 output"""
    def __init__(self, pretrained=True, out_ch=2): # <-- MODIFIED
        super(DeepLabV3Plus, self).__init__()
        
        aspp_out_channels = 256
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.segmentation.deeplabv3_resnet50(weights=weights, progress=True)

        # Replace the classifier head
        if hasattr(self.model, "classifier") and isinstance(self.model.classifier, DeepLabHead):
            try:
                final_conv_layer = self.model.classifier[-1]
                if not isinstance(final_conv_layer, nn.Conv2d):
                    raise TypeError(f"Expected last layer to be nn.Conv2d, but got {type(final_conv_layer)}")
                
                in_channels: int = final_conv_layer.in_channels
                new_final_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1) # <-- Uses out_ch
                
                nn.init.kaiming_normal_(new_final_conv.weight, mode='fan_out', nonlinearity='relu')
                if new_final_conv.bias is not None:
                    nn.init.constant_(new_final_conv.bias, 0)
                self.model.classifier[-1] = new_final_conv
            
            except Exception as e:
                print(f"[WARN] Failed to replace DeepLabHead final layer: {e}. Fallback.")
                self.model.classifier = nn.Sequential(
                    DeepLabHead(2048, aspp_out_channels),
                    nn.Conv2d(aspp_out_channels, out_ch, 1) # <-- Uses out_ch
                )
        else:
            print("[WARN] Could not find DeepLabHead, replacing entire classifier.")
            self.model.classifier = nn.Sequential(
                DeepLabHead(2048, aspp_out_channels),
                nn.Conv2d(aspp_out_channels, out_ch, 1) # <-- Uses out_ch
            )

        if hasattr(self.model, "aux_classifier"):
            self.model.aux_classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: Dict[str, torch.Tensor] = self.model(x)
        
        # --- THIS IS THE FIX ---
        # Use .get() for safer dictionary access that satisfies Pylance
        out_tensor = outputs.get("out")
        if out_tensor is None:
            # This should not happen if the model is correct
            raise KeyError("Model output dictionary does not contain 'out' key.")
        return out_tensor
        # -----------------------



#   Loss 

def combined_ce_dice_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    num_classes: int, 
    device: torch.device,
    ce_weight=1.0, 
    dice_weight=1.0
) -> torch.Tensor:
    """Combines CrossEntropy and multiclass Dice loss"""
    
    # Class weights for CE
    try:
        cw = compute_batch_class_weights(targets, num_classes)
    except Exception:
        cw = torch.ones(num_classes, device=device)
        
    # Cross Entropy Loss
    ce = F.cross_entropy(logits, targets, weight=cw)
    
    # Dice Loss (from lossFunction.py)
    d_loss = dice_loss(logits, targets)
    
    return ce_weight * ce + dice_weight * d_loss


#   Unified Training Routine 

def train_one_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                    device: torch.device, num_classes: int, num_epochs: int = 5, lr: float = 1e-4, 
                    weight_decay: float = 1e-4, grad_clip: float = 1.0, 
                    save_dir: str = "./checkpoints", 
                    early_stop_patience: int = 10, image_size: tuple = (256, 256)):
    """
    Train model with mixed precision & safe resizing.
    MODIFIED:
    - Expects multiclass masks [B,H,W] (0, 1 for Fish).
    - Uses CE + Multiclass Dice loss.
    - Uses compute_segmentation_metrics for evaluation.
    - Returns model, history dict, and best epoch index.
    """
    set_seed(42)
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs), eta_min=1e-6)
    
    amp_enabled = (device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    best_val_dice = -1.0
    best_state = None
    best_epoch_idx = -1
    epochs_no_improve = 0
    
    metrics_history: Dict[str, Dict[str, List[float]]] = {
        "train": {k: [] for k in METRIC_KEYS},
        "val": {k: [] for k in METRIC_KEYS}
    }

    print(f"\nTraining started on {device.type.upper()} for {num_epochs} epochs (AMP: {amp_enabled})...")
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        n_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", ncols=120)

        for batch in pbar:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device) # [B, H, W] Long
            
            # Resize
            imgs = safe_resize_tensor(imgs, size=image_size, is_mask=False)
            masks = safe_resize_tensor(masks, size=image_size, is_mask=True)

            optimizer.zero_grad()
            
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits_output = model(imgs) # Can be Tensor or Tuple
                
                # Handle physics model output
                if isinstance(logits_output, (tuple, list)):
                    logits = logits_output[0]
                else:
                    logits = logits_output
                
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                
                loss = combined_ce_dice_loss(
                    logits, masks, num_classes=num_classes, device=device
                )

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            n_train += 1
            pbar.set_postfix({"loss": f"{total_train_loss / n_train:.4f}"})

        # --- Calculate Full Train Metrics (Slow) ---
        print("Calculating full training metrics...")
        train_metrics = compute_segmentation_metrics_baseline(
            model, train_loader, device, num_classes
        )
        avg_train_loss = total_train_loss / n_train
        metrics_history["train"]["Loss"].append(avg_train_loss)
        for k in METRIC_KEYS:
            if k in train_metrics:
                metrics_history["train"][k].append(train_metrics[k])


        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        n_val = 0
        
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", ncols=120)
        with torch.no_grad():
            for batch in pbar_val:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device) # [B, H, W] Long
                
                imgs = safe_resize_tensor(imgs, size=image_size, is_mask=False)
                masks = safe_resize_tensor(masks, size=image_size, is_mask=True)

                with autocast(device_type=device.type, enabled=amp_enabled):
                    logits_output = model(imgs) # Can be Tensor or Tuple
                    
                    # Handle physics model output
                    if isinstance(logits_output, (tuple, list)):
                        logits = logits_output[0]
                    else:
                        logits = logits_output # It's just a tensor (from UNet, AttnUNet, or DeepLab)

                    if logits.shape[-2:] != masks.shape[-2:]:
                        logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    loss = combined_ce_dice_loss(
                        logits, masks, num_classes=num_classes, device=device
                    )

                total_val_loss += loss.item()
                n_val += 1
                pbar_val.set_postfix({"val_loss": f"{total_val_loss / n_val:.4f}"})

        # --- Calculate Full Val Metrics ---
        print("Calculating full validation metrics...")
        val_metrics = compute_segmentation_metrics_baseline(
            model, val_loader, device, num_classes
        )
        avg_val_loss = total_val_loss / n_val
        metrics_history["val"]["Loss"].append(avg_val_loss)
        avg_val_dice = 0.0
        for k in METRIC_KEYS:
            if k in val_metrics:
                metrics_history["val"][k].append(val_metrics[k])
                if k == "Dice":
                    avg_val_dice = val_metrics[k]

        # Scheduler step
        try:
            scheduler.step()
        except Exception:
            pass

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")
        print(f"  Val mIoU: {val_metrics['mIoU']:.4f} | Val Precision: {val_metrics['Precision']:.4f} | Val Recall: {val_metrics['Recall']:.4f} | Val PixAcc: {val_metrics['pixel_accuracy']}")

        # Checkpoint best by Dice
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_state = copy.deepcopy(model.state_dict())
            best_epoch_idx = epoch
            best_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}_dice_{best_val_dice:.4f}.pth")
            torch.save(best_state, best_path)
            print(f"Saved best model -> {best_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping: no improvement for {early_stop_patience} epochs.")
            break

    # Load best weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)
        
    print(f"\nTraining finished. Best model (Epoch {best_epoch_idx + 1}) loaded.")
    return model, metrics_history, best_epoch_idx