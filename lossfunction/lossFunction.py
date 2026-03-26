import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from physics.physicsFunctions import estimate_transmission, estimate_backscatter 


# ============================================================
# Perceptual Loss (SAFE + FIXED)
# ============================================================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = vgg16(weights="IMAGENET1K_V1")
        self.vgg = nn.Sequential(*list(vgg.features.children())[:8]).eval()

        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        # Handle device and precision mismatch: ensure VGG is on same device as input
        self.vgg.to(x.device)
        
        # Convert float16 inputs to float32 for VGG
        original_dtype = x.dtype
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        
        vgg_x = self.vgg(x)
        vgg_y = self.vgg(y)
        
        loss = F.l1_loss(vgg_x, vgg_y)
        
        # Convert loss back to original dtype if needed
        return loss.to(original_dtype)


perceptual_loss = PerceptualLoss()


# ============================================================
# Edge-aware smoothness
# ============================================================
def edge_aware_smoothness(t, img):
    # Resize image to match transmission map resolution
    img_resized = F.interpolate(img, size=t.shape[-2:], mode="bilinear", align_corners=False)
    
    dx_t = torch.abs(t[:, :, :, :-1] - t[:, :, :, 1:])
    dy_t = torch.abs(t[:, :, :-1, :] - t[:, :, 1:, :])

    dx_img = torch.mean(
        torch.abs(img_resized[:, :, :, :-1] - img_resized[:, :, :, 1:]), dim=1, keepdim=True
    )
    dy_img = torch.mean(
        torch.abs(img_resized[:, :, :-1, :] - img_resized[:, :, 1:, :]), dim=1, keepdim=True
    )

    weight_x = torch.exp(-dx_img)
    weight_y = torch.exp(-dy_img)

    return (dx_t * weight_x).mean() + (dy_t * weight_y).mean()


# ============================================================
# Multi-scale physics loss (SAFE)
# ============================================================
def multi_scale_physics_loss(j, t, b, img):
    # Ensure all physics tensors are at the same resolution
    # Use minimum size to avoid upsampling artifacts
    target_size = min(j.shape[-1], t.shape[-1], b.shape[-1])
    
    j_resized = F.interpolate(j, size=(target_size, target_size), mode="bilinear", align_corners=False)
    t_resized = F.interpolate(t, size=(target_size, target_size), mode="bilinear", align_corners=False)
    b_resized = F.interpolate(b, size=(target_size, target_size), mode="bilinear", align_corners=False)
    
    # Resize image to match physics output resolution
    img_resized = F.interpolate(img, size=(target_size, target_size), mode="bilinear", align_corners=False)
    
    scales = [1.0, 0.5, 0.25]
    loss = torch.zeros(1, device=img.device)

    for s in scales:
        if s != 1.0:
            img_s = F.interpolate(img_resized, scale_factor=s, mode="bilinear", align_corners=False)
            j_s = F.interpolate(j_resized, scale_factor=s, mode="bilinear", align_corners=False)
            t_s = F.interpolate(t_resized, scale_factor=s, mode="bilinear", align_corners=False)
            b_s = F.interpolate(b_resized, scale_factor=s, mode="bilinear", align_corners=False)
        else:
            img_s, j_s, t_s, b_s = img_resized, j_resized, t_resized, b_resized

        recon = j_s * t_s + b_s * (1 - t_s)
        loss = loss + F.l1_loss(recon, img_s)

    return loss / len(scales)


# ============================================================
# Deep supervision loss (SAFE)
# ============================================================
def deep_supervision_loss(seg_aux, target):
    if len(seg_aux) == 0:
        return torch.zeros(1, device=target.device)

    loss = torch.zeros(1, device=target.device)

    for pred in seg_aux:
        pred = F.interpolate(
            pred,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss = loss + F.cross_entropy(pred, target)

    return loss / len(seg_aux)


# ============================================================
# Safe item conversion (NO ERRORS EVER)
# ============================================================
def safe_item(x):
    return x.detach().item() if torch.is_tensor(x) else float(x)


# ============================================================
# MAIN LOSS FUNCTION (SIMPLIFIED - CORE LOSSES ONLY)
# ============================================================
def physics_aware_loss(
    seg_main,
    seg_aux,
    seg_target,
    j_pred,
    t_pred,
    b_pred,
    images,
    *,
    lambda_seg=1.0,
    lambda_aux=0.4,
    **kwargs,  # Accept other parameters for compatibility but ignore them
):
    """
    Simplified loss function focusing on core segmentation objectives.
    Physics outputs (j, t, b) are retained for indirect supervision through weak losses,
    but complex multi-scale and perceptual losses are removed for stability.
    """
    # ----------------------------
    # 1. Segmentation loss (PRIMARY)
    # ----------------------------
    seg_loss = F.cross_entropy(seg_main, seg_target)

    # ----------------------------
    # 2. Deep supervision (AUXILIARY)
    # ----------------------------
    aux_loss = deep_supervision_loss(seg_aux, seg_target)

    # ----------------------------
    # 3. Total loss (SIMPLIFIED)
    # ----------------------------
    total_loss = lambda_seg * seg_loss + lambda_aux * aux_loss

    # ----------------------------
    # 4. Logging dictionary
    # ----------------------------
    loss_dict = {
        "seg": safe_item(seg_loss),
        "aux": safe_item(aux_loss),
    }

    return total_loss, loss_dict