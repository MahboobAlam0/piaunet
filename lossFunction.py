from typing import Optional, Tuple, Dict
import torch
import torch.nn.functional as F
from physicsFunctions import estimate_backscatter, estimate_transmission

def compute_batch_class_weights(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    device = target.device
    flat = target.view(-1)
    counts = torch.bincount(flat, minlength=num_classes).float()
    counts = counts + 1e-6
    freq = counts / counts.sum()
    weights = 1.0 / freq
    weights = weights / (weights.sum() / float(num_classes))
    return weights.to(device)

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num_classes = pred_logits.shape[1]
    pred_prob = torch.softmax(pred_logits, dim=1)
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0,3,1,2).float()
    intersection = (pred_prob * target_onehot).sum(dim=(0,2,3))
    union = (pred_prob + target_onehot).sum(dim=(0,2,3))
    dice_per_class = (2.0 * intersection + eps) / (union + eps)
    if num_classes > 1:
        return 1.0 - dice_per_class[1:].mean()
    else:
        return 1.0 - dice_per_class.mean()

def spatial_tv(x: Optional[torch.Tensor]) -> torch.Tensor:
    if x is None:
        return torch.tensor(0.0)
    dx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).mean()
    dy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).mean()
    return dx + dy

def physics_aware_loss(
    seg_logits: torch.Tensor,
    seg_target: torch.Tensor,
    j_pred: Optional[torch.Tensor] = None,
    t_pred: Optional[torch.Tensor] = None,
    b_pred: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    *,
    lambda_seg: float = 1.0,
    lambda_phys: float = 0.5,
    lambda_smooth: float = 0.05,
    lambda_B_sup: float = 0.02,
    lambda_T_sup: float = 0.05,
    num_classes: int = 8,
    class_weights_override: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    device = seg_logits.device
    if class_weights_override is not None:
        cw = class_weights_override.to(device)
    else:
        try:
            cw = compute_batch_class_weights(seg_target, num_classes)
        except Exception:
            cw = torch.ones(num_classes, device=device)

    ce = F.cross_entropy(seg_logits, seg_target, weight=cw)
    dloss = dice_loss(seg_logits, seg_target)
    seg_loss = ce + dloss

    phys_loss = torch.tensor(0.0, device=device)
    recon_used = False
    if j_pred is not None and t_pred is not None and b_pred is not None and images is not None:
        # resize to seg resolution
        target_size = seg_logits.shape[2:]
        j_res = F.interpolate(j_pred, size=target_size, mode="bilinear", align_corners=False) if j_pred.shape[2:] != target_size else j_pred
        t_res = F.interpolate(t_pred, size=target_size, mode="bilinear", align_corners=False) if t_pred.shape[2:] != target_size else t_pred
        b_res = F.interpolate(b_pred, size=target_size, mode="bilinear", align_corners=False) if b_pred.shape[2:] != target_size else b_pred
        img_res = F.interpolate(images, size=target_size, mode="bilinear", align_corners=False) if images.shape[2:] != target_size else images

        t_rep = t_res.repeat(1,3,1,1) if t_res.shape[1] == 1 else t_res
        b_rep = b_res.repeat(1,3,1,1) if b_res.shape[1] == 1 else b_res

        recon = j_res * t_rep + b_rep * (1.0 - t_rep)
        phys_loss = F.l1_loss(recon, img_res)
        recon_used = True

    smooth_loss = torch.tensor(0.0, device=device)
    if t_pred is not None:
        smooth_loss = spatial_tv(t_pred)
    if b_pred is not None:
        smooth_loss = smooth_loss + spatial_tv(b_pred)

    B_sup_loss = torch.tensor(0.0, device=device)
    T_sup_loss = torch.tensor(0.0, device=device)
    if images is not None:
        try:
            est_t = estimate_transmission(images)
            est_b = estimate_backscatter(images)
            if b_pred is not None:
                # adapt shapes
                b_est_res = est_b
                if b_est_res.shape[2:] != b_pred.shape[2:]:
                    b_est_res = F.interpolate(b_est_res, size=b_pred.shape[2:], mode="bilinear", align_corners=False)
                if b_pred.shape[1] == 1:
                    b_est_res = b_est_res.mean(dim=1, keepdim=True)
                B_sup_loss = F.l1_loss(b_pred, b_est_res)
            if t_pred is not None:
                est_t_res = est_t
                if est_t_res.shape[2:] != t_pred.shape[2:]:
                    est_t_res = F.interpolate(est_t_res, size=t_pred.shape[2:], mode="bilinear", align_corners=False)
                T_sup_loss = F.l1_loss(t_pred, est_t_res)
        except Exception:
            B_sup_loss = torch.tensor(0.0, device=device)
            T_sup_loss = torch.tensor(0.0, device=device)

    total_loss = (
        lambda_seg * seg_loss
        + lambda_phys * phys_loss
        + lambda_smooth * smooth_loss
        + lambda_B_sup * B_sup_loss
        + lambda_T_sup * T_sup_loss
    )

    loss_dict = {
        "total_loss": float(total_loss.detach().cpu().item()),
        "seg_loss": float(seg_loss.detach().cpu().item()),
        "ce_loss": float(ce.detach().cpu().item()),
        "dice_loss": float(dloss.detach().cpu().item()),
        "physics_loss": float(phys_loss.detach().cpu().item() if recon_used else 0.0),
        "smooth_loss": float(smooth_loss.detach().cpu().item()),
        "B_sup_loss": float(B_sup_loss.detach().cpu().item()),
        "T_sup_loss": float(T_sup_loss.detach().cpu().item()),
    }

    return total_loss, loss_dict
