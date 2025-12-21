# physicsFunctions.py
"""
Simple classical estimators used for weak supervision.
These are heuristic and intentionally simple — replace with your domain-specific estimators if you have them.
"""

import torch
import torch.nn.functional as F

def estimate_transmission(images: torch.Tensor) -> torch.Tensor:
    """
    Naive transmission estimator: 1 - local brightness normalized.
    images: [B,3,H,W] in [0,1]
    returns [B,1,H,W]
    """
    avg = images.mean(dim=1, keepdim=True)  # [B,1,H,W]
    # invert brightness to approximate turbidity: brighter -> less turbidity
    t = 1.0 - avg
    return t.clamp(0.0, 1.0)

def estimate_backscatter(images: torch.Tensor) -> torch.Tensor:
    """
    Naive backscatter estimator: local average color with low-pass filter (gaussian-like).
    returns [B,3,H,W]
    """
    # simple box blur implemented via depthwise conv
    B, C, H, W = images.shape
    kernel = torch.ones((1,1,3,3), device=images.device) / 9.0
    pad = 1
    # apply per-channel by grouping
    images_pad = F.pad(images, (pad,pad,pad,pad), mode="reflect")
    # depthwise conv via grouping across channels
    kernel = kernel.repeat(C,1,1,1)
    out = F.conv2d(images_pad, kernel, groups=C)
    return out.clamp(0.0,1.0)
