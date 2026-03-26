# physicsFunctions.py
import torch
import torch.nn.functional as F

def estimate_transmission(images: torch.Tensor) -> torch.Tensor:
    avg = images.mean(dim=1, keepdim=True)
    t = 1.0 - avg
    return t.clamp(0.0, 1.0)

def estimate_backscatter(images: torch.Tensor) -> torch.Tensor:
    B, C, H, W = images.shape
    kernel = torch.ones((1,1,3,3), device=images.device) / 9.0
    pad = 1
    images_pad = F.pad(images, (pad,pad,pad,pad), mode="reflect")
    kernel = kernel.repeat(C,1,1,1)
    out = F.conv2d(images_pad, kernel, groups=C)
    return out.clamp(0.0,1.0)