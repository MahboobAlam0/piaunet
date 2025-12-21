# physicsComponents.py
"""
MODIFIED to support the architecture diagram.
The PhysicsGuidedModule now returns the intermediate feature map 'x'
which is used as the "Physics Features (p)" in the main model.
"""

import torch
import torch.nn as nn

class PhysicsGuidedModule(nn.Module):
    def __init__(self, in_channels: int = 512, mid: int = 256):
        super().__init__()
        # reduce channels
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        # turbidity: single channel map
        self.turbidity_head = nn.Sequential(
            nn.Conv2d(mid, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),  # map in [0,1]
        )
        # backscatter: 3-channel map (RGB)
        self.backscatter_head = nn.Sequential(
            nn.Conv2d(mid, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid(),  # map in [0,1]
        )
        
    def forward(self, feat: torch.Tensor):
        """
        feat: [B, C, H, W] from bottleneck
        returns:
            t: turbidity [B, 1, H, W]
            b: backscatter [B, 3, H, W]
            x: physics features (p) [B, mid, H, W]
        """
        x = self.reduce(feat)
        t = self.turbidity_head(x)
        b = self.backscatter_head(x)
        # Return T, B, and the intermediate features 'x' as 'p'
        return t, b, x