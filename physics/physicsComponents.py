# physicsComponents.py
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
        x = self.reduce(feat)
        t = self.turbidity_head(x)
        b = self.backscatter_head(x)
        return t, b, x
