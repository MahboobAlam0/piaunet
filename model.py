import torch
import torch.nn as nn
import torch.nn.functional as F
from physicsComponents import PhysicsGuidedModule


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
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

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class PAG(nn.Module):
    
    def __init__(self, skip_channels: int, physics_channels: int):
        super().__init__()
        self.proj_p = nn.Conv2d(physics_channels, skip_channels, kernel_size=1)
        
        self.gate = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels, skip_channels, 1),
            nn.Sigmoid()
        )
        
        self.conv_out = DoubleConv(skip_channels, skip_channels)

    def forward(self, F_skip: torch.Tensor, p_feat: torch.Tensor) -> torch.Tensor:
        
        p_up = F.interpolate(p_feat, size=F_skip.shape[2:], mode='bilinear', align_corners=False)
        
        p_proj = self.proj_p(p_up)
        
        gate = self.gate(p_proj)
        
        F_gated = F_skip * (1 + gate)
        
        return self.conv_out(F_gated)

class PhysicsAwareAttentionUNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 2):
        super().__init__()
        
        # --- Encoder ---
        self.inc = DoubleConv(in_ch, 64)       # F1: 64
        self.down1 = Down(64, 128)             # F2: 128
        self.down2 = Down(128, 256)            # F3: 256
        self.down3 = Down(256, 512)            # F4: 512
        self.down4 = Down(512, 1024)           # F5: 1024

        # --- Physics Branch ---
        # Takes F5 (1024 ch) and outputs T, B, and p_features (256 ch)
        self.physics_branch = PhysicsGuidedModule(in_channels=1024, mid=256)
        
        # --- Decoder ---
        self.pag4 = PAG(512, 256)
        self.pag3 = PAG(256, 256)
        self.pag2 = PAG(128, 256)
        self.pag1 = PAG(64, 256)
        
        # Upsampling layers
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.conv_d4 = DoubleConv(512, 512) # For F4'
        self.conv_d3 = DoubleConv(256, 256) # For F3'
        self.conv_d2 = DoubleConv(128, 128) # For F2'
        self.conv_d1 = DoubleConv(64, 64)   # For F1'

        self.outc = nn.Conv2d(64, out_ch, kernel_size=1)
        
        self.j_head = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1), 
            nn.Sigmoid()                     
        )

    def forward(self, x: torch.Tensor):
        # --- Encoder ---
        f1 = self.inc(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)

        # --- Physics Branch ---
        t_pred, b_pred, p_feat = self.physics_branch(f5)

        # --- Decoder ---
        f4_p = self.pag4(f4, p_feat) # 512 ch
        d4 = self.conv_d4(f4_p)

        d3_up = self.up4(d4) # 256 ch
        f3_p = self.pag3(f3, p_feat) # 256 ch
        d3 = self.conv_d3(d3_up + f3_p) 

        d2_up = self.up3(d3) # 128 ch
        f2_p = self.pag2(f2, p_feat) # 128 ch
        d2 = self.conv_d2(d2_up + f2_p) 

        d1_up = self.up2(d2) # 64 ch
        f1_p = self.pag1(f1, p_feat) # 64 ch
        d1 = self.conv_d1(d1_up + f1_p) 

        logits = self.outc(d1)
        
        j_pred = self.j_head(d1)

        return logits, t_pred, b_pred, j_pred