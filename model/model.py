import torch
import torch.nn as nn
from ..physics.physicsComponents import PhysicsGuidedModule


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class PhysicsAwareAttentionGate(nn.Module):
    """Attention gate that uses physics information (turbidity) for gating"""
    def __init__(self, f_g, f_l, f_int, use_physics=True):
        super().__init__()
        self.use_physics = use_physics
        
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        
        # Physics branch: turbidity-aware gating
        if use_physics:
            self.W_phys = nn.Sequential(
                nn.Conv2d(1, f_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(f_int)
            )
        
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, t=None):
        """
        g: gating signal (decoder feature)
        x: skip connection (encoder feature)
        t: turbidity map (physics info)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        if self.use_physics and t is not None:
            t_up = torch.nn.functional.interpolate(t, size=x.shape[-2:], mode='nearest')
            t1 = self.W_phys(t_up)
            psi = self.relu(g1 + x1 + t1)
        else:
            psi = self.relu(g1 + x1)
        
        psi = self.psi(psi)
        return x * psi


class PhysicsInformedAttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        super().__init__()

        # encoder (4 depth levels)
        self.enc1 = ConvBlock(in_ch, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # bottleneck: physics-guided module
        self.physics_module = PhysicsGuidedModule(in_channels=512, mid=256)
        
        # Project concatenated features back to 512
        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(512 + 256, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # decoder (4 depth levels) with physics-aware attention gates
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.att3 = PhysicsAwareAttentionGate(f_g=256, f_l=256, f_int=128, use_physics=True)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.att2 = PhysicsAwareAttentionGate(f_g=128, f_l=128, f_int=64, use_physics=True)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.att1 = PhysicsAwareAttentionGate(f_g=64, f_l=64, f_int=32, use_physics=True)

        # segmentation heads
        self.seg_main = nn.Conv2d(64, out_ch, 1)
        self.seg_aux1 = nn.Conv2d(128, out_ch, 1)
        self.seg_aux2 = nn.Conv2d(256, out_ch, 1)

        # illumination head from physics features
        self.j_head = nn.Conv2d(64, 3, 1)

    def forward(self, x):

        # encoder (4 depth)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # physics-guided bottleneck
        t, b, phys_feat = self.physics_module(e4)
        
        # Combine encoder and physics features
        bottleneck = self.bottleneck_proj(torch.cat([e4, phys_feat], dim=1))

        # decoder (4 depth) with physics-aware attention
        # Level 3
        d3 = self.up3(bottleneck)
        e3_att = self.att3(g=d3, x=e3, t=t)  # Physics-aware attention
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        # Level 2
        d2 = self.up2(d3)
        e2_att = self.att2(g=d2, x=e2, t=t)  # Physics-aware attention
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        # Level 1
        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1, t=t)  # Physics-aware attention
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        # segmentation outputs
        seg_main = self.seg_main(d1)
        seg_aux = [
            self.seg_aux1(d2),
            self.seg_aux2(d3)
        ]

        # physics outputs: turbidity and backscatter from module, illumination from decoder
        j = torch.sigmoid(self.j_head(d1))

        return seg_main, seg_aux, j, t, b