import torch.nn as nn
from torchvision import models

class FrameEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        f = self.features(x).flatten(1)
        f = self.fc(f)
        return f.view(B, T, -1)
