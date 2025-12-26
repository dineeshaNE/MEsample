# MambaMER is defined here
# it referes mamba from imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class FrameFeatureExtractor(nn.Module):
    """
    CNN to extract spatial features from each frame.
    """
    def __init__(self, in_channels=1, feature_dim=128):
        super(FrameFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # Bx32xHxW
            nn.ReLU(),
            nn.MaxPool2d(2),  # Bx32xH/2xW/2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Bx64xH/2xW/2
            nn.ReLU(),
            nn.MaxPool2d(2),  # Bx64xH/4xW/4
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),  # Bx128xH/4xW/4
            nn.AdaptiveAvgPool2d((1, 1))  # Bx128x1x1
        )

    def forward(self, x):  # x: [B*T, C, H, W]
        x = self.cnn(x)
        return x.view(x.size(0), -1)  # [B*T, feature_dim]


class MambaMER(nn.Module):
    def __init__(self, in_channels=1, feature_dim=128, mamba_dim=128, num_classes=5, seq_len=16):
        super(MambaMER, self).__init__()
        self.feature_extractor = FrameFeatureExtractor(in_channels, feature_dim)
        self.seq_len = seq_len
        self.mamba = Mamba(d_model=mamba_dim)
        self.proj = nn.Linear(feature_dim, mamba_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(mamba_dim),
            nn.Linear(mamba_dim, num_classes)
        )

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x)  # [B*T, feature_dim]
        feats = feats.view(B, T, -1)  # [B, T, feature_dim]
        feats = self.proj(feats)  # [B, T, mamba_dim]
        mamba_out = self.mamba(feats)  # [B, T, mamba_dim]
        pooled = mamba_out.mean(dim=1)  # temporal average
        return self.classifier(pooled)  # [B, num_classes]
