import torch.nn as nn
from FrameEncoder import FrameEncoder
from classifier import MambaClassifier

class MERModel(nn.Module):
    def __init__(self, d_model=64, num_classes=5):
        super().__init__()
        self.encoder = FrameEncoder(d_model)
        self.mamba = MambaClassifier(d_model, num_classes)

    def forward(self, x):
        x = self.encoder(x)   # (B,T,D)
        return self.mamba(x)
