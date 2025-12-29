import torch.nn as nn
from FrameEncoder import FrameEncoder
from classifier import MambaClassifier

class MERModel(nn.Module):
    def __init__(self, d_model=64, num_classes=7):
        super().__init__()
        self.encoder = FrameEncoder(d_model)
        self.mamba = MambaClassifier(d_model, num_classes)
        #print("Initialized MERModel",d_model,num_classes,self.encoder,self.mamba)

    def forward(self, x):
        x = self.encoder(x)   # (B,T,D)
        print("Encoder output:", x.shape)
        return self.mamba(x)
