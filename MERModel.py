############### MerModel.py ########################
"""
Input video frames (B, T, C, H, W)
        ▼
FrameEncoder → Temporal feature embedding (B, T, D)
        ▼
Dropout (regularization)
        ▼
MambaClassifier → Class scores (B, num_classes)

# Input(B,T,C,H,W) → FrameEncoder(B,T,D) → Dropout(B,T,D) → MambaClassifier(B,num_classes) → Output(B,num_classes)
"""
##########################################################################

import torch.nn as nn
from FrameEncoder import FrameEncoder
#from classifier import MambaClassifier
from MambaClassifier import MambaClassifier

class MERModel(nn.Module):
    def __init__(self, d_model=64, num_classes=7, dropout=0.2):
        super().__init__()
        self.encoder = FrameEncoder(d_model)
        self.mamba = MambaClassifier(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)  # optional dropout for regularization
        #print("Initialized MERModel",d_model,num_classes,self.encoder,self.mamba)
        print("Initialized MERModel",d_model,num_classes)

    def forward(self, x):
        x = self.encoder(x)   # (B,T,D)
        x = self.dropout(x)
        #print("Encoder output:", x.shape)
        out = self.mamba(x)       # (B, num_classes)
        return out
        
        