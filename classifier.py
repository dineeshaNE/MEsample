import torch
import torch.nn as nn
# from mamba_ssm import Mamba
from mamba import SimpleMamba

"""
Video frames (B, T, C, H, W)
        │
FrameEncoder → features (B, T, out_dim)
        │
Temporal Model → video-level embedding → classifier → Prediction
        │
Output → emotion class

Input
 ├─ Norm
 ├─ Linear projection → split into 2 paths
 │
 ├─ Path A:   Conv1D → SSM Scan (Ω) 
 │
 └─ Path B:   Activation
 
 Then:
   Path A ⊙ Path B   (element-wise gate)
   → Linear projection
   → Residual add with original input

"""

class MambaClassifier(nn.Module):
    def __init__(self, d_model=64, num_classes=7):
        super().__init__()
        


        #feature extractor
        self.backbone = SimpleMamba(d_model)
        ''' for real mamba;  
        #self.backbone = Mamba(d_model, ...)
        self.backbone = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )

    
        '''


        #decision head
        self.classifier = nn.Linear(d_model,num_classes)
        #print("Initialized MambaClassifier",d_model,num_classes,self.backbone,self.classifier)

    def forward(self, x):
        h = self.backbone(x)       # (B, T, D)
        h_last = h[:, -1, :]       # B D
        logits = self.classifier(h_last)  # (B, num_classes)

        print(f"MambaClassifier done, Backbone:{h.shape} Last: {h_last.shape} Logits: {logits.shape}")
        return logits
    
""" mamba_ssm for real mamba;
class MambaClassifier(nn.Module):
    def __init__(self, d_model=64, num_classes=7):
        super().__init__()
        
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        h = self.mamba(x)           # (B, T, D)
        h_last = h[:, -1, :]        # last timestep
        return self.classifier(h_last)
        """
