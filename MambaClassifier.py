import torch
import torch.nn as nn
# from mamba_ssm import Mamba
#from mamba import SimpleMamba
from mamba import VisionMamba

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

   Input
  ↓
Linear Projection
  ↓
Depthwise Causal Convolution  ← local mixing
  ↓
Gating + Feature Modulation   
  ↓
Selective SSM                ← long-range memory
  ↓
Output Projection


"""

class MambaClassifier(nn.Module):
    def __init__(self, d_model=64, num_classes=7):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)

        # input projection and split
        self.in_proj = nn.Linear(d_model, 2 * d_model)

        # Path A
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

        #feature extractor
        self.backbone = VisionMamba(d_model)
        #self.backbone = SimpleMamba(d_model)
        ''' for real mamba;  
        #self.backbone = Mamba(d_model, ...)
        self.backbone = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )

    
        '''
        # Path B
        self.activation = nn.SiLU()

        # output projection
        self.out_proj = nn.Linear(d_model, d_model)

        #decision head
        self.classifier = nn.Linear(d_model,num_classes)
        print("Initialized MambaClassifier",d_model,num_classes)

    def forward(self, x):
        residual = x

        x = self.norm(x)
        x = self.in_proj(x)

        x1, x2 = x.chunk(2, dim=-1)   # split

        '''# Path A: Conv → SSM
        x1 = self.conv1d(x1.transpose(1,2)).transpose(1,2)
        # Path B: Activation
        x2 = self.activation(x2)

        # Gating path A
        x1_modulated = x1 * torch.sigmoid(x2)
        x1 = self.backbone(x1)
        # Gating
        x = x1 * x2

        # Output projection + residual
        x = self.out_proj(x)
        return x + residual
        '''

        # Path A: Conv → SSM
        x1 = self.conv1d(x1.transpose(1,2)).transpose(1,2)
        x1 = self.backbone(x1)

        # Path B: Gating
        x2 = self.activation(x2)

        # Gated fusion
        x = x1 * torch.sigmoid(x2)

        # Output projection + residual
        x = self.out_proj(x)
        x = x + residual          # [B, T, 64]

        # TEMPORAL POOLING
        x = x.mean(dim=1)         # [B, 64]

        #  CLASSIFICATION HEAD
        logits = self.classifier(x)   # [B, num_classes]

        return logits

    
        """h = self.backbone(x)       # (B, T, D)

        h_last = h[:, -1, :]       # B D
        logits = self.classifier(h_last)  # (B, num_classes)

        print(f"MambaClassifier done, Backbone:{h.shape} Last: {h_last.shape} Logits: {logits.shape}")
    
        return logits"""
    
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
