import torch
import torch.nn as nn
# from mamba_ssm import Mamba
from mamba import SimpleMamba

class MambaClassifier(nn.Module):
    def __init__(self, d_model=64, num_classes=2):
        super().__init__()
        
        #Video → Backbone → Features → Classifier → Prediction
        #feature extractor
        self.backbone = SimpleMamba(d_model)
        ''' for real mamba;  
        self.backbone = Mamba(d_model, ...)'''

        #decision head
        self.classifier = nn.Linear(d_model,num_classes)
        #print("Initialized MambaClassifier",d_model,num_classes,self.backbone,self.classifier)

    def forward(self, x):
        h = self.backbone(x)       # (B, T, D)
        h_last = h[:, -1, :]       # B D
        logits = self.classifier(h_last)  # (B, num_classes)

        print("Backbone:", h.shape)
        print("Last:", h_last.shape)
        print("Logits:", logits.shape)
        return logits
    
""" mamba_ssm
class MambaClassifier(nn.Module):
    def __init__(self, d_model=64, num_classes=2):
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
