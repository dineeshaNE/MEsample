# mamba_block.py
import torch
import torch.nn as nn
#from ssm import SSM
from ssm import SimpleSSM

class SimpleMamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.ssm = SimpleSSM(d_model)

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """

        ssm_out = self.ssm(x)
        gate = self.gate(x)

        y = gate * ssm_out
        return self.out(y)
    
    """Shape & Logic Sanity Check
Component	Shape
x	(B, T, D)
ssm_out	(B, T, D)
gate	(B, T, D)
y = gate * ssm_out	(B, T, D)
out(y)	(B, T, D)
"""

class Mamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.ssm = SimpleSSM(d_model)

        # Input-dependent gates (key Mamba idea)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        # Split input into value + gate
        v, gate = self.in_proj(x).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)

        # Pass through SSM
        ssm_out = self.ssm(v)

        # Gate the SSM output
        y = gate * ssm_out
        return self.out_proj(y)