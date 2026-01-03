# mamba_block.py
"""
Input
  ↓
Linear Projection
  ↓
Depthwise Causal Convolution  ← local mixing
  ↓
Gating + Feature Modulation   ← 🔥 THIS is the "in-between" part
  ↓
Selective SSM                ← long-range memory
  ↓
Output Projection
"""

import torch
import torch.nn as nn
#from ssm import SSM
from ssm import SimpleSSM

class SimpleMamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.ssm = SimpleSSM(d_model)
        print("Initialized SimpleMamba SSM",d_model,self.ssm)


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
    
class VisionMamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        # project into x and z
        self.in_proj = nn.Linear(d_model, 2 * d_model)

        # forward branch
        self.fwd_conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.fwd_ssm  = SimpleSSM(d_model)

        # backward branch
        self.bwd_conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.bwd_ssm  = SimpleSSM(d_model)

        # gate branch
        self.activation = nn.SiLU()

        # output projection
        self.out_proj = nn.Linear(d_model, d_model)
        print("Initialized VisionMamba",d_model)

    def forward(self, x):
        residual = x

        x = self.norm(x)
        x, z = self.in_proj(x).chunk(2, dim=-1)

        # forward path
        f = self.fwd_conv(x.transpose(1,2)).transpose(1,2)
        f = self.fwd_ssm(f)

        # backward path
        b = torch.flip(x, dims=[1])
        b = self.bwd_conv(b.transpose(1,2)).transpose(1,2)
        b = self.bwd_ssm(b)
        b = torch.flip(b, dims=[1])

        # fb = (f + b) * z  |     y = self.ssm(fb)

        # gate
        z = self.activation(z)

        # combine
        y = f * z + b * z

        return self.out_proj(y) + residual
