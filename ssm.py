# ssm.py

"""
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


# Input x (B, T, D)
    ↓
Init state (s or h) = 0
    ↓
For t in 1…T:
    ├─ Extract xt = x[:, t, :]
    ├─ Update state:
    │     SimpleSSM:  s = A*s + B*xt
    │     SSM:        h = A*h + B(xt)
    ├─ Compute output:
    │     SimpleSSM:  yt = C*s + D*xt
    │     SSM:        y = C(h)
    └─ Append output to list
    ↓
Stack outputs → y (B, T, D)
    ↓
(Optional) LayerNorm(y) [SimpleSSM only]
    ↓
Return y


"""

import torch
import torch.nn as nn

class SimpleSSM(nn.Module):
    """
    Real State Space Model (causal, recurrent)

    Discrete-time State Space Model:
        h_t = A h_{t-1} + B x_t
        y_t = C h_t + D x_t

        True diagonal SSM
        Linear time in sequence length
        Stable & parallelizable
        Exactly how Mamba works internally
        complexity O(TD)
    """

    def __init__(self, d_model, use_nonlinearity=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)


        # Diagonal SSM parameters
        self.A = nn.Parameter(torch.randn(d_model))
        self.B = nn.Parameter(torch.randn(d_model))
        self.C = nn.Parameter(torch.randn(d_model))
        self.D = nn.Parameter(torch.randn(d_model))

        self.norm = nn.LayerNorm(d_model)
        self.use_nonlinearity = use_nonlinearity
        self.act = nn.GELU() if use_nonlinearity else nn.Identity() # not to limit for complex ME patterns
        print("Initialized SimpleSSM",d_model)
        

    def forward(self, x):
        #        x: (batch, seq_len, d_model)
        
        B, T, D = x.shape
        s = torch.zeros(B, D, device=x.device)
        outputs = []

        for t in range(T):
            xt = x[:, t, :]

            # STATE UPDATE (this is the missing part before)
            s = self.A * s + self.B * xt

            # OUTPUT
            yt = self.C * s + self.D * xt
            outputs.append(yt)
            #print(f"SSM timestep {t}, output shape: {yt.shape}",outputs.__len__)
            
        #print(f"SSM final output shape: {outputs[0].shape} yt {yt.shape}")
        #return torch.stack(outputs, dim=1)
        y = torch.stack(outputs, dim=1)
        return self.norm(y)

    

"""# quick/sanity test
if __name__ == "__main__":
    x = torch.randn(2, 10, 64)
    model = SimpleSSM(64)

    y = model(x)
    print(y.shape)
"""

class SSM(nn.Module):

    def __init__(self, d_model, use_nonlinearity=True):
        super().__init__()
        self.d_model = d_model

        # Learnable state parameters
        self.A = nn.Parameter(torch.randn(d_model))
        self.B = nn.Linear(d_model, d_model)
        self.C = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.use_nonlinearity = use_nonlinearity
        self.act = nn.GELU() if use_nonlinearity else nn.Identity()

    def forward(self, x):
        
        #x: (B, T, D)
        
        B, T, D = x.shape
        h = torch.zeros(B, D, device=x.device)

        outputs = []

        for t in range(T):
            #  RECURRENCE (this is what makes it an SSM)
            h = self.A * h + self.B(x[:, t])
            y = self.C(h)
            outputs.append(y)

        return torch.stack(outputs, dim=1)