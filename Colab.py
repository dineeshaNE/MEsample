import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiDirectionalMamba(nn.Module):
    """
    PyTorch replacement for Mamba-SSM with 4 directional scans:
        - TL→BR (top-left → bottom-right)  : row-major
        - TR→BL (top-right → bottom-left)  : reversed row-major
        - BL→TR (bottom-left → top-right)  : column-major reversed
        - BR→TL (bottom-right → top-left)  : column-major
    Supports fusion (average) of all directions.
    """
    def __init__(self, d_model=64, directions=None):
        super().__init__()
        self.d_model = d_model
        if directions is None:
            # default to all four directions
            self.directions = ["TLBR", "TRBL", "BLTR", "BRTL"]
        else:
            self.directions = directions

        # Input projection
        self.in_proj = nn.Linear(d_model, d_model)
        # 1D Conv for temporal/local patterns
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.GELU()
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def scan(self, x, direction):
        """
        x: (B, T, D)
        direction: one of ["TLBR", "TRBL", "BLTR", "BRTL"]
        """
        # Input projection
        x = self.in_proj(x)

        # Convert to (B, D, T) for Conv1D
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)

        # Apply directional flipping
        if direction == "TRBL":
            x = torch.flip(x, dims=[1])
        elif direction == "BLTR":
            x = torch.flip(x, dims=[0])  # simulate bottom start
        elif direction == "BRTL":
            x = torch.flip(x, dims=[0, 1])  # flip both batch and time

        return x

    def forward(self, x):
        """
        x: (B, T, D)
        Returns the **fused output** across all directions
        """
        outputs = []
        for dir in self.directions:
            outputs.append(self.scan(x, dir))
        # Average fusion
        out = torch.stack(outputs, dim=0).mean(dim=0)
        out = self.out_proj(out)
        return out

# Example
batch_size = 2
T = 30       # number of frames
d_model = 64

x = torch.randn(batch_size, T, d_model)  # input sequence features

mamba_block = MultiDirectionalMamba(d_model=d_model)
out = mamba_block(x)
print(out.shape)  # should be (B, T, D)
