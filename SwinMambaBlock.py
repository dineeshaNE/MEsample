import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Simple Mamba (placeholder SSM)
# -------------------------------
class SimpleMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, dim))
        self.B = nn.Parameter(torch.randn(dim, dim))
        self.C = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        # x: (B, L, C)
        h = torch.zeros_like(x)
        for t in range(x.size(1)):
            prev = h[:, t-1] if t > 0 else 0
            h[:, t] = x[:, t] @ self.B + prev @ self.A
        return h @ self.C


# -------------------------------
# Swin-Mamba Block
# -------------------------------
class SwinMambaBlock(nn.Module):
    def __init__(self, dim, window_size=7, shift=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.mamba = SimpleMamba(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )

    def window_partition(self, x):
        B, H, W, C = x.shape
        ws = self.window_size
        x = x.view(B, H//ws, ws, W//ws, ws, C)
        x = x.permute(0,1,3,2,4,5).contiguous()
        return x.view(-1, ws*ws, C)

    def window_reverse(self, windows, H, W):
        ws = self.window_size
        B = int(windows.shape[0] / (H*W / ws / ws))
        x = windows.view(B, H//ws, W//ws, ws, ws, -1)
        x = x.permute(0,1,3,2,4,5).contiguous()
        return x.view(B, H, W, -1)

    def forward(self, x):
        B, H, W, C = x.shape

        if self.shift:
            x = torch.roll(x, shifts=(-self.window_size//2, -self.window_size//2), dims=(1,2))

        windows = self.window_partition(x)
        windows = self.norm1(windows)
        windows = self.mamba(windows)
        x = self.window_reverse(windows, H, W)

        if self.shift:
            x = torch.roll(x, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))

        x = x + self.mlp(self.norm2(x))
        return x
