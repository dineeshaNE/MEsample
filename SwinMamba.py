import torch
import torch.nn as nn
from SwinMambaBlock import SwinMambaBlock

# -------------------------------
# Patch Embedding
# -------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)                 # B, C, H/P, W/P
        x = x.permute(0,2,3,1)           # B, H, W, C
        return x


# -------------------------------
# Patch Merging (Downsample)
# -------------------------------
class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4*dim, 2*dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2]
        x1 = x[:, 1::2, 0::2]
        x2 = x[:, 0::2, 1::2]
        x3 = x[:, 1::2, 1::2]
        x = torch.cat([x0,x1,x2,x3], -1)
        return self.reduction(x)


# -------------------------------
# Swin-Mamba Stage
# -------------------------------
class SwinMambaStage(nn.Module):
    def __init__(self, dim, depth, window):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinMambaBlock(dim, window, shift=(i%2==1))
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# -------------------------------
# Full Backbone
# -------------------------------
class SwinMamba(nn.Module):
    def __init__(self, in_ch=3, num_classes=7):
        super().__init__()

        self.patch = PatchEmbed(in_ch, 64)

        self.stage1 = SwinMambaStage(64, 2, 7)
        self.merge1 = PatchMerging(64)

        self.stage2 = SwinMambaStage(128, 2, 7)
        self.merge2 = PatchMerging(128)

        self.stage3 = SwinMambaStage(256, 6, 7)
        self.merge3 = PatchMerging(256)

        self.stage4 = SwinMambaStage(512, 2, 7)

        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.patch(x)

        x = self.stage1(x)
        x = self.merge1(x)

        x = self.stage2(x)
        x = self.merge2(x)

        x = self.stage3(x)
        x = self.merge3(x)

        x = self.stage4(x)

        B, H, W, C = x.shape
        x = x.view(B, H*W, C)
        x = x.mean(dim=1)     # global spatial pooling
        return self.head(x)

    
model = SwinMamba(in_ch=3, num_classes=7)
x = torch.randn(2, 3, 224, 224)
y = model(x)
print(y.shape)   # (2, 7)
