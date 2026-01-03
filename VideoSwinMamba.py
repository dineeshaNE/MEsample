from SwinMamba import SwinMamba
from SwinMambaBlock import SimpleMamba

class VideoSwinMamba(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.backbone = SwinMamba(in_ch=3, num_classes=512)
        self.temporal = TemporalMamba(512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        feats = []
        for t in range(T):
            f = self.backbone(x[:, t])   # (B, 512)
            feats.append(f)

        feats = torch.stack(feats, dim=1)  # (B, T, 512)

        feats = self.temporal(feats)       # Temporal MER modeling

        return self.classifier(feats.mean(dim=1))


class TemporalMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = SimpleMamba(dim)

    def forward(self, x):
        # x: (B, T, C)
        x = self.norm(x)
        x = self.mamba(x)
        return x

model = VideoSwinMamba(num_classes=7)
x = torch.randn(2, 30, 3, 224, 224)
y = model(x)
print(y.shape)   # (2, 7)
