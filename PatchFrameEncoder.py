import torch
import torch.nn as nn
from torchvision import models, transforms

class PatchExtractor(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        Ph, Pw = self.patch_size

        assert H % Ph == 0 and W % Pw == 0

        x = x.view(B, T, C, H//Ph, Ph, W//Pw, Pw)
        x = x.permute(0, 1, 3, 5, 2, 4, 6)
        x = x.reshape(B, T, -1, C, Ph, Pw)

        return x

class PatchFrameEncoder(nn.Module):
    def __init__(self, out_dim=64, patch_size=(32, 32)):
        super().__init__()

        self.patchify = PatchExtractor(patch_size)

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(512, out_dim)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        x = self.patchify(x)
        # (B, T, N, C, Ph, Pw)

        B, T, N, C, Ph, Pw = x.shape
        x = x.view(B*T*N, C, Ph, Pw)

        x = torch.stack([self.normalize(p) for p in x])

        f = self.features(x).flatten(1)
        f = self.fc(f)

        return f.view(B, T, N, -1)


class FrameEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        #backbone = models.resnet18(weights="IMAGENET1K_V1")
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) #pretrained weights expect inputs normalized with ImageNet stats:
        self.features = nn.Sequential(*list(backbone.children())[:-1]) #Removes the final classification layer
        self.fc = nn.Linear(512, out_dim)

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        print("Initialized FrameEncoder",out_dim,self.features,self.fc)

    def forward(self, x):
        # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        # Normalize
        x = torch.stack([self.normalize(frame) for frame in x])

        f = self.features(x).flatten(1) #ResNet’s last conv layer output is (B*T, 512, 1, 1) and flattening gives (B*T, 512)
        f = self.fc(f)
        return f.view(B, T, -1)
