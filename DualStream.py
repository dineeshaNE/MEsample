class DualStreamEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()

        self.rgb = FrameEncoder(out_dim//2)
        self.flow = FrameEncoder(out_dim//2)

    def forward(self, frames, flows):
        f1 = self.rgb(frames)     # (B,T,D/2)
        f2 = self.flow(flows)     # (B,T,D/2)
        return torch.cat([f1, f2], dim=-1)  # (B,T,D)
#--------------------------------------------

class MERModel(nn.Module):
    def __init__(self, d_model=64, num_classes=5):
        super().__init__()
        self.encoder = DualStreamEncoder(d_model)
        self.mamba = MambaClassifier(d_model, num_classes)

    def forward(self, frames, flows):
        x = self.encoder(frames, flows)
        return self.mamba(x)

#--------------------------------------------
for frames, flows, labels in loader:
    logits = model(frames, flows)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
