# with differnt learnining rates

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classifier import MambaClassifier
from dataset import ToySequenceDataset   # replace later with CASME dataset

# -------------------------
# Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = ToySequenceDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = MambaClassifier(d_model=64, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()

# -------------------------
# Phase 1 — Warm-up
# -------------------------
print("\n🧊 Phase 1: Freezing backbone")

for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

warmup_epochs = 5

for epoch in range(warmup_epochs):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Warm-up] Epoch {epoch+1}/{warmup_epochs} — Loss: {total_loss/len(loader):.4f}")

# -------------------------
# Phase 2 — Unfreeze
# -------------------------
print("\n🔥 Phase 2: Unfreezing backbone")

for param in model.backbone.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {"params": model.backbone.parameters(), "lr": 1e-4},
    {"params": model.classifier.parameters(), "lr": 1e-3},
])

finetune_epochs = 15

for epoch in range(finetune_epochs):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Fine-tune] Epoch {epoch+1}/{finetune_epochs} — Loss: {total_loss/len(loader):.4f}")

print("\n✅ Training complete.")
