import torch, torch.nn as nn
from torch.utils.data import DataLoader
from CASME2Dataset import CASME2Dataset
from MERModel import MERModel
from classifier import MambaClassifier
#from dataset import ToySequenceDataset
from transforms import mytransforms
import pandas as pd

"""
CASME II video clip
      │
      ▼
Face Detection + Alignment
      │
      ▼
Frame Encoder (CNN / ViT patch embed)
      │
      ▼
Sequence Tensor x (B,T,D)
      │
      ▼
     Mamba
      │
      ▼
Hidden states h (B,T,D)
      │
      ▼
Select last timestep
      │
      ▼
Temporal summary h_last (B,D)
      │
      ▼
Linear classifier
      │
      ▼
Emotion logits (B,C)
      │
      ▼
CrossEntropyLoss


Video → (B,T,C,H,W)
→ CNN → (B,T,64)
→ Mamba → (B,T,64)
→ Last timestep → (B,64)
→ Linear → (B,Classes)
→ Loss
"""
##############################################################################
"""
main
 └── CASME2Dataset.__init__
 └── DataLoader.__init__
 └── MERModel.__init__
 └── CrossEntropyLoss.__init__
 └── Adam.__init__
 └── for epoch
       └── DataLoader.__iter__ → __next__ → CASME2Dataset.__getitem__
       └── MERModel.forward → MambaClassifier.forward → SimpleMamba.forward → return logits
       └── CrossEntropyLoss.forward → loss
       └── optimizer.zero_grad
       └── loss.backward
       └── Adam.step
 └── print(stats)

 """
################################################################################

def main():


    #dataset = ToySequenceDataset()
    dataset = CASME2Dataset(
    root="CASME2/raw",
    annotation_file="CASME2/annotations.csv",
    transform=mytransforms,
    T=30,
    limit = 20
)
    #print(f"Dataset length: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    #print(f"DataLoader created with batch size 4")


    #print("Initializing model...MambaClassifier")
    #model = MambaClassifier(64,2) with toy version
    model = MERModel(64, 7) # emotion classes
    print("Model and components initialized.")
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # with CASME2

    """  this often gives 10–25% accuracy gains on small MER datasets.        
    The backbone already knows useful visual dynamics
    The classifier is newly initialized and must learn fast
    {"params": model.backbone.parameters(), "lr": 1e-4},
    {"params": model.classifier.parameters(), "lr": 1e-3},"""
           

     #training loop
    for epoch in range(2):
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    
    print("x min/max:", x.min().item(), x.max().item())
    print("logits min/max:", logits.min().item(), logits.max().item())

    print("Training step OK")
    
    

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("Exception in main:", repr(e))
        traceback.print_exc()
        raise
