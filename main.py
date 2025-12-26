import torch, torch.nn as nn
from torch.utils.data import DataLoader
from CASME2Dataset import CASME2Dataset
from MERModel import MERModel
from classifier import MambaClassifier
from dataset import ToySequenceDataset
from transforms import mytransforms

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

def main():

    #dataset = ToySequenceDataset()
    dataset = CASME2Dataset(
    root="CASME2/raw",
    annotation_file="CASME2/annotations.xlsx",
    transform=mytransforms,
    T=30
)
    #print(f"Dataset length: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    #print(f"DataLoader created with batch size 8")


    #print("Initializing model...MambaClassifier")
    model = MambaClassifier(64,2)
    model = MERModel(64, 5)
    print("Model and components initialized.")
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # with CASME2

    """ with toy
    optimizer = torch.optim.Adam(
        # to avoid duplicates comment the below line
        # list(model.parameters())  + list(model.classifier.parameters()),
        list(model.parameters()) ,
        lr=1e-3
        #Learning rate = step size. 1e-3 = 0.001
        #input → logits → loss → gradients → parameter update
    )   
    """ 
    """  this often gives 10–25% accuracy gains on small MER datasets.        
    The backbone already knows useful visual dynamics
    The classifier is newly initialized and must learn fast
    {"params": model.backbone.parameters(), "lr": 1e-4},
    {"params": model.classifier.parameters(), "lr": 1e-3},"""
           
    #print(f"Optimizer: {optimizer}")
            

     #training loop
    for epoch in range(10):
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    
    print("Training step OK")
    
    

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("Exception in main:", repr(e))
        traceback.print_exc()
        raise
