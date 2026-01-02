from pyexpat import model
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from CASME2Dataset import CASME2Dataset
from MERModel import MERModel
#from classifier import MambaClassifier
from MambaClassifier import MambaClassifier
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
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for sending data and model to the correct device

    #print("Initializing model...MambaClassifier")
    #model = MambaClassifier(64,2) with toy version
    model = MERModel(64, 7).to(device) # 7 emotion classes
    print("Model and components initialized.")
    

    criterion = nn.CrossEntropyLoss()

    """this often gives 10–25% accuracy gains on small MER datasets
    The backbone already knows useful visual dynamics
    The classifier is newly initialized and must learn fast"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # with CASME2
    '''optimizer = torch.optim.Adam([
    {"params": model.encoder.parameters(), "lr": 1e-5},
    {"params": model.mamba.parameters(), "lr": 1e-3}
])'''

    best_acc = 0.0
    best_epoch = 0
    total_correct =0
    total_samples =0
    running_loss =0
       
     #training loop
    for epoch in range(2):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # inside batch loop (optional for debugging)
            preds = logits.argmax(dim=1)
            batch_acc = (preds == y).float().mean()

            # accumulate for epoch
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
            running_loss += loss.item()


        '''preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Acc: {acc:.2%}")'''

 
        # ===== compute epoch metrics =====
        acc = 100.0 * total_correct / total_samples
        avg_loss = running_loss / len(loader)

        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

        # freeze the best model=
        if not torch.isnan(torch.tensor(acc)) and acc > best_acc:
            best_acc = acc
            best_epoch = epoch

            torch.save({
                  'epoch': epoch,
                  'model_state': model.state_dict(),
                  'optimizer_state': optimizer.state_dict(),
                  'best_acc': best_acc
            }, "best_model.pth")

            print(f"🔥 New best model saved at epoch {epoch} with acc = {acc:.2f}%")

            
    print("x min/max:", x.min().item(), x.max().item())
    print("logits min/max:", logits.min().item(), logits.max().item())

    print("Training step OK")

    #reload the best model
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print(f"✅ Best model restored from epoch {checkpoint['epoch']} with acc = {checkpoint['best_acc']:.2f}%")

      
    

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("Exception in main:", repr(e))
        traceback.print_exc()
        raise
