
##### main.py##########################################################################
"""
main
 └──Build dataset
      CASME2Dataset.__init__
      DataLoader.__init__
 └── Build model
      MERModel.__init__
      CrossEntropyLoss.__init__
      Adam.__init__
 └── Training & Validation
     TRAIN on train_loader
     VALIDATE on val_loader
      for epoch
       └── DataLoader.__iter__ → __next__ → CASME2Dataset.__getitem__
       └── MERModel.forward → 
       └── CrossEntropyLoss.forward → loss
       └── optimizer.zero_grad
       └── loss.backward
       └── Adam.step
 └── Testing & Evaluations
      after all epochs:
      LOAD BEST MODEL
      TEST on test_loader   ← only once, at the end


 """
################################################################################

from pyexpat import model
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from CASME2Dataset import CASME2Dataset
#from MERModel import MERModel
#from classifier import MambaClassifier
#from MambaClassifier import MambaClassifier
#from dataset import ToySequenceDataset
from transforms import mytransforms
import pandas as pd
from torch.utils.data import random_split
from VideoSwinMamba import VideoSwinMamba

def main():


    #dataset = ToySequenceDataset()
    dataset = CASME2Dataset(
    root="CASME2/raw",
    annotation_file="CASME2/CASME2.csv",
    transform=mytransforms,
    T=30,
    limit = 20
)
    
    N = len(dataset)
    train_len = int(0.7 * N)
    val_len   = int(0.15 * N)
    test_len  = N - train_len - val_len
    
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    #print(f"Dataset length: {len(dataset)}")
    #loader = DataLoader(dataset, batch_size=4, shuffle=True)
    #print(f"DataLoader created with batch size 4")
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=4, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=4, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for sending data and model to the correct device

    #model = MambaClassifier(64,2) with toy version
    #model = MERModel(64, 7).to(device) # 7 emotion classes
    model = VideoSwinMamba(num_classes=7).to(device)


    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # with CASME2
    """this often gives 10–25% accuracy gains on small MER datasets
    The backbone already knows useful visual dynamics
    The classifier is newly initialized and must learn fast"""

    optimizer = torch.optim.Adam([
    {"params": model.backbone.parameters(), "lr": 1e-5},
    {"params": model.temporal.parameters(), "lr": 1e-3}
])
    print("Model and components initialized.")

    best_val_acc = 0.0
    best_epoch = 0

      # quick test run to verify dimensions
    x, y = next(iter(train_loader))
    x = x.to(device)

    with torch.no_grad():
      out = model(x)

    print("Input:", x.shape)
    print("Output:", out.shape)


       
     #training loop
    for epoch in range(2):
      # ========== TRAIN ==========
      model.train()
      train_loss, train_correct, train_total = 0, 0, 0

      for x, y in train_loader:
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
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            train_loss += loss.item()

            print(f"Batch Loss: {loss.item():.4f} | Acc: {batch_acc:.2%}")
      train_acc = 100 * train_correct / train_total
      train_loss /= len(train_loader)

      # ========== VALIDATE ==========
      model.eval()
      val_loss, val_correct, val_total = 0, 0, 0

      with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

      val_acc = 100 * val_correct / val_total
      val_loss /= len(val_loader)


      # ===== compute epoch metrics =====

      print(f"Epoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")     
      
      
      # freeze the best model
      # val accuracy not to overfit the model training acc → memorization, validation acc → generalization, test acc → final report
      if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch= epoch
            torch.save(model.state_dict(), "best_model.pth")

            '''torch.save({                  'epoch': best_epoch,
                  'model_state': model.state_dict(),
                  'optimizer_state': optimizer.state_dict(),
                  'best_acc': best_val_acc
            }, "best_model.pth")            
            print("🔥 Best model saved at epoch {best_epoch} with acc = {best_val_acc:.2f}%")'''
               
    #print("x min/max:", x.min().item(), x.max().item())
    #print("logits min/max:", logits.min().item(), logits.max().item())
    print("Training & Validation step OK")

    # ========== TEST ==========
    # reload the best model
    '''
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print(f" Best model restored from epoch {checkpoint['epoch']} with acc = {checkpoint['best_acc']:.2f}%")
    '''      
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()

    test_correct, test_total = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)

    test_acc = 100 * test_correct / test_total
    print(f"🧪 Final Test Accuracy: {test_acc:.2f}%")

\
    print("✅ Experiment finished")

'''
    If train acc is low → model underfitting
    If train acc is high and val acc is low → overfitting
    If both are low → architecture or data problem
    '''

      
    


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("Exception in main:", repr(e))
        traceback.print_exc()
        raise
