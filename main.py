import torch, torch.nn as nn
from torch.utils.data import DataLoader
from classifier import MambaClassifier
from dataset import ToySequenceDataset



def main():

    dataset = ToySequenceDataset()
    #print(f"Dataset length: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    #print(f"DataLoader created with batch size 8")

    ''' model = SimpleSSM(64)
    classifier = nn.Linear(64, 2)'''

    #print("Initializing model...MambaClassifier")
    model = MambaClassifier(64,2)
    print("Model and components initialized.")
    

    criterion = nn.CrossEntropyLoss()
    print(f"Criterion: {criterion}")

    optimizer = torch.optim.Adam(
        # to avoid duplicates comment the below line
        # list(model.parameters())  + list(model.classifier.parameters()),
        list(model.parameters()) ,
        lr=1e-3
        #Learning rate = step size. 1e-3 = 0.001
        #input → logits → loss → gradients → parameter update
    )    
    """  this often gives 10–25% accuracy gains on small MER datasets.        
    The backbone already knows useful visual dynamics
    The classifier is newly initialized and must learn fast
    {"params": model.backbone.parameters(), "lr": 1e-4},
    {"params": model.classifier.parameters(), "lr": 1e-3},"""
           
    
    #print(f"Optimizer: {optimizer}")
            

# training step
    for x, y in loader:#
        print   (f"Input batch shape: {x.shape}")

        #out = model(x)              # x (B, T, D)
        logits = model(x)
        print(f"Model output shape: {logits.shape} {logits}")
            
        #out = out[:, -1, :]         # last timeste
        #logits already done with model
        ##logits = model.classifier(out)
        #logits = model prediction scores

        # correct the class

        loss = criterion(logits, y)
        #loss = error between logits and correct class
        loss.backward()
        #parameter.grad = ∂loss / ∂parameter
        optimizer.step()
        # parameter = parameter - learning_rate * gradient (with Adam magic)
        optimizer.zero_grad()
        # SimpleSSM → hidden states → last timestep → classifier → logits

    print("Training step OK")

    """ training loop
    for epoch in range(10):
    for x, y in loader:
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    
      print("Training step OK")"""

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("Exception in main:", repr(e))
        traceback.print_exc()
        raise
"""if __name__ == "__main__":
    print("mamba.py loaded")
    m = Mamba(64)
    x = torch.randn(2, 10, 64)
    print(m(x).shape)"""