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

    model = MambaClassifier(64,2)
    #print(f"Model initialized: {model}")


    criterion = nn.CrossEntropyLoss()
    #print(f"Criterion: {criterion}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(model.classifier.parameters()),
        lr=1e-3
        #Learning rate = step size. 1e-3 = 0.001
        #input → logits → loss → gradients → parameter update
        
              
    )
    #print(f"Optimizer: {optimizer}")
            

# training step
    for x, y in loader:
        out = model(x)              # (B, T, D)
        out = out[:, -1, :]         # last timestep
        logits = model.classifier(out)
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
