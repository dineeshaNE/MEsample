#--------------------------
# ColabSwinMamba
#--------------------------

from pyexpat import model
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd

import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

from torchvision import transforms

from sklearn.metrics import classification_report, confusion_matrix, f1_score

#--------------------------
# CASME2 Dataset    
#--------------------------

class CASME2Dataset(Dataset):
    def __init__(self, root, annotation_file, transform=None, T=30, limit=None):
        self.root = root
        #self.ann = pd.read_excel(annotation_file)
        self.transform = transform
        self.T = T

        if annotation_file.endswith(".csv"):
            self.ann = pd.read_csv(annotation_file)
        #elif annotation_file.endswith(".xlsx"):
            #self.ann = pd.read_excel(annotation_file)
        else:
            raise ValueError("Annotation file must be .csv or .xlsx")

        #print(self.ann.columns)

        if limit is not None:
            self.ann = self.ann.iloc[:limit].reset_index(drop=True)

        self.label_map = {
            'happiness': 0,
            'disgust': 1,
            'surprise': 2,
            'repression': 3,
            'fear': 4,
            'sadness': 5,
            'others': 6
        }

    def _format_subject(self, subject):
        return f"sub{int(subject):02d}"

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        row = self.ann.iloc[idx]
        #print  (f"Processing row {idx}")
        #print(f"Processing row {idx}: {row.to_dict()}")

        subject = row['Subject']
        video = row['Filename']

        #label = int(row['Label'])
        emotion = row['Estimated Emotion'].strip().lower()
        label = self.label_map[emotion]
        if emotion not in self.label_map:
            raise ValueError(f"Unknown emotion: {emotion}")


        #subject = self._format_subject(subject)
        clip_dir = os.path.join(self.root, f"sub{int(subject):02d}", video)
        frames = sorted(os.listdir(clip_dir))

        images = []
        for f in frames:
            img = cv2.imread(os.path.join(clip_dir, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img) #(3, 224, 224)
            images.append(img)

        x = torch.stack(images)   # (T, C, H,  W)
        #print(f"Frames done: {clip_dir}")

        # Compute simple motion magnitude  for strong, learnable temporal signature
        diffs = (x[1:] - x[:-1]).abs().mean(dim=(1,2,3)) #peaks near the apex
        scores = torch.cat([diffs[:1], diffs])  # align length

        # Normalize
        scores = scores / (scores.sum() + 1e-6) # avoid div by zero in no motion clips


        # Weighted temporal sampling
        indices = torch.multinomial(scores, self.T, replacement=True)
        indices = torch.sort(indices).values
        x = x[indices]


        # Temporal normalization for robust batch processing


        if x.shape[0] < self.T:
            pad = self.T - x.shape[0]
            x = torch.cat([x, x[-1:].repeat(pad,1,1,1)])


        return x, label

# -------------------------------
# Mamba instead in ssm_mamba
# -------------------------------
class Mamba(nn.Module):
    """
    Real State Space Model (causal, recurrent)

    Discrete-time State Space Model:
        h_t = A h_{t-1} + B x_t
        y_t = C h_t + D x_t

        True diagonal SSM
        Linear time in sequence length
        Stable & parallelizable
        Exactly how Mamba works internally
        complexity O(TD)
    """

    def __init__(self, d_model, use_nonlinearity=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)


        # Diagonal SSM parameters
        self.A = nn.Parameter(torch.randn(d_model))
        A = torch.tanh(self.A)
        self.B = nn.Parameter(torch.randn(d_model))
        self.C = nn.Parameter(torch.randn(d_model))
        self.D = nn.Parameter(torch.randn(d_model))

        self.norm = nn.LayerNorm(d_model)
        self.use_nonlinearity = use_nonlinearity
        self.act = nn.GELU() if use_nonlinearity else nn.Identity() # not to limit for complex ME patterns
        #print("Initialized SimpleSSM",d_model)


    def forward(self, x):
        #        x: (batch, seq_len, d_model)

        B, T, D = x.shape
        s = torch.zeros(B, D, device=x.device)
        outputs = []

        for t in range(T):
            xt = x[:, t, :]

            # STATE UPDATE (this is the missing part before)
            s = self.A * s + self.B * xt

            # OUTPUT
            yt = self.C * s + self.D * xt
            outputs.append(yt)
            #print(f"SSM timestep {t}, output shape: {yt.shape}",outputs.__len__)

        #print(f"SSM final output shape: {outputs[0].shape} yt {yt.shape}")
        #return torch.stack(outputs, dim=1)
        y = torch.stack(outputs, dim=1)
        return self.norm(y)

#-------------------------------
# OrthogonalSpatial Mamba Modules
#-------------------------------

class OrthogonalSpatialMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ssm = Mamba(d_model=dim)

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape

        scans = []

        # 1. TL → BR
        s1 = x.reshape(B, H*W, C)
        scans.append(self.ssm(s1))

        # 2. BR → TL
        s2 = torch.flip(s1, dims=[1])
        scans.append(self.ssm(s2))

        # 3. TR → BL
        s3 = x.flip(2).reshape(B, H*W, C)
        scans.append(self.ssm(s3))

        # 4. BL → TR
        s4 = x.flip(1).reshape(B, H*W, C)
        scans.append(self.ssm(s4))

        # Fuse
        y = sum(scans) / len(scans)

        return y.reshape(B, H, W, C)
    
    # -------------------------------
    # Window Orthogonal Mamba
    # -------------------------------
    
class WindowOrthogonalMamba(nn.Module):
    def __init__(self, dim, window_size):
        super().__init__()
        self.ssm = Mamba(d_model=dim)
        self.ws = window_size

    def forward(self, w):
        # w: (Bwin, ws*ws, C)
        B, L, C = w.shape
        ws = self.ws

        w = w.view(B, ws, ws, C)

        s1 = w.reshape(B, L, C)
        y1 = self.ssm(s1)

        s2 = torch.flip(s1, dims=[1])
        y2 = self.ssm(s2)

        s3 = w.flip(2).reshape(B, L, C)
        y3 = self.ssm(s3)

        s4 = w.flip(1).reshape(B, L, C)
        y4 = self.ssm(s4)

        y = (y1 + y2 + y3 + y4) / 4
        return y.view(B, L, C)


# -------------------------------
# Swin-Mamba Block
# -------------------------------
class SwinMambaBlock(nn.Module):
    def __init__(self, dim, window_size=7, shift=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        #self.mamba = SimpleMamba(dim)
        #self.mamba = Mamba(dim)
        #self.spatial_mamba = OrthogonalSpatialMamba(dim)
        self.spatial_mamba = WindowOrthogonalMamba(dim, window_size)

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )

    def window_partition(self, x):
        B, H, W, C = x.shape
        ws = self.window_size
        x = x.view(B, H//ws, ws, W//ws, ws, C)
        x = x.permute(0,1,3,2,4,5).contiguous()
        return x.view(-1, ws*ws, C)

    def window_reverse(self, windows, H, W):
        ws = self.window_size
        B = int(windows.shape[0] / (H*W / ws / ws))
        x = windows.view(B, H//ws, W//ws, ws, ws, -1)
        x = x.permute(0,1,3,2,4,5).contiguous()
        return x.view(B, H, W, -1)

    def forward(self, x):
        B, H, W, C = x.shape

        if self.shift:
            x = torch.roll(x, shifts=(-self.window_size//2, -self.window_size//2), dims=(1,2))

        windows = self.window_partition(x)
        windows = self.norm1(windows)
        #windows = self.mamba(windows)
        #windows = self.ssm(windows)
        windows =self.spatial_mamba(windows)
        x = self.window_reverse(windows, H, W)

        if self.shift:
            x = torch.roll(x, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))

        x = x + self.mlp(self.norm2(x))
        return x

# -------------------------------
# Temporal Mamba
#-------------------------------

class TemporalMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(dim)

    def forward(self, x):
        # x: (B, T, C)
        x = self.norm(x)
        #x = self.mamba(x)
        y_fwd = self.mamba(x)
        y_bwd = self.mamba(torch.flip(x, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])

        return self.alpha * y_fwd + (1 - self.alpha) * y_bwd
        #return x
    
# -------------------------------
# Bidirectional Temporal Mamba
#-------------------------------    
class BidirectionalTemporalMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ssm = Mamba(d_model=dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: (B, T, C)

        y_fwd = self.ssm(x)
        y_bwd = self.ssm(torch.flip(x, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])

        return self.alpha * y_fwd + (1 - self.alpha) * y_bwd

    
    
# -------------------------------
# Patch Embedding
# -------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)                 # B, C, H/P, W/P
        x = x.permute(0,2,3,1)           # B, H, W, C
        return x


# -------------------------------
# Patch Merging (Downsample)
# -------------------------------
class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4*dim, 2*dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2]
        x1 = x[:, 1::2, 0::2]
        x2 = x[:, 0::2, 1::2]
        x3 = x[:, 1::2, 1::2]
        x = torch.cat([x0,x1,x2,x3], -1)
        return self.reduction(x)


# -------------------------------
# Swin-Mamba Stage
# -------------------------------
class SwinMambaStage(nn.Module):
    '''
    def __init__(self, dim, depth, window):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinMambaBlock(dim, window, shift=(i%2==1))
            for i in range(depth)
        ])'''

    def __init__(self, dim, depth, window_size=7):
        super().__init__()
        self.blocks = nn.ModuleList()

        for i in range(depth):
            self.blocks.append(
                SwinMambaBlock(
                    dim=dim,
                    window_size=window_size,
                    shift=(i % 2 == 1)
                )
            )
            

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# -------------------------------
# Full Backbone
#
# -------------------------------
class SwinMamba(nn.Module):
    def __init__(self, in_ch=3, out_dim=512):
        super().__init__()

        self.patch = PatchEmbed(in_ch, 64)

        self.stage1 = SwinMambaStage(64, 2, 7) # window size 7
        self.merge1 = PatchMerging(64)

        self.stage2 = SwinMambaStage(128, 2, 7)
        self.merge2 = PatchMerging(128)

        self.stage3 = SwinMambaStage(256, 6, 7)
        self.merge3 = PatchMerging(256)

        self.stage4 = SwinMambaStage(512, 2, 7)

        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        x = self.patch(x)

        x = self.stage1(x)
        x = self.merge1(x)

        x = self.stage2(x)
        x = self.merge2(x)

        x = self.stage3(x)
        x = self.merge3(x)

        x = self.stage4(x)

        B, H, W, C = x.shape
        x = x.view(B, H*W, C)
        x = x.mean(dim=1)     # global spatial pooling
        return self.head(x)

# -------------------------------
# Full Video Swin-Mamba for MER
# -------------------------------

class VideoSwinMamba(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Spatial Mamba (orthogonal scans)
        # Bidirectional temporal SSM
        self.backbone = SwinMamba(in_ch=3, out_dim=512)
        self.temporal = BidirectionalTemporalMamba(512) #TemporalMamba(512)
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



mytransforms = transforms.Compose([

    # 1️⃣ Resize all faces to a fixed size
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),

    # 2️⃣ Convert to grayscale (optional but strongly recommended for MER)
    transforms.Grayscale(num_output_channels=3),

    # 3️⃣ Data normalization
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),

    # 4️⃣ Micro-expression friendly augmentation (training only)
    transforms.RandomHorizontalFlip(p=0.5),
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3),
])

# -------------------------------
# Main Training Loop
# -------------------------------

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

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for sending data and model to the correct device

    model = VideoSwinMamba(num_classes=7).to(device)

    # Handle class imbalance with weighted loss - higher weight to minority classes
    labels = dataset.ann['Estimated Emotion'].str.lower().map(dataset.label_map).values
    class_counts = torch.bincount(torch.tensor(labels))
    weights = 1.0 / (class_counts.float() + 1e-6)
    weights = weights / weights.sum() * len(class_counts)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
    {"params": model.backbone.parameters(), "lr": 1e-5},
    {"params": model.temporal.parameters(), "lr": 1e-3}
])
    #  gradually lowers the learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

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

    log = []

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # avoid exploding gradients
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
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
)
      log.append([epoch, train_loss, train_acc, val_loss, val_acc])
      
      scheduler.step()


      # freeze the best model
      # val accuracy not to overfit the model training acc → memorization, validation acc → generalization, test acc → final report
      if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch= epoch
            torch.save(model.state_dict(), "best_model.pth")

    print("Training & Validation step OK")
    log_df = pd.DataFrame(log, columns=[
    "epoch", "train_loss", "train_acc", "val_loss", "val_acc"
])

    log_df.to_csv("training_log.csv", index=False)

    # ========== TEST ==========
    
    # reload the best model

    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()

    test_correct, test_total = 0, 0
    all_preds = []
    all_gt = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_gt.extend(y.cpu().numpy())

#from sklearn.metrics import classification_report, confusion_matrix, f1_score

    test_acc = 100 * test_correct / test_total
    print(f"🧪 Final Test Accuracy: {test_acc:.2f}%")


    print(classification_report(all_gt, all_preds, digits=4, zero_division=0))
    print("Micro-F1:", f1_score(all_gt, all_preds, average='micro', zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(all_gt, all_preds))

    # ---------- Save evaluation results ----------
    report = classification_report(all_gt, all_preds, digits=4, output_dict=True)
    cm = confusion_matrix(all_gt, all_preds)

    pd.DataFrame(report).transpose().to_csv("classification_report.csv")

    pd.DataFrame(cm).to_csv("confusion_matrix.csv", index=False)

    print("📁 Saved: classification_report.csv & confusion_matrix.csv")

\
    print("✅ Experiment finished")


if __name__ == "__main__":
  import traceback
try:
    main()
except Exception as e:
    print("Exception in main:", repr(e))
    traceback.print_exc()
    raise
 
