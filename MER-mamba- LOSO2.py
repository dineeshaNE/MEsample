# SwOSMBTM using official mamba-ssm, with:

from mamba_ssm import Mamba

#--------------------------
# All-in-One SwOSMBTM implementation for Micro-Expression Recognition (MER)
#--------------------------

from datetime import datetime
from pyexpat import model
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import warnings
import csv

import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

from torchvision import transforms

# Import metrics and time
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import time

#for CUDA
from torch.amp import autocast, GradScaler


# Suppress the UndefinedMetricWarning
#warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._classification')

#------------------------------------
# Central config.py
#-----------------------------------
class Config:
    def __init__(self):
        # Dataset / IO
        self.DATA_ROOT = "collection/raw"
        self.ANNOTATION_FILE = "collection/CASME2.csv"
        self.LIMIT = None
        self.NUM_FRAMES = 16
        self.BATCH_SIZE = 4

        # Training
        self.LR = 1e-3
        self.EPOCHS = 2
        self.TRAIN_SPLIT = 0.7
        self.VAL_SPLIT = 0.15
        self.TEST_SPLIT = 0.15

        # Model
        self.NUM_CLASSES = 7
        self.EMBED_DIM = 512
        self.WINDOW_SIZE = 7
        self.BKBONE_LR = 1e-5
        self.TEMP_LR = 1e-3


        # Experiment
        self.EXP_NAME = "swin_mamba_experiment"

        # =====================
        # Spatial Ablation
        # =====================
        self.SPATIAL_TYPE = "window_mamba"
        # options:
        # "window_mamba"
        # "single_mamba"
        # "no_spatial"
        # "mlp"

        # =====================
        # Temporal Ablation
        # =====================
        self.TEMPORAL_TYPE = "bidirectional_mamba"
        # options:
        # "bidirectional_mamba"
        # "unidirectional_mamba"
        # "lstm"
        # "none"

        # =====================
        # Data Ablation
        # =====================
        self.USE_ALIGNMENT = False
        self.SAMPLING_TYPE = "motion"
        # options:
        # "motion"
        # "uniform"
        # "first"

        # =====================
        # Stability Ablation
        # =====================
        self.A_STABILITY = "exp"
        # options:
        # "exp"
        # "abs"
        # "raw"

    def __repr__(self): #print reports
            lines = ["\n===== Experiment Configuration ====="]
            for k, v in self.__dict__.items():
                lines.append(f"{k}: {v}")
            return "\n".join(lines)

cfg = Config()


#--------------------------
# CASME2 Dataset    
#--------------------------
import face_alignment
import numpy as np

class CASME2DatasetFA(Dataset):
    def __init__(self, root, annotation_file, transform=None, T=32, limit=None):
        self.root = root
        #self.ann = pd.read_excel(annotation_file)
        self.transform = transform
        self.T = T

        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        #If memory becomes tight, change to: device='cpu' Face alignment can stay on CPU to save GPU VRAM.

        # Directory to store alignment matrices
        self.cache_dir = os.path.join(self.root, "alignment_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        if annotation_file.endswith(".csv"):
            self.ann = pd.read_csv(annotation_file)
        else:
            raise ValueError("Annotation file must be .csv")

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
    def get_cache_path(self, subject, video):
        subject = f"sub{int(subject):02d}"
        filename = f"{subject}_{video}.npy"
        return os.path.join(self.cache_dir, filename)

    def compute_alignment_matrix(self, img):
        landmarks = self.fa.get_landmarks(img)

        if landmarks is None:
            return None

        lm = landmarks[0]

        # Eye centers (68-point model)
        left_eye = lm[36:42].mean(axis=0)
        right_eye = lm[42:48].mean(axis=0)

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        eyes_center = tuple(((left_eye + right_eye) / 2).astype(int))

        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

        return M

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

        cache_path = self.get_cache_path(subject, video)

        # 🔵 Try loading cached matrix
        if os.path.exists(cache_path):
            M = np.load(cache_path)
        else:
            M = None

        #label = int(row['Label'])
        emotion = row['Estimated Emotion'].strip().lower()
        label = self.label_map[emotion]
        if emotion not in self.label_map:
            raise ValueError(f"Unknown emotion: {emotion}")


        #subject = self._format_subject(subject)
        clip_dir = os.path.join(self.root, f"sub{int(subject):02d}", video)
        frames = sorted(os.listdir(clip_dir))

        images = []
        for i, f in enumerate(frames):
            img = cv2.imread(os.path.join(clip_dir, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #  Compute only once per clip
            if i == 0 and M is None:
                M = self.compute_alignment_matrix(img)

                if M is None:
                    # Fallback (identity transform)
                    M = np.eye(2, 3, dtype=np.float32)

                # 💾 Save permanently
                np.save(cache_path, M)

            # 🔵 Apply alignment
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            if self.transform:
                img = self.transform(img) #(3, 224, 224)
            images.append(img)

        x = torch.stack(images)   # (T, C, H,  W)
        #print(f"Frames done: {clip_dir}")

        '''
        # Compute simple motion magnitude  for strong, learnable temporal signature
        diffs = (x[1:] - x[:-1]).abs().mean(dim=(1,2,3)) #peaks near the apex
        scores = torch.cat([diffs[:1], diffs])  # align length
        # Normalize
        scores = scores / (scores.sum() + 1e-6) # avoid div by zero in no motion clips
        # Weighted temporal sampling
        indices = torch.multinomial(scores, self.T, replacement=True)
        indices = torch.sort(indices).values
        x = x[indices]
        '''

        #-----------------
        # Alternative sampling strategies for ablation
        #-----------------
        if cfg.SAMPLING_TYPE == "motion":
            # Compute simple motion magnitude  for strong, learnable temporal signature
            diffs = (x[1:] - x[:-1]).abs().mean(dim=(1,2,3))
            scores = torch.cat([diffs[:1], diffs])
            scores = scores / (scores.sum() + 1e-6)# Normalize to avoid div by zero in no motion clips
            indices = torch.multinomial(scores, self.T, replacement=True) # Weighted temporal sampling

        elif cfg.SAMPLING_TYPE == "uniform":
            indices = torch.linspace(0, x.shape[0]-1, self.T).long()

        elif cfg.SAMPLING_TYPE == "first":
            indices = torch.arange(min(self.T, x.shape[0]))

        x = x[indices]


        # Temporal normalization for robust batch processing


        if x.shape[0] < self.T:
            pad = self.T - x.shape[0]
            x = torch.cat([x, x[-1:].repeat(pad,1,1,1)])


        return x, label

#---------------------------------
#Dataset face Aligned
#----------------------------------
class CASME2Dataset(Dataset):
    def __init__(self, root, annotation_file, transform=None, T=32, limit=None):
        self.root = root
        #self.ann = pd.read_excel(annotation_file)
        self.transform = transform
        self.T = T

        if annotation_file.endswith(".csv"):
            self.ann = pd.read_csv(annotation_file)
        else:
            raise ValueError("Annotation file must be .csv")

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

#-------------
#  Spatial Module (Official Mamba)
#-------------

class WindowMamba(nn.Module):
    def __init__(self, dim, window_size):
        super().__init__()
        self.ws = window_size
        
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2
        )

    def forward(self, w):
        # w: (Bwin, ws*ws, C)
        return self.mamba(w)
    
# -------------------------
# Spatial Orthogonal Mamba
# -------------------------

class WindowMambaOrthogonal(nn.Module):
    def __init__(self, dim, window_size):
        super().__init__()
        self.ws = window_size

        # Official Mamba configuration
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2
        )

    def forward(self, w):
        """
        w: (Bwin, ws*ws, C)
        """

        B, L, C = w.shape
        ws = self.ws

        # Reshape to 2D window
        w2d = w.view(B, ws, ws, C)

        # ---- Direction 1: Normal raster scan ----
        s1 = w2d.reshape(B, L, C)
        y1 = self.mamba(s1)

        # ---- Direction 2: Reverse sequence ----
        s2 = torch.flip(s1, dims=[1])
        y2 = self.mamba(s2)

        # ---- Direction 3: Horizontal flip ----
        s3 = w2d.flip(2).reshape(B, L, C)
        y3 = self.mamba(s3)

        # ---- Direction 4: Vertical flip ----
        s4 = w2d.flip(1).reshape(B, L, C)
        y4 = self.mamba(s4)

        # Average all directions
        y = (y1 + y2 + y3 + y4) / 4

        return y
        
#---------------------------------
# Spacial Factory - Flexible spatial module builder for ablation
#----------------------------------
def build_spatial_module(dim, window_size, spatial_type):

    if spatial_type == "window_mamba":
        return WindowMambaOrthogonal(dim, window_size) #for MAMBA-SSM

    elif spatial_type == "single_mamba":
        #return Mamba(dim)
        return WindowMamba(dim, window_size)

    elif spatial_type == "mlp":
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    elif spatial_type == "no_spatial":
        return nn.Identity()

    else:
        raise ValueError("Unknown spatial type")

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
        #self.spatial_mamba = WindowOrthogonalMamba(dim, window_size)
        self.spatial_mamba = build_spatial_module(
            dim,
            window_size,
            cfg.SPATIAL_TYPE
        )

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

        #x = x + spatial_out
        x = x + self.spatial_mamba.norm1(windows) # for MAMBA-SSM, the spatial output is already added inside the module, so we can directly add the normalized input here for the residual connection. For other spatial types, this will just be an additional skip connection from the input to the MLP, which should not hurt performance and can even help stabilize training.
        #x = x + mlp_out
        x = x + self.mlp(self.norm2(x))
        return x

#-------------------------------
# Temporal Mamba (Unidirectional)
#-------------------------------

class TemporalMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2
        )

    def forward(self, x):
        return self.mamba(x)
        
# -------------------------------
# Bidirectional Temporal Mamba
#-------------------------------    
class BidirectionalTemporalMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        #self.ssm = TemporalSelectiveTrueMamba(dim)#FOR MAMBA-SSM
        self.mamba =Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: (B, T, C)

        y_fwd = self.mamba(x)
        y_bwd = self.mamba(torch.flip(x, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])

        return (self.self.alpha * y_fwd) + ((1 - self.alpha) * y_bwd)     

    
    
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

    def __init__(self, dim, depth, window_size=cfg.WINDOW_SIZE):
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
# -------------------------------
class SwinMamba(nn.Module):
    def __init__(self, in_ch=3, out_dim=512):
        super().__init__()
        self.window_size =cfg.WINDOW_SIZE

        self.patch = PatchEmbed(in_ch, 64)

        self.stage1 = SwinMambaStage(64, 2, self.window_size) # window size 7
        self.merge1 = PatchMerging(64)

        self.stage2 = SwinMambaStage(128, 2, self.window_size)
        self.merge2 = PatchMerging(128)

        self.stage3 = SwinMambaStage(256, 6, self.window_size)
        self.merge3 = PatchMerging(256)

        self.stage4 = SwinMambaStage(512, 2, self.window_size)
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

#--------------------------------
# Temporal Factory - Flexible temporal module builder for ablation
#  ---------------------------------

def build_temporal_module(dim, temporal_type):

    if temporal_type == "bidirectional_mamba":
        return BidirectionalTemporalMamba(dim)

    elif temporal_type == "unidirectional_mamba":
        #return TemporalSelectiveTrueMamba(dim) FOR MAMBA-SSM
        return TemporalMamba(dim)

    elif temporal_type == "lstm":
        return nn.LSTM(dim, dim, batch_first=True)

    elif temporal_type == "none":
        return nn.Identity()

    else:
        raise ValueError("Unknown temporal type")

# -------------------------------
# SwOSMBTM for MER
# -------------------------------

class SwOSMBTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = cfg.NUM_FRAMES
        self.num_classes =  cfg.NUM_CLASSES

        # Spatial Mamba (orthogonal scans)
        # Bidirectional temporal SSM
        self.spacial = SwinMamba(in_ch=3, out_dim=512)
        #self.temporal = BidirectionalTemporalMamba(512) #TemporalMamba(512)
        self.temporal = build_temporal_module(512,cfg.TEMPORAL_TYPE)
        self.classifier = nn.Linear(512, self.num_classes)


    def forward(self, x):
         
        B = x.size(0) # never cfg.batch_size because of the last batch, which can be smaller
        T = self.T
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        '''feats = []'''

            
        # Flatten temporal dimension
        x = x.reshape(B * T, C, H, W)
        feats = self.spacial(x) # (B*T, 512)
        feats = feats.reshape(B, T, -1)   # (B, T, 512)
        #feats = self.temporal(feats)       # Temporal MER modeling
        #---- Temporal ablation here ----
        if cfg.TEMPORAL_TYPE == "lstm":
            feats, _ = self.temporal(feats)
        elif cfg.TEMPORAL_TYPE != "none":
            feats = self.temporal(feats)

        feats = feats.mean(dim=1)

        return self.classifier(feats)

#--------------------------------
# Centralized transforms for all datasets (can be extended to multiple datasets with different transforms)
#--------------------------------

mytransforms = transforms.Compose([

    # 1️⃣ Resize all faces to a fixed size
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),

    # 2️⃣ Convert to grayscale (optional but strongly recommended for MER)
    transforms.Grayscale(num_output_channels=3), # keep 3 channels for pretrained backbones, but all contain the same grayscale info

    # 3️⃣ Data normalization  (using ImageNet stats as a common practice for pretrained backbones)
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




# ======================================
#  Utility Functions
# ======================================

#-------------------------------
#  Experiment folder management - create unique folder for each run
#-------------------------------
def create_experiment_folder(base_dir="experiments"):
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"SwOSMBTM_Epoch{cfg.EPOCHS}_{timestamp}"
    #exp_name = f"{cfg.SPATIAL_TYPE}__{cfg.TEMPORAL_TYPE}__{cfg.SAMPLING_TYPE}__{cfg.A_STABILITY}"
    exp_dir = os.path.join(base_dir, f"exp_{exp_name}" )
    

    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

#-------------------------------
# Save config to experiment folder for reproducibility
#-------------------------------
def save_config(cfg, exp_dir):
    config_path = os.path.join(exp_dir, "config.txt")
    with open(config_path, "w") as f:
        for key, value in vars(cfg).items():
            f.write(f"{key}: {value}\n")

#-------------------------------
# Checkpointing - save latest and best models
#-------------------------------
def save_checkpoint(model, optimizer, epoch, exp_dir, is_best=False):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # Save latest model
    torch.save(checkpoint, os.path.join(exp_dir, "model.pth"))

    # Save best model separately
    if is_best:
        torch.save(checkpoint, os.path.join(exp_dir, "best_model.pth"))

#--------------------------------
# Save training log to centralized CSV files (root level)
#-------------------------------
def save_training_log(log, epoch, train_loss, val_loss, train_acc, val_acc, exp_dir):

    # Extract experiment name from exp_dir path
    exp_name = os.path.basename(exp_dir)
    
    # Save full log as DataFrame
    log_df = pd.DataFrame(log, columns=[
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc"
    ])
    log_df.to_csv("training_log.csv", index=False)

    # Append current epoch to centralized experiment log
    log_path = "experiment_log.csv"
    
    # Create log file header if it doesn't exist
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
    
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            round(train_loss, 5),
            round(val_loss, 5),
            round(train_acc, 4),
            round(val_acc, 4)
        ])

# ============================================================================
# Checkpoint Loading with Parameter Migration
# ============================================================================

def load_checkpoint_with_migration(model, checkpoint_path, device):

    import os
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found. Proceeding with random initialization.")
        return model
    
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
        
        # Handle both old format (with epoch/optimizer) and new format (direct state_dict)
        if "model_state_dict" in checkpoint:
            print(f"Loading checkpoint from wrapped format (epoch {checkpoint.get('epoch', 'unknown')})")
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        print(f"Checkpoint has {len(state_dict)} model parameters")
        
        # Check if we need to migrate temporal parameters
        has_old_temporal = any(k in state_dict for k in ["temporal.ssm.A", "temporal.ssm.B", "temporal.ssm.C", "temporal.ssm.D"])
        has_new_temporal = any(k in state_dict for k in ["temporal.ssm.A_log", "temporal.ssm.in_proj.weight"])
        
        if has_old_temporal and not has_new_temporal:
            print("\n[MIGRATE] Detected old temporal SSM format. Migrating parameters...")
            
            # Count and remove old temporal incompatible parameters
            old_temporal_keys = [k for k in state_dict.keys() if k.startswith("temporal.ssm.") and k.split(".")[-1] in ["A", "B", "C", "D"]]
            
            print(f"  Found {len(old_temporal_keys)} old temporal parameters:")
            for key in old_temporal_keys:
                print(f"    - Removing: {key}")
                del state_dict[key]
            
            # Spatial mamba and backbone should still be compatible
            spatial_keys = [k for k in state_dict.keys() if "spatial_mamba" in k]
            backbone_keys = [k for k in state_dict.keys() if "backbone" in k]
            
            print(f"\n  Preserving {len(spatial_keys)} spatial mamba parameters [OK]")
            print(f"  Preserving {len(backbone_keys)} backbone parameters [OK]")
            
            print("\nLoading checkpoint with strict=False to allow temporal SSM reinitialization...")
            model.load_state_dict(state_dict, strict=False)
            print("[OK] Checkpoint loaded successfully! Temporal SSM will be retrained.")
            
        else:
            # Normal loading
            model.load_state_dict(state_dict)
            print("[OK] Checkpoint loaded successfully (no migration needed).")
            
        return model
        
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\n[ERROR] Error loading checkpoint: {e}")
        print("Proceeding with random initialization.")
        return model

# ============================================================================
# Main training and evaluation loop
# ============================================================================

def main():

    # -------------------------------
    # Reproducibility
    # -------------------------------
    import random
    import numpy as np

    def set_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)

    # -------------------------------
    # Device
    # -------------------------------
    use_gpu = torch.cuda.is_available()   # change to False when you want CPU
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(device)

    #from config import cfg
    #print(cfg)   # print experiment configuration

    #---------------
    # Load dataset and create splits
    #--------------
    dataset = CASME2Dataset(
        root=cfg.DATA_ROOT,
        annotation_file=cfg.ANNOTATION_FILE,
        transform=mytransforms,
        T=cfg.NUM_FRAMES,
        limit=cfg.LIMIT
    )
    N = len(dataset)
    train_len = int(cfg.TRAIN_SPLIT * N)
    val_len   = int(cfg.VAL_SPLIT * N)
    test_len  = N - train_len - val_len
   
    best_val_acc = 0.0
    best_epoch = 0

    all_fold_results = []
    
    import numpy as np

    subjects = np.array(dataset.subjects)
    unique_subjects = np.unique(subjects)


    for test_subject in unique_subjects:
        print(f"\n===== LOSO Fold: Test Subject {test_subject} =====")
        log = []

        # Indices
        train_indices = np.where(subjects != test_subject)[0]
        test_indices  = np.where(subjects == test_subject)[0]

        #------------------
        # Handle class imbalance with weighted loss - higher weight to minority classes
        #-------------
        labels = dataset.ann['Estimated Emotion'].str.lower().map(dataset.label_map).values 
        train_labels = labels[train_indices]
        class_counts = torch.bincount(torch.tensor(train_labels), minlength=cfg.NUM_CLASSES)
        weights = 1.0 / (class_counts.float() + 1e-6)
        weights = weights / weights.sum() * len(class_counts)


        # Subsets
        train_set = torch.utils.data.Subset(dataset, train_indices)
        test_set  = torch.utils.data.Subset(dataset, test_indices)

        # Optional validation split inside training
        val_size = int(0.2 * len(train_set))
        train_size = len(train_set) - val_size

        #---------------
        # # ensure reproducible splits
        #-----------------------
        #generator = torch.Generator().manual_seed(42) 
        generator = torch.Generator().manual_seed(42 + int(test_subject))
        train_set, val_set = torch.utils.data.random_split(
            train_set,
            [train_size, val_size],
            generator=generator
        )

        #-------------------
        # Create DataLoaders reproducible in the same order with fixed seed and shuffle for training
        #-------------
        
        train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True,    num_workers=4,
        pin_memory=True) # num_workers=0 for debugging otherwissen2nfor GPU
        val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False,    num_workers=4,
        pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=False,    num_workers=4,
        pin_memory=True)

        #-------------
        # model and training components
        #-------------------
        model = SwOSMBTM().to(device)

        criterion = nn.CrossEntropyLoss(weight=weights.to(device))

        optimizer = torch.optim.AdamW(
            [
                {"params": model.spatial.parameters(), "lr": cfg.BKBONE_LR},   # Spatial LR
                {"params": model.temporal.parameters(), "lr": cfg.TEMP_LR},  # Temporal LR
            ],
            weight_decay=0.05
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)#  gradually lowers the learning rate

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Total params:", total_params)
        print("Trainable params:", trainable_params)
        print("Model and components initialized.")

        #-------------------------------
        # Quick test run to verify dimensions and checkpoint loading before full training
        #-----------------------------


        
        x, y = next(iter(train_loader))# quick test run to verify dimensions
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            out = model(x)

        print("Input:", x.shape)
        print("Output:", out.shape)

        # Create experiment folder
        exp_dir = create_experiment_folder()
        print("Experiment directory:", exp_dir)

        best_val_acc = float('-inf')
        best_epoch = -1

        save_config(cfg, exp_dir)

        start= time.time()
        

        from torch.amp import autocast, GradScaler
        #scaler = GradScaler("cuda")
        scaler = GradScaler(enabled=use_gpu)



        #=====================================================================
        # training loop
        #====================================================================

        for epoch in range(cfg.EPOCHS):
        
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            #-------------------
            #resolveCUDA issue
            #-------------------


            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad()

                #with autocast("cuda"):
                with autocast(device_type='cuda', enabled=use_gpu):
                    logits = model(x)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()

                # ✅ Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

                # before UDA issue
                '''         
                for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                #-------------------
                # Forward pass, loss, backward, optimize
                #-------------------
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # avoid exploding gradients
                optimizer.step()'''

                # inside batch loop (optional for debugging)
                preds = logits.argmax(dim=1)
                batch_acc = (preds == y).float().mean()

                #------------------------
                # Train accumulate for epoch
                #-------------------
                train_correct += (preds == y).sum().item()
                train_total += y.size(0)
                train_loss += loss.item()

                #if batch_idx % 10 == 0:
                print(f"Batch Loss: {loss.item():.4f} | Acc: {batch_acc:.2%}")

            train_acc = 100 * train_correct / train_total
            train_loss /= len(train_loader)
        
            Tend = time.time()-start
            print("Traing Time:", Tend)

            #=============================
            # Validate
            #==================================
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)

                    #------------------------
                    # Validation accumulate for epoch
                    #-------------------
                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == y).sum().item()
                    val_total += y.size(0)

            val_acc = 100 * val_correct / val_total
            val_loss /= len(val_loader)

            #-----------------------------
            # compute epoch metrics 
            #---------------------------

            print(f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
            )
            log.append([epoch, train_loss, train_acc, val_loss, val_acc])
        


            #-----------    
            # freeze the best model
            #-----------------
        
            is_best = val_acc >= best_val_acc

            if is_best:
                best_val_acc = val_acc
                best_epoch = epoch

            save_checkpoint(
                model,
                optimizer,
                epoch,
                exp_dir,
                is_best=is_best
            )

            #------------
            # Next Epoch Learning Rate Update
            #----------------
            scheduler.step() # update learning rate according to sched
        #------End of epoch loop-------

        print("Training & Validation step OK ",Tend)

        save_training_log(log, epoch, train_loss, val_loss, train_acc, val_acc, exp_dir)

        from sklearn.metrics import f1_score, recall_score, accuracy_score

        # -----------------------------
        # Fold Metrics (MER standard)
        # -----------------------------

        fold_acc = accuracy_score(all_gt, all_preds) * 100
        fold_uf1 = f1_score(all_gt, all_preds, average='micro', zero_division=0)
        fold_uar = recall_score(all_gt, all_preds, average='micro', zero_division=0)

        print(f"[FOLD RESULT]")
        print(f"Accuracy: {fold_acc:.2f}%")
        print(f"UF1 (Macro-F1): {fold_uf1:.4f}")
        print(f"UAR (Macro-Recall): {fold_uar:.4f}")

        all_fold_results.append((fold_acc, fold_uf1, fold_uar))

        # val accuracy not to overfit the model training acc → memorization, 
        # validation acc → generalization, 
        # test acc → final report

        #=======================================
        # TEST EVALUATION
        #=========================================
        
        #------------------------------
        # reload the best model from experiment directory
        #------------------------------
        best_model_path = os.path.join(exp_dir, "best_model.pth")
        load_checkpoint_with_migration(model, best_model_path, device)
        model.eval()

        test_correct, test_total = 0, 0
        all_preds = []
        all_gt = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)

                #------------------------
                # Test accumulate for final evaluation
                #-------------------
                preds = logits.argmax(dim=1)
                test_correct += (preds == y).sum().item()
                test_total += y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_gt.extend(y.cpu().numpy())

        #from sklearn.metrics import classification_report, confusion_matrix, f1_score

        test_acc = 100 * test_correct / test_total
        print(f"[TEST] Final Test Accuracy: {test_acc:.2f}%")

        #------------------------------
        # - Save evaluation results
        #---------------
        report = classification_report(
            all_gt, all_preds,
            digits=4,
            output_dict=True,
            zero_division=0
        )

        cm = confusion_matrix(all_gt, all_preds)

        pd.DataFrame(report).transpose().to_csv(
            os.path.join(exp_dir, "classification_report.csv")
        )

        pd.DataFrame(cm).to_csv(
            os.path.join(exp_dir, "confusion_matrix.csv"),
            index=False
        )

        with open(os.path.join(exp_dir, "summary.txt"), "w") as f:
            f.write(f"Final Test Accuracy: {exp_dir} {test_acc:.4f}\n")
            f.write(f"Micro-F1: {f1_score(all_gt, all_preds, average='micro', zero_division=0):.4f}\n")

        #print(" Saved: classification_report.csv & confusion_matrix.csv")
        print(" Experiment finished")

        fold_acc = test_acc
        fold_f1 = f1_score(all_gt, all_preds, average='micro', zero_division=0)

        all_fold_results.append((fold_acc, fold_f1))
    #-------Enf of LOSO fold loop-------
    
    #-----------------------------
    # Final LOSO results aggregation and reporting
    #-----------------------------

    accs = [r[0] for r in all_fold_results]
    f1s  = [r[1] for r in all_fold_results]

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)

    mean_f1 = np.mean(f1s)
    std_f1  = np.std(f1s)


    # ----------------------------------
    # Final LOSO Aggregation
    # ----------------------------------

    accs = [r[0] for r in all_fold_results]
    uf1s = [r[1] for r in all_fold_results]
    uars = [r[2] for r in all_fold_results]

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)

    mean_uf1 = np.mean(uf1s)
    std_uf1  = np.std(uf1s)

    mean_uar = np.mean(uars)
    std_uar  = np.std(uars)

    print("\n==============================")
    print("FINAL LOSO RESULTS")
    print("==============================")
    print(f"Accuracy : {mean_acc:.2f} ± {std_acc:.2f}")
    print(f"UF1      : {mean_uf1:.4f} ± {std_uf1:.4f}")
    print(f"UAR      : {mean_uar:.4f} ± {std_uar:.4f}")

#=============END OF MAIN================

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("Exception in main:", repr(e))
        traceback.print_exc()
        raise 
