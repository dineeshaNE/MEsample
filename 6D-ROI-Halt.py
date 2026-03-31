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
        self.DATA_ROOT = "CASME2/raw"
        self.ANNOTATION_FILE = "CASME2/CASME2.csv"
        self.LIMIT = 20
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
        self.SEED = 42


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

class CASME2DatasetFAProcess(Dataset):
    def __init__(self, root, annotation_file, transform=None, T=cfg.NUM_FRAMES, limit=None):
        self.root = root
        self.ann = pd.read_csv(annotation_file)
        self.transform = transform
        self.T = T

         # Directory to store alignment matrices
        self.cache_dir = os.path.join(self.root, "alignment_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Directory to store processed aligned frames (for ROI ablation)
        self.proc_cache_dir = os.path.join(self.root, "processed_cache")
        os.makedirs(self.proc_cache_dir, exist_ok=True)


         # Check if cache already exists for ALL clips---------
        self.use_alignment = True
        all_cached = True
        for _, row in self.ann.iterrows():
            subject = row['Subject']
            video = row['Filename']
            if not os.path.exists(self.get_cache_path(subject, video)):
                all_cached = False
                break

        if not all_cached:
            print("Initializing face alignment model...")
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            print("All alignment cache found. Skipping face_alignment model.")
            self.fa = None


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

    #-------------------------------
    # Get processed cache path for ROI emphasis
    #-------------------------------
    def get_processed_cache_path(self, subject, video):
        subject = f"sub{int(subject):02d}"
        #filename = f"{subject}_{video}.pt"
        filename = f"{subject}_{video}_T{self.T}_v1.pt"
        return os.path.join(self.proc_cache_dir, filename)

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

        center = (left_eye + right_eye) / 2
        eyes_center = (int(center[0]), int(center[1]))

        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

        return M

    def _format_subject(self, subject):
        return f"sub{int(subject):02d}"

    def __len__(self):
        return len(self.ann)
    
    #-------------------------------
    # Get landmarks for ROI mask
    #-------------------------------
    def get_landmarks(self, img):
    
        if self.fa is None:
            return None
        preds = self.fa.get_landmarks(img)
        if preds is None:
            return None
        landmarks = preds[0]  # first detected face
        return landmarks.astype(int)



    #-------------------------------
    # Region-based motion emphasis with facial landmarks
    #-------------------------------

    def create_roi_mask(image_shape, landmarks):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        # left eye
        left_eye = landmarks[36:42]
        cv2.fillConvexPoly(mask, left_eye, 1)
        # right eye
        right_eye = landmarks[42:48]
        cv2.fillConvexPoly(mask, right_eye, 1)
        # mouth
        mouth = landmarks[48:68]
        cv2.fillConvexPoly(mask, mouth, 1)
        return mask

    #--------------------------------
    # Expanded facial landmarks
    #--------------------------------
    def expand_mask(mask, k=15):
        kernel = np.ones((k, k), np.uint8)
        expanded = cv2.dilate(mask, kernel)
        return expanded

    #--------------------------------
    # soft masking for smooth emphasis
    #--------------------------------
    def soft_mask(image, mask):
        mask = mask.astype(np.float32)
        mask = mask * 1.0 + (1-mask) * 0.2
        mask = np.expand_dims(mask, -1)
        return image * mask

    #-------------------------------
    # Apply soft mask to tensor
    #-------------------------------
    def apply_soft_roi_mask(x, mask):
        mask = torch.tensor(mask).float()
        mask = mask * 1.0 + (1 - mask) * 0.2
        mask = mask.unsqueeze(0)  # H,W → 1,H,W
        return x * mask

    def __getitem__(self, idx):

        proc_cache_path = self.get_processed_cache_path(subject, video)

        # 🔥 If already processed → LOAD and RETURN
        if os.path.exists(proc_cache_path):
            x = torch.load(proc_cache_path)
            return x, label

        row = self.ann.iloc[idx]
        #print  (f"Processing row {idx}")
        #print(f"Processing row {idx}: {row.to_dict()}")

        subject = row['Subject']
        video = row['Filename']

        cache_path = self.get_cache_path(subject, video)

        # Try loading cached matrix
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
                if self.fa is not None:
                        M = self.compute_alignment_matrix(img)
                else:
                        M = np.eye(2, 3, dtype=np.float32)

                if M is None:
                    # Fallback (identity transform)
                    M = np.eye(2, 3, dtype=np.float32)


                #  Save permanently
                np.save(cache_path, M)

            #  Apply alignment
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            #-----ROI Masking for motion emphasis-----
            # detect landmarks
            landmarks = self.get_landmarks(img)

            if landmarks is not None:
                # create mask
                mask = self.create_roi_mask((224,224), landmarks)

                # expand mask (crow-feet etc.)
                mask = self.expand_mask(mask, 21)

                # apply soft mask
                img = self.soft_mask(img, mask)

            # convert to tensor
            img = img.astype(np.float32) / 255.0
            img = torch.tensor(img).permute(2,0,1)

            #----------------------

            if self.transform:
                img = self.transform(img) #(3, 224, 224)
            images.append(img)

        x = torch.stack(images)   # (T, C, H,  W)
        #print(f"Frames done: {clip_dir}")

        replacement=False if len(x) >= self.T else True

        #-----------------
        # Alternative sampling strategies for ablation
        #-----------------
        if cfg.SAMPLING_TYPE == "motion":
            # Compute simple motion magnitude  for strong, learnable temporal signature
            diffs = (x[1:] - x[:-1]).abs().mean(dim=(1,2,3))
            scores = torch.cat([diffs[:1], diffs])
            scores = scores / (scores.sum() + 1e-6)# Normalize to avoid div by zero in no motion clips
            indices = torch.multinomial(scores, self.T, replacement=replacement) # Weighted temporal sampling # replacement = TRUE

        elif cfg.SAMPLING_TYPE == "uniform":
            indices = torch.linspace(0, x.shape[0]-1, self.T).long()

        elif cfg.SAMPLING_TYPE == "first":
            indices = torch.arange(min(self.T, x.shape[0]))

        #Save processed tensor b4 sampling for generalization
        # torch.save(x, proc_cache_path)        
        
        x = x[indices]


        # Temporal normalization for robust batch processing

        if x.shape[0] < self.T:
            pad = self.T - x.shape[0]
            x = torch.cat([x, x[-1:].repeat(pad,1,1,1)])

        # Save processed tensor for future fast loading
        torch.save(x, proc_cache_path)

        return x, label


class CASME2DatasetFA(Dataset):
    def __init__(self, root, annotation_file, transform=None, T=cfg.NUM_FRAMES, limit=None):
        self.root = root
        self.ann = pd.read_csv(annotation_file)
        self.transform = transform
        self.T = T

         # Directory to store alignment matrices
        self.cache_dir = os.path.join(self.root, "alignment_cache")
        os.makedirs(self.cache_dir, exist_ok=True)


         # Check if cache already exists for ALL clips---------
        self.use_alignment = True
        all_cached = True
        for _, row in self.ann.iterrows():
            subject = row['Subject']
            video = row['Filename']
            if not os.path.exists(self.get_cache_path(subject, video)):
                all_cached = False
                break

        if not all_cached:
            print("Initializing face alignment model...")
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            print("All alignment cache found. Skipping face_alignment model.")
            self.fa = None


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

        center = (left_eye + right_eye) / 2
        eyes_center = (int(center[0]), int(center[1]))

        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

        return M

    def _format_subject(self, subject):
        return f"sub{int(subject):02d}"

    def __len__(self):
        return len(self.ann)
    
    #-------------------------------
    # Get landmarks for ROI mask
    #-------------------------------
    def get_landmarks(self, img):
    
        if self.fa is None:
            return None
        preds = self.fa.get_landmarks(img)
        if preds is None:
            return None
        landmarks = preds[0]  # first detected face
        return landmarks.astype(int)

    
    #-------------------------------
    # Region-based motion emphasis with facial landmarks
    #-------------------------------

    def create_roi_mask(image_shape, landmarks):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        # left eye
        left_eye = landmarks[36:42]
        cv2.fillConvexPoly(mask, left_eye, 1)
        # right eye
        right_eye = landmarks[42:48]
        cv2.fillConvexPoly(mask, right_eye, 1)
        # mouth
        mouth = landmarks[48:68]
        cv2.fillConvexPoly(mask, mouth, 1)
        return mask

    #--------------------------------
    # Expanded facial landmarks
    #--------------------------------
    def expand_mask(mask, k=15):
        kernel = np.ones((k, k), np.uint8)
        expanded = cv2.dilate(mask, kernel)
        return expanded

    #--------------------------------
    # soft masking for smooth emphasis
    #--------------------------------
    def soft_mask(image, mask):
        mask = mask.astype(np.float32)
        mask = mask * 1.0 + (1-mask) * 0.2
        mask = np.expand_dims(mask, -1)
        return image * mask

    #-------------------------------
    # Apply soft mask to tensor
    #-------------------------------
    def apply_soft_roi_mask(x, mask):
        mask = torch.tensor(mask).float()
        mask = mask * 1.0 + (1 - mask) * 0.2
        mask = mask.unsqueeze(0)  # H,W → 1,H,W
        return x * mask

    def __getitem__(self, idx):
        row = self.ann.iloc[idx]
        #print  (f"Processing row {idx}")
        #print(f"Processing row {idx}: {row.to_dict()}")

        subject = row['Subject']
        video = row['Filename']

        cache_path = self.get_cache_path(subject, video)

        # Try loading cached matrix
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
                if self.fa is not None:
                        M = self.compute_alignment_matrix(img)
                else:
                        M = np.eye(2, 3, dtype=np.float32)

                if M is None:
                    # Fallback (identity transform)
                    M = np.eye(2, 3, dtype=np.float32)


                #  Save permanently
                np.save(cache_path, M)

            #  Apply alignment
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            #-----ROI Masking for motion emphasis-----
            # detect landmarks
            landmarks = self.get_landmarks(img)

            if landmarks is not None:
                # create mask
                mask = self.create_roi_mask((224,224), landmarks)

                # expand mask (crow-feet etc.)
                mask = self.expand_mask(mask, 21)

                # apply soft mask
                img = self.soft_mask(img, mask)

            # convert to tensor
            img = img.astype(np.float32) / 255.0
            img = torch.tensor(img).permute(2,0,1)

            #----------------------

            if self.transform:
                img = self.transform(img) #(3, 224, 224)
            images.append(img)

        x = torch.stack(images)   # (T, C, H,  W)
        #print(f"Frames done: {clip_dir}")

        replacement=False if len(x) >= self.T else True

        #-----------------
        # Alternative sampling strategies for ablation
        #-----------------
        if cfg.SAMPLING_TYPE == "motion":
            # Compute simple motion magnitude  for strong, learnable temporal signature
            diffs = (x[1:] - x[:-1]).abs().mean(dim=(1,2,3))
            scores = torch.cat([diffs[:1], diffs])
            scores = scores / (scores.sum() + 1e-6)# Normalize to avoid div by zero in no motion clips
            indices = torch.multinomial(scores, self.T, replacement=replacement) # Weighted temporal sampling # replacement = TRUE

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




# -------------------------------
# Mamba instead in ssm_mamba
# -------------------------------

class Mamba(nn.Module):

    def __init__(self, d_model, use_nonlinearity=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)


        # Diagonal SSM parameters
        #self.A = nn.Parameter(torch.randn(d_model)* 0.01)
        self.A_log = nn.Parameter(torch.randn(d_model) * 0.01)
        #A = torch.tanh(self.A)
        #A = -torch.exp(self.A)
        self.B = nn.Parameter(torch.randn(d_model)* 0.01)
        self.C = nn.Parameter(torch.randn(d_model)* 0.01)
        self.D = nn.Parameter(torch.randn(d_model))

        self.use_nonlinearity = use_nonlinearity
        self.act = nn.GELU() if use_nonlinearity else nn.Identity() # not to limit for complex ME patterns
        #print("Initialized SimpleSSM",d_model)


    def forward(self, x):
        #        x: (batch, seq_len, d_model)
        B, T, D = x.shape
        # Stable A
        A = -torch.exp(torch.clamp(self.A_log, -5, 5))
        s = torch.zeros(B, D, device=x.device)
        outputs = []

        for t in range(T):
            xt = x[:, t, :]
            # ---------------------------
            # Input debug
            # ---------------------------
            if torch.isnan(x).any():
                print("NaN detected in input x")

            # STATE UPDATE (this is the missing part before)
            s = A * s + self.B * xt
            s = torch.tanh(s)
            if torch.isnan(s).any():
                print("NaN after s")
            # Stabilize state
            #s = torch.tanh(s)

            # OUTPUT
            yt = self.C * s + self.D * xt
            if torch.isnan(yt).any():
                print("NaN after yt")
            outputs.append(yt)
            #print(f"SSM timestep {t}, output shape: {yt.shape}",outputs.__len__)

        #print(f"SSM final output shape: {outputs[0].shape} yt {yt.shape}")
        #return torch.stack(outputs, dim=1)
        y = torch.stack(outputs, dim=1)
        return self.norm(y)

    # -------------------------------
    # Window Orthogonal Mamba
    # -------------------------------
import torch
import torch.nn as nn

class WindowOrthogonalMambaHVD(nn.Module):

    def __init__(self, dim, window_size):
        super().__init__()

        self.ws = window_size

        self.ssm_h = Mamba(d_model=dim)
        self.ssm_v = Mamba(d_model=dim)
        self.ssm_d = Mamba(d_model=dim)

        self.alpha = nn.Parameter(torch.ones(6))

        self.norm = nn.LayerNorm(dim)

    def diagonal_scan_fast(self, w):
        """
        w : (B, ws, ws, C)
        return: (B, ws*ws, C)
        """

        B, ws, _, C = w.shape

        device = w.device

        rows = torch.arange(ws, device=device)
        cols = torch.arange(ws, device=device)

        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")

        diag_id = grid_r + grid_c
        order = torch.argsort(diag_id.reshape(-1))

        seq = w.reshape(B, ws*ws, C)
        seq = seq[:, order, :]

        return seq       

    def forward(self, w):
        """
        w: (Bwin, ws*ws, C)
        """

        B, L, C = w.shape
        ws = self.ws

        # reshape window
        w = w.reshape(B, ws, ws, C)

        # -------- horizontal scan --------
        s1 = w.reshape(B, L, C)
        y1 = self.ssm_h(s1)

        # reverse horizontal
        s2 = torch.flip(s1, dims=[1])
        y2 = torch.flip(self.ssm_h(s2), dims=[1])

        # -------- vertical scan --------
        v = w.permute(0,2,1,3)  # swap H,W
        s3 = v.reshape(B, L, C)
        y3 = self.ssm_v(s3)

        # reverse vertical
        s4 = torch.flip(s3, dims=[1])
        y4 = torch.flip(self.ssm_v(s4), dims=[1])

        # -------- diagonal scan --------
        #↘ scan : diagonal_scan_fast(w)
        d1 = self.diagonal_scan_fast(w)
        y5 = self.ssm_d(d1)
        #↙ scan : diagonal_scan_fast(flip(w))
        w_flip = torch.flip(w, dims=[2])
        d2 = self.diagonal_scan_fast(w_flip)
        y6 = self.ssm_d(d2)

        # merge with learnable weights (instead of simple average)
        #y = (y1 + y2 + y3 + y4) / 4
        #self.alpha = nn.Parameter(torch.ones(6)) *********************
        y = (
            self.alpha[0]*y1 +
            self.alpha[1]*y2 +
            self.alpha[2]*y3 +
            self.alpha[3]*y4 +
            self.alpha[4]*y5 +
            self.alpha[5]*y6
        ) / self.alpha.sum()

        y = self.norm(y)

        return y.reshape(B, L, C)


#---------------------------------
# Spacial Factory - Flexible spatial module builder for ablation
#----------------------------------
def build_spatial_module(dim, window_size, spatial_type):

    if spatial_type == "window_mamba":
        return WindowOrthogonalMambaHVD(dim, window_size)

    elif spatial_type == "single_mamba":
        return Mamba(dim)

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
    def __init__(self, dim, window_size=cfg.WINDOW_SIZE, shift=False):
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

        x = x + self.mlp(self.norm2(x))
        return x
    

# -------------------------------
# Temporal Selective True Mamba Style
#-------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSelectiveTrueMamba(nn.Module):

    #def __init__(self, d_model):
        
    def __init__(self, d_model, d_conv=3, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Pre-norm (standard Mamba style)
        self.norm = nn.LayerNorm(self.d_model, eps=1e-4)
        #   self.norm = nn.LayerNorm(d_model)

        # Input-dependent projections (Selective SSM)
        self.in_proj = nn.Linear(d_model, 3 * d_model)

        # Depthwise temporal convolution (local mixing)
        self.dwconv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=d_model  # depthwise
        )

        # Stable diagonal A parameterization
        self.A_log = nn.Parameter(torch.randn(d_model))

        # Residual skip scaling
        self.D = nn.Parameter(torch.ones(d_model))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        # Learnable global timestep (small initial value)
        self.dt = nn.Parameter(torch.tensor(0.1))

        #self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, D = x.shape
        assert D == self.d_model

        # ---------------------------
        # Input debug
        # ---------------------------
        if torch.isnan(x).any():
            print("NaN detected in input x")
            x = torch.nan_to_num(x)

        # ---------------------------
        # Local temporal mixing
        # ---------------------------


        # Local temporal mixing (Conv1D expects B, C, T)
        x_conv = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        if torch.isnan(x_conv).any():
            print("NaN after depthwise conv")
        
        x = self.norm(x + 0.1 * x_conv)
        if torch.isnan(x).any():
            print("NaN after LayerNorm")
            x = torch.nan_to_num(x)

        # ********x = x + 0.1 * x_conv  # Residual connection for local mixing

        # ---------------------------
        # Selective parameter generation
        # ---------------------------
        proj = self.in_proj(x)
        if torch.isnan(proj).any():
            print("NaN after in_proj")

        dt, Bx, Cx = proj.chunk(3, dim=-1)

        # Stabilize dynamics
        dt = torch.sigmoid(dt)           # (0, 1)
        Bx = torch.tanh(Bx)
        #Cx = torch.tanh(Cx)
        gate = torch.sigmoid(Cx)   # gated output
        # Stable negative A
        #A = -torch.exp(self.A_log)

        # stabilize A with different strategies for ablation
        if cfg.A_STABILITY == "exp":
            # A = -torch.exp(self.A_log)# better stability
            A = -torch.exp(torch.clamp(self.A_log, -5, 5)) #**

        elif cfg.A_STABILITY == "abs":
            A = torch.abs(self.A_log)# ensure stability

        elif cfg.A_STABILITY == "raw":
            A = self.A_log

        # Initialize state
        s = torch.zeros(B, D, device=x.device)
        outputs = []

        for t in range(T):
            dt_t = dt[:, t]
            Bx_t = Bx[:, t]
            #Cx_t = Cx[:, t]
            gate_t = gate[:, t]

            # Stable Euler update
            ds = A * s + Bx_t
            s = s + self.dt * dt_t * ds   
            if torch.isnan(s).any():
                print("NaN after s")

            s = torch.tanh(s )
            if torch.isnan(s).any():
                print("NaN detected in state")
            #s = torch.clamp(s, -10, 10)  # for better stability
            #A_bar = torch.exp(dt_t * A)
            #s = A_bar * s + Bx_t    
            #y = gate_t * s

            y = gate_t * s + self.D * x[:, t]
            outputs.append(y)

        y = torch.stack(outputs, dim=1)
        y = self.out_proj(y)
        if torch.isnan(y).any():
            print("NaN after output projection")
        y = self.dropout(y)    
        #print(f"SSM output shape:{x} {y}")

        #return self.out_proj(y)
        return x + y
    

# -------------------------------
# Bidirectional Temporal Mamba
#-------------------------------    
class BidirectionalTemporalMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.ssm = Mamba(d_model=dim)
        self.ssm = TemporalSelectiveTrueMamba(dim)
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
        return TemporalSelectiveTrueMamba(dim)

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
        self.spatial = SwinMamba(in_ch=3, out_dim=512)
        #self.temporal = BidirectionalTemporalMamba(512) #TemporalMamba(512)
        self.temporal = build_temporal_module( 512, cfg.TEMPORAL_TYPE)
        self.classifier = nn.Linear(512, self.num_classes)


    def forward(self, x):
        B = x.size(0) # never cfg.batch_size because of the last batch, which can be smaller
        T = self.T
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        feats = []

            
        # Flatten temporal dimension
        x = x.reshape(B * T, C, H, W)
        feats = self.spatial(x) # (B*T, 512)
        feats = feats.reshape(B, T, -1)   # (B, T, 512)
        #feats = self.temporal(feats)       # Temporal MER modeling
        #---- Temporal ablation here ----
        if cfg.TEMPORAL_TYPE == "lstm":
            feats, _ = self.temporal(feats)
        elif cfg.TEMPORAL_TYPE != "none":
            feats = self.temporal(feats)

        '''if cfg.TEMPORAL_TYPE == "none":
            feats = feats.mean(dim=1)
        else:'''
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
    exp_name = f"SwOSMBiTM_Epoch{cfg.EPOCHS}_SEED{cfg.SEED}_{timestamp}"
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
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    log_df.to_csv("training_log.csv", index=False)

    # Append current epoch to centralized experiment log
    log_path = os.path.join(exp_dir, "experiment_log.csv")
    
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
    
# =========================================
# Precompute function (OUTSIDE the class)
# =========================================
def precompute_dataset(dataset):
    print("Starting preprocessing and caching...")

    for i in range(len(dataset)):
        if i % 10 == 0:
            print(f"Processing {i}/{len(dataset)}")

        _ = dataset[i]   # 🔥 triggers preprocessing + saving

    print("Preprocessing complete!")    

# ============================================================================
# Main training and evaluation loop
# ============================================================================

def main():

    # -------------------------------
    # Reproducibility
    # -------------------------------
    import random
    import numpy as np

    def set_seed(seed=cfg.SEED):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark =  False

    set_seed(cfg.SEED)

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
    dataset = CASME2DatasetFA(
        root=cfg.DATA_ROOT,
        annotation_file=cfg.ANNOTATION_FILE,
        transform=mytransforms,
        T=cfg.NUM_FRAMES,
        limit=cfg.LIMIT
    )
    #Run once to precompute and cache all aligned frames for faster training (optional but recommended)
    precompute_dataset(dataset)

    N = len(dataset)
    train_len = int(cfg.TRAIN_SPLIT * N)
    val_len   = int(cfg.VAL_SPLIT * N)
    test_len  = N - train_len - val_len

    #------------------
    # Handle class imbalance with weighted loss - higher weight to minority classes
    #-------------
    labels = dataset.ann['Estimated Emotion'].str.lower().map(dataset.label_map).values 
    class_counts = torch.bincount(torch.tensor(labels), minlength=cfg.NUM_CLASSES)

    weights = 1.0 / (class_counts.float() + 1e-6)
    weights = weights / weights.sum() * len(class_counts)
    if (weights> 10).any():
      weights = torch.clamp(weights, max=5.0)

    #---------------
    # # ensure reproducible splits
    #-----------------------
    generator = torch.Generator().manual_seed(cfg.SEED) 
    train_set, val_set, test_set = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=generator
    )
    
    #-------------------
    # Create DataLoaders reproducible in the same order with fixed seed and shuffle for training
    #-------------
    pin_memory = torch.cuda.is_available()
    if pin_memory:
        num_workers=2
    else:
        num_workers=0

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True,    num_workers=num_workers,
    pin_memory=pin_memory) # num_workers=0 for debugging otherwissen2nfor GPU
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False,    num_workers=num_workers,
    pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=False,    num_workers=num_workers,
    pin_memory=pin_memory)
    
    #-------------
    # model and training components
    #-------------------
    model = SwOSMBTM().to(device)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.Adam([
        {"params": model.spatial.parameters(), "lr": cfg.BKBONE_LR},
        {"params": model.temporal.parameters(), "lr": cfg.TEMP_LR},
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)#  gradually lowers the learning rate

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total params:", total_params)
    print("Trainable params:", trainable_params)
    print("Model and components initialized.")

    #-------------------------------
    # Quick test run to verify dimensions and checkpoint loading before full training
    #-----------------------------

    best_val_acc = 0.0
    best_epoch = 0
    log = []
    
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
    # Save initial log with epoch 0 (before training)
    pd.DataFrame(log).to_csv(os.path.join(exp_dir, "training_log.csv"),columns=[
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc" ], index=False)
    
    save_config(cfg, exp_dir)

    start= time.time()
    

    '''    from torch.amp import autocast, GradScaler

    if torch.cuda.is_available():
        #scaler = torch.cuda.amp.GradScaler()
        scaler = GradScaler("cuda")
    else:
        scaler = None'''

    #=====================================================================
    # training loop
    #====================================================================

    for epoch in range(cfg.EPOCHS):
      
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        #-------------------
        #resolveCUDA issue
        #-------------------

        #for x, y in train_loader:
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            #print("Input range:", x.min().item(), x.max().item())
            #print("Any NaN in input:", torch.isnan(x).any())
            optimizer.zero_grad()

            #-------------------
            # Forward pass, loss, backward, optimize
            #-------------------
            #with autocast("cuda"):
            '''if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                    
                # *Nan Debug checks for performance
                if torch.isnan(loss):
                    print("NaN detected in loss!")
                    break
                if torch.isnan(logits).any():
                    print("NaN detected in logits!")
                    break

                scaler.scale(loss).backward()

                # ✅ Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
            else:'''
            logits = model(x)
            loss = criterion(logits, y)
            #optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # avoid exploding gradients
            optimizer.step()


            # inside batch loop (optional for debugging)
            preds = logits.argmax(dim=1)
            batch_acc = (preds == y).float().mean()

            #------------------------
            # Train accumulate for epoch
            #-------------------
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Index: {batch_idx} | Batch Loss: {loss.item():.4f} | Acc: {batch_acc:.2%}")

        train_acc = 100 * train_correct / train_total
        #train_loss /= len(train_loader)
        if len(train_loader) > 0:
            train_loss /= len(train_loader)
        else:
            train_loss = 0
    
        Tend = time.time()-start
        print("Traing Time:", Tend)

        #=============================
        # Validate
        #==================================
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
      
        if epoch % 5 == 0:
            pd.DataFrame(log).to_csv(os.path.join(exp_dir, "training_log.csv"), mode='a', header=False, index=False)
            log = []

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

    print("Training & Validation step OK ",Tend)

    save_training_log(log, epoch, train_loss, val_loss, train_acc, val_acc, exp_dir)

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
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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

    print("Experiment directory:", exp_dir)

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



if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("Exception in main:", repr(e))
        traceback.print_exc()
        raise 
