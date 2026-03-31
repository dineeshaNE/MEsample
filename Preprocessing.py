import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import cv2
import face_alignment
import numpy as np
from torchvision import transforms

#------------------------------------
# Central config.py
#-----------------------------------
class Config:
    def __init__(self):
        # Dataset / IO
        self.DATA_ROOT = "CASME2/raw"
        self.ANNOTATION_FILE = "CASME2/CASME2.csv"
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
        self.SEED = 31


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
        filename = f"{subject}_{video}.pt"
        #filename = f"{subject}_{video}_T{self.T}_v1.pt"
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

        proc_cache_path = self.get_processed_cache_path(subject, video)
        # 🔥 If already processed → LOAD and RETURN
        if os.path.exists(proc_cache_path):
            x = torch.load(proc_cache_path)
            print(f"Loaded processed tensor from cache for {subject}_{video}")
            return x, label
        
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

        diffs = (x[1:] - x[:-1]).abs().mean(dim=(1,2,3))
        scores = torch.cat([diffs[:1], diffs])
        scores = scores / (scores.sum() + 1e-6)# Normalize to avoid div by zero in no motion clips
        indices = torch.multinomial(scores, self.T, replacement=replacement) # Weighted temporal sampling # replacement = TRUE

        '''#-----------------
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
            indices = torch.arange(min(self.T, x.shape[0]))'''

        #Save processed tensor b4 sampling for generalization
        # torch.save(x, proc_cache_path)        
        
        x = x[indices]


        # Temporal normalization for robust batch processing

        if x.shape[0] < self.T:
            pad = self.T - x.shape[0]
            x = torch.cat([x, x[-1:].repeat(pad,1,1,1)])

        # Save processed tensor for future fast loading
        torch.save(x,  proc_cache_path)

        return x, label

def precompute():
    dataset = CASME2DatasetFAProcess(
        root="CASME2/raw",
        annotation_file="CASME2/CASME2.csv",
        transform=mytransforms,
        T=cfg.NUM_FRAMES,
        limit=cfg.LIMIT
    )

    print("Starting preprocessing...")

    for i in range(len(dataset)):
        _ = dataset[i]   # 🔥 THIS triggers processing + saving

        if i % 10 == 0:
            print(f"Processed {i}/{len(dataset)}")

    print("Preprocessing complete!")

if __name__ == "__precompute__":
    import traceback
try:
    precompute()
except Exception as e:
    print("Exception in main:", repr(e))
    traceback.print_exc()
    raise 