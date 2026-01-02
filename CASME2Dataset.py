import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

"""
CASME2/
 ├── raw/
 │   ├── s01/
 │   ├── s02/
 │   └── ...
 ├── annotations.xlsx
 └── processed/
 
 Annotation row (idx) from CSV/Excel
          │
          ▼
Extract subject, video, emotion → map to label
          │
          ▼
Build clip directory path → list sorted frames
          │
          ▼
For each frame:
   Read with cv2 → BGR to RGB → apply self.transform
          │
          ▼
Stack frames → x: (T_original, C, H, W)
          │
          ▼
Compute frame-to-frame motion magnitudes → scores
          │
          ▼
Weighted temporal sampling using scores → select T frames
          │
          ▼
Pad if x.shape[0] < T → x: (T, C, H, W)
          │
          ▼
Return clip tensor x and label

 
 """
 
 
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
        
        print(self.ann.columns)

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
        """if x.shape[0] > self.T:
            #x = x[:self.T]
            num_frames = x.shape[0]
            start = torch.randint(0, num_frames - self.T + 1, (1,)).item() # random temporal crop
            x = x[start : start + self.T]

        elif x.shape[0] < self.T:
            pad = self.T - x.shape[0]
            x = torch.cat([x, x[-1:].repeat(pad,1,1,1)]) """

        if x.shape[0] < self.T:
            pad = self.T - x.shape[0]
            x = torch.cat([x, x[-1:].repeat(pad,1,1,1)])


        return x, label
