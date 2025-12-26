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
 └── processed/"""
 
 
class CASME2Dataset(Dataset):
    def __init__(self, root, annotation_file, transform=None, T=30):
        self.root = root
        self.ann = pd.read_excel(annotation_file)
        self.transform = transform
        self.T = T

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        row = self.ann.iloc[idx]

        subject = row['Subject']
        video = row['Filename']
        label = int(row['Label'])

        clip_dir = os.path.join(self.root, subject, video)
        frames = sorted(os.listdir(clip_dir))

        images = []
        for f in frames:
            img = cv2.imread(os.path.join(clip_dir, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            images.append(img)

        x = torch.stack(images)   # (T, C, H, W)

        # Temporal normalization
        if x.shape[0] > self.T:
            x = x[:self.T]
        elif x.shape[0] < self.T:
            pad = self.T - x.shape[0]
            x = torch.cat([x, x[-1:].repeat(pad,1,1,1)])

        return x, label
