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

        subject = row['Subject']
        video = row['Filename']

        #label = int(row['Label'])
        emotion = row['Estimated Emotion'].strip().lower()
        label = self.label_map[emotion]
        if emotion not in self.label_map:
            raise ValueError(f"Unknown emotion: {emotion}")


        #subject = self._format_subject(subject)
        clip_dir = os.path.join(self.root, f"sub{int(subject):02d}", video)
        print(f"Loading clip from: {clip_dir}")
        frames = sorted(os.listdir(clip_dir))

        images = []
        for f in frames:
            img = cv2.imread(os.path.join(clip_dir, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img) #(3, 224, 224)
            images.append(img)

        x = torch.stack(images)   # (T, C, H,  W)
        print(f"Frames done: {clip_dir}")

        # Temporal normalization
        if x.shape[0] > self.T:
            x = x[:self.T]
        elif x.shape[0] < self.T:
            pad = self.T - x.shape[0]
            x = torch.cat([x, x[-1:].repeat(pad,1,1,1)])

        return x, label
