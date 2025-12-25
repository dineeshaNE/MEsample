import torch, torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class ToySequenceDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, d_model=64):
        self.x = torch.randn(num_samples, seq_len, d_model)
        self.y = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class MicroExpressionDataset(Dataset):
    def __init__(self, root_dir, seq_len=16, transform=None):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.samples = []
        self.label_map = {cls_name: i for i, cls_name in enumerate(sorted(os.listdir(root_dir)))}

        for cls_name in os.listdir(root_dir):
            cls_path = os.path.join(root_dir, cls_name)
            for clip in os.listdir(cls_path):
                clip_path = os.path.join(cls_path, clip)
                self.samples.append((clip_path, self.label_map[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_path, label = self.samples[idx]
        frame_paths = sorted([
            os.path.join(clip_path, fname)
            for fname in os.listdir(clip_path)
            if fname.endswith(('.jpg', '.png'))
        ])
        
        # Uniformly sample or pad to seq_len
        total = len(frame_paths)
        if total >= self.seq_len:
            indices = torch.linspace(0, total-1, steps=self.seq_len).long()
        else:
            indices = torch.cat([torch.arange(total), torch.full((self.seq_len - total,), total - 1)])

        frames = []
        for i in indices:
            img = Image.open(frame_paths[i]).convert('L')  # Grayscale
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        video = torch.stack(frames, dim=0)  # [T, C, H, W]
        return video, label
