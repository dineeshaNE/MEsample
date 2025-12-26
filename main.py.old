# main  calls MambaMER

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from MERModel import FrameFeatureExtractor, MambaMER


model = MambaMER(in_channels=1, num_classes=5, seq_len=16)  # For grayscale input

dummy_input = torch.randn(8, 16, 1, 64, 64)  # [Batch, Time, Channels, H, W]
output = model(dummy_input)
print(output.shape)  # Expected: [8, 5]