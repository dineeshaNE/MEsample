"""
CASME II video clip
      │
      ▼
Face Detection + Alignment
      │
      ▼
Frame Encoder (CNN / ViT patch embed)
      │
      ▼
Sequence Tensor x (B,T,D)
      │
      ▼
     Mamba
      │
      ▼
Hidden states h (B,T,D)
      │
      ▼
Select last timestep
      │
      ▼
Temporal summary h_last (B,D)
      │
      ▼
Linear classifier
      │
      ▼
Emotion logits (B,C)
      │
      ▼
CrossEntropyLoss


Video → (B,T,C,H,W)
→ CNN → (B,T,64)
→ Mamba → (B,T,64)
→ Last timestep → (B,64)
→ Linear → (B,Classes)
→ Loss
"""