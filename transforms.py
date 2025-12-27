from torchvision import transforms

"""
One clip:
T = 30 frames
Each frame = (3, 224, 224)

Batch:
x = (B, T, 3, 224, 224)

Encoder output:
(B, T, 64)
"""

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