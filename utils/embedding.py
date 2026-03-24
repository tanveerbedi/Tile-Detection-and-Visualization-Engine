import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== MODEL =====
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.fc = torch.nn.Identity()
model.eval().to(DEVICE)

tfm = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])

def l2norm(x, eps=1e-12):
    n = np.linalg.norm(x)
    return x / max(n, eps)


@torch.no_grad()
def embed_crop_tta(pil_img: Image.Image) -> np.ndarray:
    rotations = [0, 90, 180, 270]
    batch = []

    for r in rotations:
        img = pil_img.rotate(r, expand=True).convert("RGB")
        batch.append(tfm(img))

    x = torch.stack(batch).to(DEVICE)
    feat = model(x)
    feat = feat.cpu().numpy().astype(np.float32)
    feat = feat.mean(axis=0)

    return l2norm(feat)


def get_embedding_from_path(image_path: str):
    img = Image.open(image_path).convert("RGB")
    return embed_crop_tta(img)

