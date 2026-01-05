import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from config import IMG_SIZE, CLASS_NAMES
from PIL import Image

# ---------- Path handling ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_resnet34.pth")

device = "cpu"

# ---------- Model architecture (MUST MATCH TRAINING) ----------
model = models.resnet34(weights=None)

# Single-channel input (spectrogram)
model.conv1 = nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False
)

# ⚠️ IMPORTANT: Sequential FC (matches your checkpoint)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, len(CLASS_NAMES))
)

# ---------- Load weights ----------
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

model.eval()

# ---------- Preprocessing ----------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict(img: Image.Image) -> str:
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        pred = torch.argmax(outputs, dim=1).item()
    return CLASS_NAMES[pred]
