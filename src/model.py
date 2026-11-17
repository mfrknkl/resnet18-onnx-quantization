from pathlib import Path
import torch
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "models" / "best_resnet18_cifar100.pth"

def load_model(checkpoint_path=CHECKPOINT_PATH):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model
