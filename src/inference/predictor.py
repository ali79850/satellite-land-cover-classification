"""
Standalone inference pipeline for the final trained model.
No dependency on training code, dataset manifest, or evaluation scripts —
this is the self-contained interface used by the Streamlit app and tests.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "final_model.pth"
CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"


def _build_model(num_classes: int) -> torch.nn.Module:
    """Builds a bare ResNet18 architecture — weights loaded separately."""
    model = resnet18(weights=None)  # no ImageNet download needed; we load our own trained weights
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


class ModelPredictor:
    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH, config_path: Path = CONFIG_PATH):
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {checkpoint_path}. "
                "Run src/models/save_final_model.py first (see Phase 10)."
            )
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found at {config_path}.")

        with open(config_path) as f:
            self.config = json.load(f)

        self.idx_to_class = self.config["idx_to_class"]
        self.input_size = tuple(self.config["input_size"])
        mean = self.config["normalization"]["mean"]
        std = self.config["normalization"]["std"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = _build_model(num_classes=self.config["num_classes"])
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def predict(self, image: Image.Image) -> dict:
        """
        image: a PIL Image (any mode/size — will be converted and resized).

        Returns:
            {
                "class": str,
                "confidence": float,
                "probabilities": {class_name: float, ...}
            }
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_class = self.idx_to_class[str(pred_idx)]
        confidence = float(probs[pred_idx])

        probabilities = {
            self.idx_to_class[str(i)]: float(probs[i])
            for i in range(len(probs))
        }

        return {
            "class": pred_class,
            "confidence": confidence,
            "probabilities": probabilities,
        }