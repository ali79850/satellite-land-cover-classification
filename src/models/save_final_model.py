"""
Packages the final selected model (fine-tuned ResNet18) for deployment:
- Copies the model weights to a clearly-named final location
- Saves the class index mapping
- Saves the preprocessing/model configuration needed to reproduce inference
"""

import json
import shutil
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import EuroSATDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CHECKPOINT = PROJECT_ROOT / "models" / "finetune_best.pth"
FINAL_CHECKPOINT = PROJECT_ROOT / "models" / "final_model.pth"
CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"


def package_final_model():
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"{SOURCE_CHECKPOINT} not found. Make sure finetune_best.pth "
            "was downloaded from Colab and placed in models/."
        )

    shutil.copy(SOURCE_CHECKPOINT, FINAL_CHECKPOINT)
    print(f"Copied {SOURCE_CHECKPOINT.name} -> {FINAL_CHECKPOINT.name}")

    # Derive class mapping the same way the dataset class does, so it's
    # guaranteed consistent with what the model was actually trained on.
    train_ds = EuroSATDataset(split="train")
    class_to_idx = train_ds.class_to_idx

    config = {
        "model_name": "resnet18_finetuned",
        "architecture": "torchvision.models.resnet18",
        "num_classes": 10,
        "class_to_idx": class_to_idx,
        "idx_to_class": {v: k for k, v in class_to_idx.items()},
        "input_size": [224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "note": "ImageNet stats — required since backbone is ImageNet-pretrained, per Phase 8 decision",
        },
        "training_summary": {
            "test_accuracy": 0.9793,
            "macro_f1": 0.9787,
            "weighted_f1": 0.9792,
            "selected_from_experiment": "Experiment 3 - fine-tuned (Phase 9)",
        },
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved model config to {CONFIG_PATH}")
    print(f"\nClass mapping: {class_to_idx}")


if __name__ == "__main__":
    package_final_model()