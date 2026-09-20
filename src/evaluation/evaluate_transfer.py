"""
Evaluates the frozen-backbone ResNet18 transfer-learning model on the
held-out TEST split. First and only time this model touches the test set.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import EuroSATDataset
from src.data.transforms_pretrained import get_eval_transforms_pretrained
from src.models.transfer_resnet18 import build_resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "transfer_frozen_best.pth"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "transfer_frozen_test_evaluation.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    test_ds = EuroSATDataset(split="test", transform=get_eval_transforms_pretrained())
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    class_names = test_ds.classes

    model = build_resnet18(num_classes=10, freeze_backbone=True).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Weighted F1:   {weighted_f1:.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("ResNet18 (Frozen) — Confusion Matrix (Test Set)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "transfer_frozen_confusion_matrix.png", dpi=150)
    plt.show()

    full_report = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nFull report saved to {REPORT_PATH}")


if __name__ == "__main__":
    evaluate()