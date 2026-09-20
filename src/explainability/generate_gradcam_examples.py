"""
Generates Grad-CAM visualizations for three categories of test predictions:
- Correct, high confidence
- Correct, low confidence
- Incorrect

Saves each as original | heatmap overlay, under reports/figures/explainability/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import random
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import EuroSATDataset
from src.data.transforms_pretrained import get_eval_transforms_pretrained
from src.models.transfer_resnet18 import build_resnet18
from src.explainability.gradcam import GradCAM, overlay_heatmap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "final_model.pth"
CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures" / "explainability"

DEVICE = torch.device("cpu")  # small workload, CPU is fine for a handful of images

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr, 0, 1)


def find_examples(model, test_ds, idx_to_class, order):
    correct_high, correct_low, incorrect = None, None, None
    lowest_correct_conf = 1.0
    highest_correct_conf = 0.0

    with torch.no_grad():
        for count, i in enumerate(order):
            img, label = test_ds[i]
            output = model(img.unsqueeze(0).to(DEVICE))
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            conf = probs[0, pred].item()

            if pred == label:
                if conf > highest_correct_conf:
                    highest_correct_conf = conf
                    correct_high = i
                if conf < lowest_correct_conf:
                    lowest_correct_conf = conf
                    correct_low = i
            else:
                if incorrect is None:
                    incorrect = i

            if correct_high is not None and correct_low is not None and incorrect is not None and count > 500:
                break

    return correct_high, correct_low, incorrect

def visualize_example(model, gradcam, test_ds, idx, idx_to_class, title_prefix, save_path):
    img, true_label = test_ds[idx]
    input_tensor = img.unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(False)

    heatmap, pred_class, confidence = gradcam.generate(input_tensor.clone())

    img_np = denormalize(img)
    overlay = overlay_heatmap(img_np, heatmap)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original\nTrue: {idx_to_class[str(true_label)]}")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    correctness = "Correct" if pred_class == true_label else "INCORRECT"
    axes[1].set_title(f"Grad-CAM ({correctness})\nPred: {idx_to_class[str(pred_class)]} ({confidence:.1%})")
    axes[1].axis("off")

    plt.suptitle(title_prefix)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")
    print(f"  True: {idx_to_class[str(true_label)]}, Predicted: {idx_to_class[str(pred_class)]}, Confidence: {confidence:.1%}\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    idx_to_class = config["idx_to_class"]

    test_ds = EuroSATDataset(split="test", transform=get_eval_transforms_pretrained())
    test_ds = EuroSATDataset(split="test", transform=get_eval_transforms_pretrained())

    import random
    random.seed(42)
    shuffled_indices = list(range(len(test_ds)))
    random.shuffle(shuffled_indices)


    model = build_resnet18(num_classes=10, freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    gradcam = GradCAM(model, target_layer=model.layer4)

    print("Scanning test set for example categories...")
    correct_high, correct_low, incorrect = find_examples(model, test_ds, idx_to_class, shuffled_indices)

    print(f"Found: correct_high_conf idx={correct_high}, correct_low_conf idx={correct_low}, incorrect idx={incorrect}")

    if correct_high is not None:
        visualize_example(model, gradcam, test_ds, correct_high, idx_to_class,
                           "Correct, High Confidence", OUTPUT_DIR / "example_correct_high_confidence.png")
    if correct_low is not None:
        visualize_example(model, gradcam, test_ds, correct_low, idx_to_class,
                           "Correct, Low Confidence", OUTPUT_DIR / "example_correct_low_confidence.png")
    if incorrect is not None:
        visualize_example(model, gradcam, test_ds, incorrect, idx_to_class,
                           "Incorrect Prediction", OUTPUT_DIR / "example_incorrect.png")


if __name__ == "__main__":
    main()