"""
Systematic error analysis on the test set: identifies all misclassified
examples, groups them by (true, predicted) pair, and saves a visual grid
for the most common confusion patterns.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import EuroSATDataset
from src.data.transforms_pretrained import get_eval_transforms_pretrained
from src.models.transfer_resnet18 import build_resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "final_model.pth"
CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures" / "error_analysis"
REPORT_PATH = PROJECT_ROOT / "reports" / "error_analysis.json"

DEVICE = torch.device("cpu")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr, 0, 1)


def collect_errors(model, test_ds, idx_to_class):
    errors = []  # list of dicts: {idx, true_class, pred_class, confidence}
    with torch.no_grad():
        for i in range(len(test_ds)):
            img, label = test_ds[i]
            output = model(img.unsqueeze(0).to(DEVICE))
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            conf = probs[0, pred].item()

            if pred != label:
                errors.append({
                    "idx": i,
                    "true_class": idx_to_class[str(label)],
                    "pred_class": idx_to_class[str(pred)],
                    "confidence": conf,
                })
    return errors


def summarize_errors(errors):
    pair_counts = defaultdict(int)
    for e in errors:
        pair_counts[(e["true_class"], e["pred_class"])] += 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])
    return sorted_pairs


def plot_error_grid(errors, test_ds, true_class, pred_class, save_path, max_examples=6):
    """Show up to max_examples misclassified images for one specific confusion pair."""
    matching = [e for e in errors if e["true_class"] == true_class and e["pred_class"] == pred_class]
    matching = matching[:max_examples]

    if not matching:
        return

    n = len(matching)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, e in zip(axes, matching):
        img, _ = test_ds[e["idx"]]
        img_np = denormalize(img)
        ax.imshow(img_np)
        ax.set_title(f"True: {true_class}\nPred: {pred_class} ({e['confidence']:.0%})", fontsize=9)
        ax.axis("off")

    plt.suptitle(f"{true_class} misclassified as {pred_class}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    idx_to_class = config["idx_to_class"]

    test_ds = EuroSATDataset(split="test", transform=get_eval_transforms_pretrained())

    model = build_resnet18(num_classes=10, freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    print(f"Scanning all {len(test_ds)} test images for errors...")
    errors = collect_errors(model, test_ds, idx_to_class)
    print(f"\nTotal errors: {len(errors)} / {len(test_ds)} ({100*len(errors)/len(test_ds):.2f}%)")

    sorted_pairs = summarize_errors(errors)
    print("\nConfusion pairs, ranked by frequency:")
    for (true_c, pred_c), count in sorted_pairs:
        print(f"  {true_c:22s} -> {pred_c:22s}: {count}")

    # Visualize the top 3 most common confusion pairs
    print("\nGenerating error grids for top 3 confusion pairs...")
    for (true_c, pred_c), count in sorted_pairs[:3]:
        safe_name = f"{true_c}_as_{pred_c}".replace(" ", "_")
        plot_error_grid(errors, test_ds, true_c, pred_c, OUTPUT_DIR / f"errors_{safe_name}.png")

    # Save full error log
    with open(REPORT_PATH, "w") as f:
        json.dump({
            "total_errors": len(errors),
            "total_test_images": len(test_ds),
            "error_rate": len(errors) / len(test_ds),
            "confusion_pairs_ranked": [
                {"true_class": t, "pred_class": p, "count": c} for (t, p), c in sorted_pairs
            ],
            "all_errors": errors,
        }, f, indent=2)
    print(f"\nFull error log saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()