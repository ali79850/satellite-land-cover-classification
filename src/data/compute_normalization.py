"""
Computes per-channel (RGB) mean and std over the TRAINING split only.

Using only the training split avoids leaking validation/test pixel
statistics into preprocessing (see Phase 5 leakage discussion).
"""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

MANIFEST_PATH = Path("data/processed/split_manifest.csv")
OUTPUT_PATH = Path("data/processed/normalization_stats.json")


def compute_normalization() -> dict:
    with open(MANIFEST_PATH) as f:
        reader = csv.DictReader(f)
        train_paths = [row["filepath"] for row in reader if row["split"] == "train"]

    print(f"Computing normalization stats over {len(train_paths)} training images...")

    # Running sums (avoids loading all images into memory at once)
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for i, path in enumerate(train_paths):
        arr = np.array(Image.open(path)).astype(np.float64) / 255.0  # (H, W, 3)
        pixel_sum += arr.reshape(-1, 3).sum(axis=0)
        pixel_sq_sum += (arr.reshape(-1, 3) ** 2).sum(axis=0)
        n_pixels += arr.shape[0] * arr.shape[1]

        if (i + 1) % 5000 == 0:
            print(f"  processed {i + 1}/{len(train_paths)}")

    mean = pixel_sum / n_pixels
    variance = (pixel_sq_sum / n_pixels) - (mean ** 2)
    std = np.sqrt(variance)

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "computed_over": "train split only",
        "n_images": len(train_paths),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nMean (RGB): {mean}")
    print(f"Std  (RGB): {std}")
    print(f"Saved to {OUTPUT_PATH}")
    return stats


if __name__ == "__main__":
    compute_normalization()