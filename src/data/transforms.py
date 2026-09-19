"""
Reusable torchvision transform pipelines for EuroSAT RGB.

Train transforms include augmentation appropriate for overhead satellite
imagery (flips, 90-degree rotations, mild color jitter). Eval transforms
(val/test) are deterministic — no augmentation, since we need consistent,
repeatable evaluation.
"""

import json
from pathlib import Path

import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NORM_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "normalization_stats.json"

def load_normalization_stats() -> tuple[list[float], list[float]]:
    with open(NORM_STATS_PATH) as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]


def get_train_transforms() -> T.Compose:
    mean, std = load_normalization_stats()
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomChoice([
            T.RandomRotation(degrees=(0, 0)),     # no rotation
            T.RandomRotation(degrees=(90, 90)),
            T.RandomRotation(degrees=(180, 180)),
            T.RandomRotation(degrees=(270, 270)),
        ]),
        T.ColorJitter(brightness=0.1, contrast=0.1),  # mild only, see Phase 6 rationale
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_eval_transforms() -> T.Compose:
    """Used for validation and test — no augmentation, deterministic."""
    mean, std = load_normalization_stats()
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])