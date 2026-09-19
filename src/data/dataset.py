"""
PyTorch Dataset for EuroSAT RGB, driven by the split manifest CSV.
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"


class EuroSATDataset(Dataset):
    def __init__(self, split: str, transform=None, manifest_path: Path = MANIFEST_PATH):
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"
        self.transform = transform

        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            self.samples = [row for row in reader if row["split"] == split]

        self.classes = sorted(set(row["class"] for row in self.samples))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        image_path = PROJECT_ROOT / row["filepath"]
        image = Image.open(image_path).convert("RGB")
        label = self.class_to_idx[row["class"]]

        if self.transform:
            image = self.transform(image)

        return image, label