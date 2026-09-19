"""
Creates a reproducible, stratified train/validation/test split for EuroSAT RGB.

Produces a single CSV manifest (data/processed/split_manifest.csv) with columns:
    filepath, class, split

Does NOT copy or move image files — the manifest is the source of truth for
which split each image belongs to.
"""

import csv
from pathlib import Path

import numpy as np

RAW_DIR = Path("data/raw/EuroSAT_RGB")
OUTPUT_CSV = Path("data/processed/split_manifest.csv")

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # implied, but kept explicit for clarity/documentation

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "Ratios must sum to 1.0"


def create_split() -> None:
    rng = np.random.default_rng(seed=SEED)

    classes = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
    rows = []

    for cls in classes:
        img_paths = sorted((RAW_DIR / cls).glob("*.jpg"))  # sorted = deterministic order
        n = len(img_paths)

        indices = rng.permutation(n)  # shuffled indices, seeded

        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # test gets the remainder, avoids rounding-drift issues

        train_idx = set(indices[:n_train])
        val_idx = set(indices[n_train:n_train + n_val])
        # everything else -> test

        for i, img_path in enumerate(img_paths):
            if i in train_idx:
                split = "train"
            elif i in val_idx:
                split = "val"
            else:
                split = "test"
            rows.append({
                "filepath": str(img_path.as_posix()),
                "class": cls,
                "split": split,
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "class", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written to {OUTPUT_CSV} ({len(rows)} rows)")


def verify_split() -> None:
    """Sanity-check the manifest: no leakage, correct proportions per class."""
    import csv as csv_mod
    from collections import defaultdict

    with open(OUTPUT_CSV) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)

    filepaths = [r["filepath"] for r in rows]
    assert len(filepaths) == len(set(filepaths)), "Duplicate filepaths found — leakage risk!"

    per_class_split = defaultdict(lambda: defaultdict(int))
    for r in rows:
        per_class_split[r["class"]][r["split"]] += 1

    print("\nPer-class split breakdown:")
    print(f"{'Class':25s} {'Train':>8s} {'Val':>8s} {'Test':>8s} {'Total':>8s}")
    for cls in sorted(per_class_split.keys()):
        counts = per_class_split[cls]
        total = sum(counts.values())
        print(f"{cls:25s} {counts['train']:>8d} {counts['val']:>8d} {counts['test']:>8d} {total:>8d}")

    total_rows = len(rows)
    split_totals = defaultdict(int)
    for r in rows:
        split_totals[r["split"]] += 1
    print(f"\nOverall: train={split_totals['train']} ({split_totals['train']/total_rows:.1%}), "
          f"val={split_totals['val']} ({split_totals['val']/total_rows:.1%}), "
          f"test={split_totals['test']} ({split_totals['test']/total_rows:.1%})")
    print("\nNo duplicate filepaths across the manifest — split is leak-free at the file level.")


if __name__ == "__main__":
    create_split()
    verify_split()