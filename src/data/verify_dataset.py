"""
Verifies the integrity and structure of the downloaded EuroSAT RGB dataset.

Checks:
- Expected class folders are present
- Image counts per class
- No corrupted/unreadable images
- Consistent image dimensions and channel count
"""

import json
from pathlib import Path
from PIL import Image

RAW_DIR = Path("data/raw/EuroSAT_RGB")
REPORT_PATH = Path("reports/dataset_verification.json")

EXPECTED_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


def verify_dataset() -> dict:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"{RAW_DIR} not found. Run download_dataset.py first.")

    found_classes = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
    missing = set(EXPECTED_CLASSES) - set(found_classes)
    unexpected = set(found_classes) - set(EXPECTED_CLASSES)

    class_counts = {}
    corrupted_files = []
    dimensions_seen = set()
    channels_seen = set()

    for class_name in found_classes:
        class_dir = RAW_DIR / class_name
        image_files = list(class_dir.glob("*.jpg"))
        class_counts[class_name] = len(image_files)

        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    img.verify()  # checks for corruption
                # reopen after verify() — verify() invalidates the file handle
                with Image.open(img_path) as img:
                    dimensions_seen.add(img.size)
                    channels_seen.add(len(img.getbands()))
            except Exception as e:
                corrupted_files.append({"file": str(img_path), "error": str(e)})

    total_images = sum(class_counts.values())

    report = {
        "expected_classes": EXPECTED_CLASSES,
        "found_classes": found_classes,
        "missing_classes": sorted(missing),
        "unexpected_classes": sorted(unexpected),
        "total_images": total_images,
        "images_per_class": class_counts,
        "unique_dimensions": sorted(list(dimensions_seen)),
        "unique_channel_counts": sorted(list(channels_seen)),
        "num_corrupted": len(corrupted_files),
        "corrupted_files": corrupted_files,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return report


def print_summary(report: dict) -> None:
    print("=" * 50)
    print("DATASET VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Classes found: {len(report['found_classes'])}/10")
    if report["missing_classes"]:
        print(f"  MISSING: {report['missing_classes']}")
    if report["unexpected_classes"]:
        print(f"  UNEXPECTED: {report['unexpected_classes']}")
    print(f"\nTotal images: {report['total_images']}")
    print("\nImages per class:")
    for cls, count in report["images_per_class"].items():
        print(f"  {cls:25s} {count}")
    print(f"\nUnique image dimensions: {report['unique_dimensions']}")
    print(f"Unique channel counts: {report['unique_channel_counts']}")
    print(f"\nCorrupted images: {report['num_corrupted']}")
    if report["corrupted_files"]:
        for cf in report["corrupted_files"][:5]:
            print(f"  {cf['file']}: {cf['error']}")
    print("=" * 50)
    print(f"\nFull report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    report = verify_dataset()
    print_summary(report)