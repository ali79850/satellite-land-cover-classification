"""
Downloads and extracts the EuroSAT RGB dataset from the official Zenodo source.

Source: https://zenodo.org/records/7711810
DOI: 10.5281/zenodo.7711810
License: MIT
"""

import hashlib
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ZENODO_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
EXPECTED_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"

RAW_DIR = Path("data/raw")
ZIP_PATH = RAW_DIR / "EuroSAT_RGB.zip"


def compute_md5(path: Path, chunk_size: int = 8192) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_dataset() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print(f"Zip already exists at {ZIP_PATH}, verifying checksum...")
    else:
        print(f"Downloading EuroSAT_RGB.zip (94.7 MB) from Zenodo...")
        urlretrieve(ZENODO_URL, ZIP_PATH)
        print("Download complete.")

    print("Verifying MD5 checksum...")
    actual_md5 = compute_md5(ZIP_PATH)
    if actual_md5 != EXPECTED_MD5:
        raise ValueError(
            f"MD5 mismatch! Expected {EXPECTED_MD5}, got {actual_md5}. "
            "The download may be corrupted — delete the zip and retry."
        )
    print("Checksum verified successfully.")

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_DIR)
    print(f"Extraction complete. Data available under {RAW_DIR}")


if __name__ == "__main__":
    download_dataset()