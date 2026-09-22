"""
Manual sanity check for the inference pipeline — NOT the formal test suite
(see Phase 16 for that). Run this once to confirm predict() works end-to-end
on a real image before wiring it into Streamlit.
"""

from pathlib import Path
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.inference.predictor import ModelPredictor

# Grab a real image from the raw dataset for a manual check
SAMPLE_IMAGE = Path("data/raw/EuroSAT_RGB/Forest/Forest_1.jpg")


def main():
    predictor = ModelPredictor()
    image = Image.open(SAMPLE_IMAGE)

    result = predictor.predict(image)

    print(f"Predicted class: {result['class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nFull probability distribution:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        print(f"  {cls:22s} {prob:.4f}")

    prob_sum = sum(result["probabilities"].values())
    print(f"\nProbabilities sum to: {prob_sum:.6f} (should be ~1.0)")


if __name__ == "__main__":
    main()