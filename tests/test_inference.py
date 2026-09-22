"""
Automated tests for the inference pipeline (src/inference/predictor.py).

Run with: pytest tests/ -v
"""

from pathlib import Path

import pytest
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import ModelPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "EuroSAT_RGB" / "Forest" / "Forest_1.jpg"

EXPECTED_CLASSES = {
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
}


@pytest.fixture(scope="module")
def predictor():
    """Loaded once per test module, not per test — model loading is expensive."""
    return ModelPredictor()


@pytest.fixture(scope="module")
def sample_image():
    if not SAMPLE_IMAGE_PATH.exists():
        pytest.skip(f"Sample image not found at {SAMPLE_IMAGE_PATH}. Run Phase 3's dataset download first.")
    return Image.open(SAMPLE_IMAGE_PATH)


class TestModelLoading:
    def test_predictor_loads_without_error(self, predictor):
        assert predictor is not None
        assert predictor.model is not None

    def test_model_is_in_eval_mode(self, predictor):
        assert not predictor.model.training


class TestClassMapping:
    def test_all_ten_classes_present(self, predictor):
        classes_in_config = set(predictor.idx_to_class.values())
        assert classes_in_config == EXPECTED_CLASSES

    def test_exactly_ten_classes(self, predictor):
        assert len(predictor.idx_to_class) == 10

    def test_indices_are_contiguous(self, predictor):
        indices = sorted(int(k) for k in predictor.idx_to_class.keys())
        assert indices == list(range(10))


class TestPreprocessing:
    def test_transform_produces_correct_shape(self, predictor, sample_image):
        tensor = predictor.transform(sample_image.convert("RGB"))
        assert tensor.shape == (3, 224, 224)

    def test_transform_handles_grayscale_input(self, predictor, sample_image):
        grayscale = sample_image.convert("L")  # simulate an unusual input mode
        result = predictor.predict(grayscale)
        assert result["class"] in EXPECTED_CLASSES

    def test_transform_handles_rgba_input(self, predictor, sample_image):
        rgba = sample_image.convert("RGBA")  # e.g. PNG with alpha channel
        result = predictor.predict(rgba)
        assert result["class"] in EXPECTED_CLASSES


class TestInferenceOutput:
    def test_predict_returns_expected_keys(self, predictor, sample_image):
        result = predictor.predict(sample_image)
        assert set(result.keys()) == {"class", "confidence", "probabilities"}

    def test_predicted_class_is_valid(self, predictor, sample_image):
        result = predictor.predict(sample_image)
        assert result["class"] in EXPECTED_CLASSES

    def test_confidence_is_valid_probability(self, predictor, sample_image):
        result = predictor.predict(sample_image)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_contains_all_classes(self, predictor, sample_image):
        result = predictor.predict(sample_image)
        assert set(result["probabilities"].keys()) == EXPECTED_CLASSES

    def test_probabilities_sum_to_one(self, predictor, sample_image):
        result = predictor.predict(sample_image)
        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-4  # small tolerance for float rounding

    def test_predicted_class_matches_max_probability(self, predictor, sample_image):
        """The returned 'class' should always be the argmax of 'probabilities'."""
        result = predictor.predict(sample_image)
        max_class = max(result["probabilities"], key=result["probabilities"].get)
        assert result["class"] == max_class

    def test_known_forest_image_predicts_forest(self, predictor, sample_image):
        """Weak correctness check using a known-easy example (Forest had F1=0.99)."""
        result = predictor.predict(sample_image)
        assert result["class"] == "Forest"


class TestInvalidInputHandling:
    def test_predict_raises_on_non_image_input(self, predictor):
        with pytest.raises(AttributeError):
            predictor.predict("not an image")  # str has no .convert(), should fail clearly

    def test_predict_handles_tiny_image(self, predictor):
        """A 1x1 image is a valid PIL Image but an extreme edge case."""
        tiny_image = Image.new("RGB", (1, 1), color=(128, 128, 128))
        result = predictor.predict(tiny_image)  # should not crash — Resize handles it
        assert result["class"] in EXPECTED_CLASSES

    def test_predict_handles_very_large_image(self, predictor):
        """Ensures Resize correctly downscales oversized input, not just upscales."""
        large_image = Image.new("RGB", (2000, 2000), color=(50, 150, 50))
        result = predictor.predict(large_image)
        assert result["class"] in EXPECTED_CLASSES