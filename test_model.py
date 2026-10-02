"""
tests/test_model.py
===================
Automated tests for the trained ML model.

Run with:  pytest tests/ -v
"""

import os
import sys
import pytest

# Add project root to path so imports work when running from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH  = os.path.join("model", "crop_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

EXPECTED_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Known crop classes from the Crop Recommendation Dataset
KNOWN_CROPS = {
    "apple", "banana", "blackgram", "chickpea", "coconut", "coffee",
    "cotton", "grapes", "jute", "kidneybeans", "lentil", "maize",
    "mango", "motherbeans", "mungbean", "muskmelon", "orange", "papaya",
    "pigeonpeas", "pomegranate", "rice", "watermelon",
}

SAMPLE_INPUT = {
    "N": 90, "P": 42, "K": 43,
    "temperature": 25.5, "humidity": 80.0, "ph": 6.5, "rainfall": 200.0,
}


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    """Load the trained model once for all tests in this module."""
    joblib = pytest.importorskip("joblib")
    if not os.path.exists(MODEL_PATH):
        pytest.skip(
            f"Model file not found at '{MODEL_PATH}'. "
            "Run 'python model/train_model.py' to train the model first."
        )
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def scaler():
    joblib = pytest.importorskip("joblib")
    if not os.path.exists(SCALER_PATH):
        return None
    return joblib.load(SCALER_PATH)


# ── Tests ─────────────────────────────────────────────────────────

class TestModelLoads:
    def test_model_file_exists(self):
        """Model file must exist after training."""
        assert os.path.exists(MODEL_PATH), (
            f"Model file not found at '{MODEL_PATH}'. "
            "Run 'python model/train_model.py' first."
        )

    def test_scaler_file_exists(self):
        """Scaler file must exist after training."""
        assert os.path.exists(SCALER_PATH), (
            f"Scaler file not found at '{SCALER_PATH}'."
        )

    def test_model_loads_without_error(self, model):
        assert model is not None

    def test_model_has_correct_feature_count(self, model):
        """Model must expect exactly 7 features."""
        if hasattr(model, "n_features_in_"):
            assert model.n_features_in_ == len(EXPECTED_FEATURES), (
                f"Model expects {model.n_features_in_} features, "
                f"but {len(EXPECTED_FEATURES)} are required."
            )

    def test_model_has_classes(self, model):
        """Model must expose class labels."""
        assert hasattr(model, "classes_"), "Model has no 'classes_' attribute."
        assert len(model.classes_) > 0

    def test_model_knows_expected_crops(self, model):
        """At least some well-known crops should be in the model's class list."""
        model_classes = {str(c).lower() for c in model.classes_}
        overlap = KNOWN_CROPS & model_classes
        assert len(overlap) >= 5, (
            f"Expected model to know standard crops. Overlap found: {overlap}"
        )


class TestModelPrediction:
    def test_single_prediction_runs(self, model, scaler):
        import numpy as np
        features = [[SAMPLE_INPUT[f] for f in EXPECTED_FEATURES]]
        arr = np.array(features)
        if scaler is not None:
            arr = scaler.transform(arr)
        result = model.predict(arr)
        assert len(result) == 1

    def test_prediction_is_known_crop(self, model, scaler):
        import numpy as np
        features = [[SAMPLE_INPUT[f] for f in EXPECTED_FEATURES]]
        arr = np.array(features)
        if scaler is not None:
            arr = scaler.transform(arr)
        crop = model.predict(arr)[0]
        assert str(crop).lower() in KNOWN_CROPS, (
            f"Predicted crop '{crop}' is not in the known crops list."
        )

    def test_predict_proba_available(self, model, scaler):
        import numpy as np
        assert hasattr(model, "predict_proba"), (
            "Model does not support probability estimates."
        )
        features = [[SAMPLE_INPUT[f] for f in EXPECTED_FEATURES]]
        arr = np.array(features)
        if scaler is not None:
            arr = scaler.transform(arr)
        proba = model.predict_proba(arr)[0]
        assert abs(sum(proba) - 1.0) < 1e-4, "Probabilities do not sum to 1."

    def test_confidence_in_valid_range(self, model, scaler):
        import numpy as np
        features = [[SAMPLE_INPUT[f] for f in EXPECTED_FEATURES]]
        arr = np.array(features)
        if scaler is not None:
            arr = scaler.transform(arr)
        proba = model.predict_proba(arr)[0]
        confidence = float(max(proba))
        assert 0.0 <= confidence <= 1.0
