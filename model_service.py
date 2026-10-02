# ============================================================
# backend/services/model_service.py
# Loads the trained model and scaler; performs predictions.
# ============================================================

import os
import joblib
import numpy as np

# Paths relative to project root
MODEL_PATH = os.path.join("model", "crop_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

# Feature order must match training order exactly
FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

_model = None
_scaler = None


def _load_artifacts():
    """Load model and scaler from disk (once, cached in module globals)."""
    global _model, _scaler

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            "Please run 'python model/train_model.py' first."
        )

    _model = joblib.load(MODEL_PATH)

    if os.path.exists(SCALER_PATH):
        _scaler = joblib.load(SCALER_PATH)
    else:
        _scaler = None  # Some models do not require scaling


def get_model():
    """Return the loaded model, loading from disk if necessary."""
    if _model is None:
        _load_artifacts()
    return _model


def get_scaler():
    """Return the loaded scaler (may be None if not used)."""
    if _model is None:
        _load_artifacts()
    return _scaler


def predict(cleaned_data: dict) -> dict:
    """
    Run a single prediction.

    Parameters
    ----------
    cleaned_data : dict
        Validated numeric values for all seven features.

    Returns
    -------
    dict with keys:
        recommended_crop : str
        confidence       : float  (0–1)
        probabilities    : dict   {crop_name: probability}
    """
    model = get_model()
    scaler = get_scaler()

    # Build feature array in the exact training order
    features = np.array([[cleaned_data[f] for f in FEATURE_ORDER]])

    if scaler is not None:
        features = scaler.transform(features)

    crop = model.predict(features)[0]

    # Probability estimates (Random Forest / LogReg support predict_proba)
    if hasattr(model, "predict_proba"):
        proba_array = model.predict_proba(features)[0]
        classes = model.classes_
        probabilities = {str(c): round(float(p), 4) for c, p in zip(classes, proba_array)}
        confidence = float(max(proba_array))
    else:
        probabilities = {str(crop): 1.0}
        confidence = 1.0

    return {
        "recommended_crop": str(crop),
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
    }
