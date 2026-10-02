# ============================================================
# backend/routes/main_routes.py
# Non-prediction routes: home, about, history, health.
# ============================================================

from flask import Blueprint, render_template, jsonify

from backend.services.db_service import get_all_predictions, get_dashboard_stats
from backend.services.model_service import FEATURE_ORDER

main_bp = Blueprint("main", __name__)

# Number of crop classes the model was trained on — updated at runtime
_N_CLASSES = 22  # default; overwritten by app.py after model loads


@main_bp.route("/")
def home():
    """Dashboard / home page with live statistics."""
    try:
        stats = get_dashboard_stats()
    except Exception:
        stats = {"total_predictions": 0, "most_common_crop": "N/A", "avg_confidence": 0.0}

    # Attempt to read n_classes from loaded model
    try:
        from backend.services.model_service import get_model
        model = get_model()
        n_classes = len(model.classes_) if hasattr(model, "classes_") else _N_CLASSES
    except Exception:
        n_classes = _N_CLASSES

    stats["n_classes"] = n_classes
    return render_template("index.html", stats=stats)


@main_bp.route("/about")
def about():
    return render_template("about.html")


@main_bp.route("/history")
def history():
    try:
        predictions = get_all_predictions(limit=100)
    except Exception:
        predictions = []
    return render_template("history.html", predictions=predictions)


@main_bp.route("/api/health")
def health():
    """Health-check endpoint."""
    model_ready = False
    try:
        from backend.services.model_service import get_model
        get_model()
        model_ready = True
    except Exception:
        pass

    return jsonify(
        {
            "status": "ok" if model_ready else "degraded",
            "model_loaded": model_ready,
            "message": "Crop Recommendation API is running.",
        }
    )
