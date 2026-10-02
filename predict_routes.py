# ============================================================
# backend/routes/predict_routes.py
# Prediction-related Flask routes.
# ============================================================

from flask import Blueprint, request, jsonify, render_template, session

from backend.utils.validation import validate_input
from backend.services.model_service import predict
from backend.services.db_service import save_prediction

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["GET", "POST"])
def predict_view():
    """
    GET  → render the prediction form.
    POST → process form submission, render result page.
    """
    if request.method == "GET":
        return render_template("index.html")

    # --- Form POST ---
    form_data = request.form.to_dict()
    is_valid, error_msg, cleaned = validate_input(form_data)

    if not is_valid:
        return render_template("index.html", error=error_msg, form_data=form_data)

    try:
        result = predict(cleaned)
    except FileNotFoundError as exc:
        return render_template("index.html", error=str(exc), form_data=form_data)
    except Exception as exc:
        return render_template(
            "index.html",
            error="Prediction failed. Please try again.",
            form_data=form_data,
        )

    # Persist to SQLite
    try:
        save_prediction(cleaned, result["recommended_crop"], result["confidence"])
    except Exception:
        pass  # History is non-critical; don't fail the response

    return render_template(
        "result.html",
        crop=result["recommended_crop"].capitalize(),
        confidence=round(result["confidence"] * 100, 1),
        probabilities=result["probabilities"],
        inputs=cleaned,
    )


@predict_bp.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint for programmatic access.

    POST /api/predict
    Body: { "N": 90, "P": 42, "K": 43, "temperature": 25.5,
            "humidity": 80, "ph": 6.5, "rainfall": 200 }
    """
    data = request.get_json(silent=True)
    is_valid, error_msg, cleaned = validate_input(data)

    if not is_valid:
        return jsonify({"success": False, "error": error_msg}), 400

    try:
        result = predict(cleaned)
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    except Exception:
        return jsonify({"success": False, "error": "Prediction failed."}), 500

    try:
        save_prediction(cleaned, result["recommended_crop"], result["confidence"])
    except Exception:
        pass

    return jsonify(
        {
            "success": True,
            "recommended_crop": result["recommended_crop"],
            "confidence": result["confidence"],
        }
    )
