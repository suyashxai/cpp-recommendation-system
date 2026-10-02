"""
tests/test_api.py
=================
API and validation tests for the Flask application.

Run with:  pytest tests/ -v
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── App Fixture ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    from app import create_app
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture(scope="module")
def client(app):
    return app.test_client()


# ── Helper ────────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "N": 90, "P": 42, "K": 43,
    "temperature": 25.5, "humidity": 80.0, "ph": 6.5, "rainfall": 200.0,
}


# ── Health Endpoint ───────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_json_structure(self, client):
        resp = client.get("/api/health")
        data = resp.get_json()
        assert "status" in data
        assert "model_loaded" in data
        assert "message" in data


# ── Page Routes ───────────────────────────────────────────────────

class TestPageRoutes:
    def test_home_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_about_returns_200(self, client):
        resp = client.get("/about")
        assert resp.status_code == 200

    def test_history_returns_200(self, client):
        resp = client.get("/history")
        assert resp.status_code == 200

    def test_predict_get_returns_200(self, client):
        resp = client.get("/predict")
        assert resp.status_code == 200

    def test_404_page(self, client):
        resp = client.get("/nonexistent-page-xyz")
        assert resp.status_code == 404


# ── Predict API ───────────────────────────────────────────────────

class TestPredictAPI:
    def test_valid_input_returns_200(self, client):
        """A valid JSON payload must return 200 and success=True."""
        import os
        if not os.path.exists(os.path.join("model", "crop_model.pkl")):
            pytest.skip("Model not trained yet. Run 'python model/train_model.py' first.")
        resp = client.post("/api/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "recommended_crop" in data
        assert "confidence" in data

    def test_confidence_in_valid_range(self, client):
        import os
        if not os.path.exists(os.path.join("model", "crop_model.pkl")):
            pytest.skip("Model not trained yet.")
        resp = client.post("/api/predict", json=VALID_PAYLOAD)
        data = resp.get_json()
        if data.get("success"):
            assert 0.0 <= data["confidence"] <= 1.0

    def test_missing_field_returns_400(self, client):
        """Payload missing 'N' must return 400."""
        payload = dict(VALID_PAYLOAD)
        del payload["N"]
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False
        assert "N" in data["error"]

    def test_non_numeric_temperature_returns_400(self, client):
        payload = dict(VALID_PAYLOAD)
        payload["temperature"] = "hot"
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_negative_rainfall_returns_400(self, client):
        payload = dict(VALID_PAYLOAD)
        payload["rainfall"] = -10
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 400

    def test_ph_above_14_returns_400(self, client):
        payload = dict(VALID_PAYLOAD)
        payload["ph"] = 15
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 400

    def test_empty_body_returns_400(self, client):
        resp = client.post("/api/predict", json=None,
                           content_type="application/json")
        assert resp.status_code == 400

    def test_no_content_type_returns_400(self, client):
        resp = client.post("/api/predict", data="not json")
        assert resp.status_code == 400


# ── Validation Unit Tests ─────────────────────────────────────────

class TestValidation:
    """Unit-test the validation module directly, no Flask required."""

    def setup_method(self):
        from backend.utils.validation import validate_input
        self.validate = validate_input

    def test_valid_data_passes(self):
        ok, msg, cleaned = self.validate(VALID_PAYLOAD)
        assert ok is True
        assert msg == ""
        assert len(cleaned) == 7

    def test_all_values_coerced_to_float(self):
        # Pass strings — should be coerced
        payload = {k: str(v) for k, v in VALID_PAYLOAD.items()}
        ok, msg, cleaned = self.validate(payload)
        assert ok is True
        for v in cleaned.values():
            assert isinstance(v, float)

    def test_missing_field(self):
        payload = dict(VALID_PAYLOAD)
        del payload["K"]
        ok, msg, _ = self.validate(payload)
        assert ok is False
        assert "K" in msg

    def test_text_value_fails(self):
        payload = dict(VALID_PAYLOAD)
        payload["humidity"] = "abc"
        ok, msg, _ = self.validate(payload)
        assert ok is False

    def test_ph_below_zero_fails(self):
        payload = dict(VALID_PAYLOAD)
        payload["ph"] = -1
        ok, msg, _ = self.validate(payload)
        assert ok is False

    def test_temperature_above_50_fails(self):
        payload = dict(VALID_PAYLOAD)
        payload["temperature"] = 51
        ok, msg, _ = self.validate(payload)
        assert ok is False

    def test_empty_dict_fails(self):
        ok, msg, _ = self.validate({})
        assert ok is False

    def test_none_fails(self):
        ok, msg, _ = self.validate(None)
        assert ok is False
