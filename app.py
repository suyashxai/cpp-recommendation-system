# ============================================================
# app.py  —  Flask application entry point
# ============================================================

import os
from flask import Flask, render_template

from backend.routes.main_routes import main_bp
from backend.routes.predict_routes import predict_bp
from backend.services.db_service import init_db


def create_app():
    """Application factory — creates and configures the Flask app."""
    app = Flask(__name__)

    # Secret key for session handling (change in production)
    app.secret_key = os.environ.get("SECRET_KEY", "crop-rec-dev-secret-2024")

    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp)

    # Initialise database tables
    init_db()

    # ── Custom error pages ────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template("500.html"), 500

    return app


# ── Run directly ─────────────────────────────────────────────────
if __name__ == "__main__":
    application = create_app()
    application.run(debug=True, host="0.0.0.0", port=5000)
