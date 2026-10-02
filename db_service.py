# ============================================================
# backend/services/db_service.py
# SQLite helpers for prediction history.
# ============================================================

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join("crop_predictions.db")


def get_connection():
    """Open and return a SQLite connection with row_factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the predictions table if it does not exist."""
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            N           REAL    NOT NULL,
            P           REAL    NOT NULL,
            K           REAL    NOT NULL,
            temperature REAL    NOT NULL,
            humidity    REAL    NOT NULL,
            ph          REAL    NOT NULL,
            rainfall    REAL    NOT NULL,
            crop        TEXT    NOT NULL,
            confidence  REAL    NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(cleaned_data: dict, crop: str, confidence: float):
    """Insert a prediction record into the database."""
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO predictions
            (timestamp, N, P, K, temperature, humidity, ph, rainfall, crop, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cleaned_data["N"],
            cleaned_data["P"],
            cleaned_data["K"],
            cleaned_data["temperature"],
            cleaned_data["humidity"],
            cleaned_data["ph"],
            cleaned_data["rainfall"],
            crop,
            confidence,
        ),
    )
    conn.commit()
    conn.close()


def get_all_predictions(limit: int = 100):
    """Return the most recent predictions as a list of dicts."""
    conn = get_connection()
    cursor = conn.execute(
        """
        SELECT * FROM predictions ORDER BY id DESC LIMIT ?
        """,
        (limit,),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_dashboard_stats():
    """
    Return summary statistics calculated from actual prediction data.

    Returns
    -------
    dict with:
        total_predictions  : int
        most_common_crop   : str | None
        avg_confidence     : float
    """
    conn = get_connection()

    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    most_common = None
    if total > 0:
        row = conn.execute(
            """
            SELECT crop, COUNT(*) as cnt
            FROM predictions
            GROUP BY crop
            ORDER BY cnt DESC
            LIMIT 1
            """
        ).fetchone()
        most_common = row["crop"] if row else None

    avg_conf = 0.0
    if total > 0:
        row = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()
        avg_conf = round(float(row[0]) * 100, 1) if row[0] else 0.0

    conn.close()
    return {
        "total_predictions": total,
        "most_common_crop": most_common,
        "avg_confidence": avg_conf,
    }
