# ============================================================
# backend/utils/validation.py
# Input validation helpers for the prediction endpoint
# ============================================================

# Reasonable agronomic ranges for validation
VALID_RANGES = {
    "N":           (0,   140),   # Nitrogen  (kg/ha)
    "P":           (5,   145),   # Phosphorus (kg/ha)
    "K":           (5,   205),   # Potassium  (kg/ha)
    "temperature": (0,   50),    # °C
    "humidity":    (0,   100),   # %
    "ph":          (0,   14),    # pH scale
    "rainfall":    (0,   300),   # mm
}

REQUIRED_FIELDS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def validate_input(data: dict) -> tuple[bool, str, dict]:
    """
    Validate and coerce prediction input.

    Returns
    -------
    (is_valid, error_message, cleaned_data)
    """
    if not data:
        return False, "No input data provided.", {}

    cleaned = {}
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing required field: '{field}'.", {}

        raw = data[field]
        # Attempt numeric coercion
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return (
                False,
                f"Field '{field}' must be a number. Got: '{raw}'.",
                {},
            )

        if not (value == value):  # NaN check
            return False, f"Field '{field}' contains an invalid number.", {}

        lo, hi = VALID_RANGES[field]
        if not (lo <= value <= hi):
            return (
                False,
                f"Field '{field}' is out of valid range ({lo}–{hi}). Got: {value}.",
                {},
            )

        cleaned[field] = value

    return True, "", cleaned
