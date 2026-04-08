MIN_OPEN_SCORE = 0.01
MAX_OPEN_SCORE = 0.99


def clamp_open_unit_interval(value: float) -> float:
    # Keep outputs strictly inside (0, 1), including non-finite edge cases.
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return MIN_OPEN_SCORE
    if numeric_value != numeric_value:  # NaN check without extra imports.
        return MIN_OPEN_SCORE
    if numeric_value == float("inf"):
        return MAX_OPEN_SCORE
    if numeric_value == float("-inf"):
        return MIN_OPEN_SCORE

    # Round first so serialized values still remain strictly inside (0, 1).
    rounded_value = round(numeric_value, 2)
    return min(max(rounded_value, MIN_OPEN_SCORE), MAX_OPEN_SCORE)


def clamp_closed_unit_interval(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 2)


def average_open_scores(values: list[float]) -> float:
    if not values:
        return MIN_OPEN_SCORE
    return clamp_open_unit_interval(sum(values) / len(values))


def sanitize_score_fields(payload):
    if hasattr(payload, "model_dump"):
        return sanitize_score_fields(payload.model_dump())
    if isinstance(payload, dict):
        sanitized = {}
        for key, value in payload.items():
            if key in {"score", "task_score", "average_score"} and isinstance(value, (int, float)):
                sanitized[key] = clamp_open_unit_interval(float(value))
            elif key.endswith("_score") and isinstance(value, (int, float)):
                sanitized[key] = clamp_open_unit_interval(float(value))
            elif key.endswith("_scores") and isinstance(value, list):
                sanitized[key] = [
                    clamp_open_unit_interval(float(item)) if isinstance(item, (int, float)) else sanitize_score_fields(item)
                    for item in value
                ]
            else:
                sanitized[key] = sanitize_score_fields(value)
        return sanitized
    if isinstance(payload, list):
        return [sanitize_score_fields(item) for item in payload]
    return payload
