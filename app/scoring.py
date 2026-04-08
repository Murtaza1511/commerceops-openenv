MIN_OPEN_SCORE = 0.01
MAX_OPEN_SCORE = 0.99


def clamp_open_unit_interval(value: float) -> float:
    # Round first so anything serialized to two decimals still remains strictly inside (0, 1).
    rounded_value = round(value, 2)
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
            if key == "score" and isinstance(value, (int, float)):
                sanitized[key] = clamp_open_unit_interval(float(value))
            elif key.endswith("_score") and isinstance(value, (int, float)):
                sanitized[key] = clamp_open_unit_interval(float(value))
            else:
                sanitized[key] = sanitize_score_fields(value)
        return sanitized
    if isinstance(payload, list):
        return [sanitize_score_fields(item) for item in payload]
    return payload
