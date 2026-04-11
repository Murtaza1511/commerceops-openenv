from app.scoring import MIN_OPEN_SCORE, clamp_open_unit_interval


def _clamp_task_score(score):
    return clamp_open_unit_interval(score)


def grade_task(state, task):
    if state is None:
        return MIN_OPEN_SCORE

    weighted_score = 0.0
    weight_total = 0.0

    def add_component(weight, condition, partial=0.0):
        nonlocal weighted_score, weight_total
        weight_total += weight
        if condition:
            weighted_score += weight
        else:
            weighted_score += weight * partial

    add_component(0.3, state.diagnosis_correct)

    if task.get("requires_clarification"):
        add_component(0.18, state.asked_clarification)

    markers = task.get("valid_fix_markers", [])
    if markers:
        unique_hits = len(set(state.matched_fix_markers))
        coverage = unique_hits / len(markers)
        add_component(0.37, coverage >= 0.99, partial=min(coverage, 1.0))

    resolved_ok = (
        state.resolved
        and state.premature_confirm_attempts == 0
        and state.fix_applied
        and state.fix_proposed
    )
    add_component(0.1, resolved_ok, partial=0.5 if state.fix_applied else 0.0)

    efficient = state.steps_taken <= max(3, task.get("max_steps", 8) - 2)
    add_component(0.05, efficient)

    if weight_total == 0:
        return MIN_OPEN_SCORE

    score = weighted_score / weight_total
    return _clamp_task_score(score)
