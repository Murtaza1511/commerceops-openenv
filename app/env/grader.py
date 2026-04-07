def grade_task(state, task):
    if state is None:
        return 0.0

    weighted_score = 0.0
    weight_total = 0.0

    def add_component(weight, condition, partial=0.0):
        nonlocal weighted_score, weight_total
        weight_total += weight
        if condition:
            weighted_score += weight
        else:
            weighted_score += weight * partial

    add_component(0.35, state.issue_type == task.get("expected_issue"))

    if task.get("requires_clarification"):
        add_component(0.2, state.asked_clarification)

    valid_solutions = task.get("valid_solutions", [])
    if valid_solutions:
        coverage = len(set(state.matched_solution_keywords)) / len(valid_solutions)
        helpful_partial = max(state.helpful_response_score, coverage)
        add_component(0.3, helpful_partial >= 0.99, partial=min(helpful_partial, 1.0))

    if task.get("requires_resolution"):
        resolved_cleanly = state.resolved and state.premature_resolution_attempts == 0
        add_component(0.1, resolved_cleanly, partial=0.5 if state.resolved else 0.0)

    if task.get("requires_escalation"):
        escalated_cleanly = state.resolved and state.premature_resolution_attempts == 0
        add_component(0.1, escalated_cleanly, partial=0.5 if state.resolved else 0.0)

    efficient = state.steps_taken <= max(2, task.get("max_steps", 5) - 1)
    add_component(0.05, efficient)

    if weight_total == 0:
        return 0.0

    score = weighted_score / weight_total
    return round(min(score, 1.0), 2)
