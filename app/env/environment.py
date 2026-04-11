import uuid
from typing import Dict, List, Optional, Tuple

from app.models.schemas import Action, Observation, Reward, State
from app.scoring import clamp_open_unit_interval


class ApiRepairEnv:
    def __init__(self):
        self.current_state: Optional[State] = None
        self.current_task: Optional[Dict] = None

    def reset(self) -> Observation:
        from app.env.tasks import TASKS

        return self.reset_with_task(TASKS[0])

    def reset_with_task(self, task: Dict) -> Observation:
        self.current_task = task
        self.current_state = State(
            request_id=str(uuid.uuid4()),
            task_id=task["id"],
            artifact=task["artifact"],
            conversation_history=[
                f"[{task['name']}]",
                task["artifact"],
            ],
            sentiment=0.0,
            steps_taken=0,
            max_steps=task.get("max_steps", 8),
        )
        return self._build_observation(self._initial_feedback(task))

    def state(self) -> Optional[State]:
        return self.current_state

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.current_state is None or self.current_task is None:
            raise ValueError("Environment must be reset before calling step().")

        state = self.current_state
        task = self.current_task
        reward = -0.03
        done = False
        info: Dict[str, object] = {"task_id": task["id"]}

        state.steps_taken += 1
        state.action_history.append(action.action_type)
        agent_line = f"agent {action.action_type}: {action.content.strip()}"
        state.conversation_history.append(agent_line)

        if action.action_type == "analyze":
            r, d = self._handle_analyze(action)
            reward += r
            done = done or d
        elif action.action_type == "ask":
            reward += self._handle_ask(action)
        elif action.action_type == "propose_fix":
            r, d = self._handle_propose_fix(action)
            reward += r
            done = done or d
        elif action.action_type == "apply_fix":
            r, d = self._handle_apply_fix(action)
            reward += r
            done = done or d
        elif action.action_type == "confirm_done":
            r, d = self._handle_confirm_done(action)
            reward += r
            done = done or d

        if state.steps_taken >= state.max_steps:
            if not state.resolved:
                reward -= 0.2
            done = True

        feedback = self._feedback_after_step(action)
        state.conversation_history.append(f"simulator: {feedback}")
        reward += 0.04 * state.sentiment

        observation = self._build_observation(feedback)
        info["state_summary"] = {
            "diagnosis_correct": state.diagnosis_correct,
            "fix_proposed": state.fix_proposed,
            "fix_applied": state.fix_applied,
            "resolved": state.resolved,
        }
        return observation, Reward(score=self._normalize_reward(reward)), done, info

    def _normalize_reward(self, reward: float) -> float:
        return clamp_open_unit_interval(reward)

    def _initial_feedback(self, task: Dict) -> str:
        tid = task["id"]
        if tid == "task_1":
            return "HTTP 400: validation error — request body missing required fields."
        if tid == "task_2":
            return "HTTP 405: method not allowed for this resource (expected JSON search)."
        return "HTTP 502: bad gateway — upstream error. Impact scope unclear."

    def _feedback_after_step(self, action: Action) -> str:
        state = self.current_state
        task = self.current_task
        assert state is not None and task is not None

        if action.action_type == "ask" and task.get("requires_clarification") and state.asked_clarification:
            return task.get("clarification_response", "Here is the missing context.")
        if action.action_type == "analyze" and state.diagnosis_correct:
            return "Diagnosis recorded in the incident log."
        if action.action_type == "propose_fix" and state.fix_proposed:
            return "Patch validated against schema checks."
        if action.action_type == "apply_fix" and state.fix_applied:
            return "CI pipeline accepted the change; staging deploy triggered."
        if action.action_type == "confirm_done" and state.resolved:
            return "Incident closed; on-call notified."
        return "Awaiting next action."

    def _handle_analyze(self, action: Action) -> Tuple[float, bool]:
        state = self.current_state
        task = self.current_task
        assert state is not None and task is not None

        state.diagnosis_attempts += 1
        if task.get("requires_clarification") and not state.asked_clarification:
            state.sentiment = max(-1.0, state.sentiment - 0.1)
            return -0.14, False

        if action.predicted_diagnosis == task["expected_diagnosis"]:
            if not state.diagnosis_correct:
                state.diagnosis_correct = True
                state.sentiment = min(1.0, state.sentiment + 0.12)
                return 0.28, False
            return -0.02, False

        state.diagnosis_correct = False
        state.sentiment = max(-1.0, state.sentiment - 0.12)
        return -0.2, False

    def _handle_ask(self, action: Action) -> float:
        state = self.current_state
        task = self.current_task
        assert state is not None and task is not None
        content = action.content.lower()

        if not task.get("requires_clarification"):
            state.sentiment = max(-1.0, state.sentiment - 0.04)
            return -0.06

        if state.asked_clarification:
            return -0.04

        if "?" in action.content or any(
            w in content for w in ("which", "what", "staging", "prod", "environment", "scope", "blast", "where")
        ):
            state.asked_clarification = True
            state.sentiment = min(1.0, state.sentiment + 0.1)
            return 0.17

        state.sentiment = max(-1.0, state.sentiment - 0.06)
        return -0.09

    def _handle_propose_fix(self, action: Action) -> Tuple[float, bool]:
        state = self.current_state
        task = self.current_task
        assert state is not None and task is not None
        content = action.content.lower()

        if not state.diagnosis_correct:
            state.sentiment = max(-1.0, state.sentiment - 0.08)
            return -0.16, False

        markers: List[str] = task.get("valid_fix_markers", [])
        merged = set(state.matched_fix_markers)
        for m in markers:
            if m.lower() in content:
                merged.add(m)
        state.matched_fix_markers = sorted(merged)

        coverage = len(state.matched_fix_markers) / len(markers) if markers else 1.0
        state.sentiment = min(1.0, state.sentiment + 0.08 * coverage)

        if coverage >= 1.0:
            state.fix_proposed = True
            return 0.18 + 0.12 * coverage, False

        return -0.04 + 0.14 * coverage, False

    def _handle_apply_fix(self, action: Action) -> Tuple[float, bool]:
        state = self.current_state
        task = self.current_task
        assert state is not None and task is not None

        if not state.fix_proposed:
            state.premature_confirm_attempts += 1
            state.sentiment = max(-1.0, state.sentiment - 0.1)
            return -0.17, False

        state.fix_applied = True
        state.sentiment = min(1.0, state.sentiment + 0.08)

        if not task.get("requires_confirm"):
            state.resolved = True
            return 0.22, True

        return 0.14, False

    def _handle_confirm_done(self, action: Action) -> Tuple[float, bool]:
        state = self.current_state
        task = self.current_task
        assert state is not None and task is not None

        if not task.get("requires_confirm"):
            state.premature_confirm_attempts += 1
            return -0.08, False

        ready = (
            state.diagnosis_correct
            and state.fix_applied
            and state.fix_proposed
            and (not task.get("requires_clarification") or state.asked_clarification)
        )
        if ready:
            state.resolved = True
            state.sentiment = min(1.0, state.sentiment + 0.1)
            return 0.24, True

        state.premature_confirm_attempts += 1
        state.sentiment = max(-1.0, state.sentiment - 0.1)
        return -0.14, False

    def _build_observation(self, feedback: str) -> Observation:
        state = self.current_state
        assert state is not None
        return Observation(
            artifact=state.artifact,
            latest_feedback=feedback,
            conversation_history=list(state.conversation_history),
            steps_remaining=max(state.max_steps - state.steps_taken, 0),
            sentiment=round(state.sentiment, 2),
            task_id=state.task_id,
            available_actions=[
                "analyze",
                "ask",
                "propose_fix",
                "apply_fix",
                "confirm_done",
            ],
        )
