import uuid
from typing import Dict, List, Optional, Tuple

from app.models.schemas import Action, Observation, Reward, State
from app.scoring import clamp_open_unit_interval


class CustomerSupportEnv:
    def __init__(self):
        self.current_state: Optional[State] = None
        self.current_task: Optional[Dict] = None
        self.knowledge_base: Dict[str, List[str]] = {
            "payment_issue": [
                "check bank hold",
                "avoid repeated retries",
                "wait 24 hours before retrying",
            ],
            "product_bug": [
                "restart app",
                "clear cache",
                "update app",
            ],
            "account_access": [
                "recover 2fa",
                "verify account ownership",
                "reset password",
            ],
            "fraud_risk": [
                "freeze card",
                "review recent orders",
                "change account password",
            ],
        }

    def reset(self) -> Observation:
        default_task = {
            "id": "adhoc_task",
            "query": "A customer says their card was charged twice after a checkout retry.",
            "expected_issue": "payment_issue",
            "valid_solutions": self.knowledge_base["payment_issue"],
            "requires_resolution": False,
            "requires_escalation": False,
            "requires_clarification": False,
            "max_steps": 5,
        }
        return self.reset_with_task(default_task)

    def reset_with_task(self, task: Dict) -> Observation:
        self.current_task = task
        self.current_state = State(
            ticket_id=str(uuid.uuid4()),
            task_id=task["id"],
            customer_query=task["query"],
            conversation_history=[task["query"]],
            sentiment=0.0,
            steps_taken=0,
            max_steps=task.get("max_steps", 5),
        )
        return self._build_observation(task["query"])

    def state(self) -> Optional[State]:
        return self.current_state

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.current_state is None or self.current_task is None:
            raise ValueError("Environment must be reset before calling step().")

        state = self.current_state
        task = self.current_task
        reward = -0.04
        done = False
        info: Dict[str, object] = {"task_id": task["id"]}

        state.steps_taken += 1
        state.action_history.append(action.action_type)
        agent_message = action.content.strip()
        state.conversation_history.append(f"agent: {agent_message}")

        if action.action_type == "classify":
            classify_reward, classify_done = self._handle_classify(action)
            reward += classify_reward
            done = done or classify_done
        elif action.action_type == "ask":
            reward += self._handle_ask(action)
        elif action.action_type == "respond":
            respond_reward, respond_done = self._handle_respond(action)
            reward += respond_reward
            done = done or respond_done
        elif action.action_type == "resolve":
            resolution_reward, done = self._handle_resolve()
            reward += resolution_reward
        elif action.action_type == "escalate":
            escalation_reward, done = self._handle_escalate()
            reward += escalation_reward
            info["escalated"] = state.resolved

        if state.steps_taken >= state.max_steps:
            if not state.resolved:
                reward -= 0.25
            done = True

        customer_reply = self._generate_customer_reply(action)
        state.conversation_history.append(f"customer: {customer_reply}")
        reward += 0.05 * state.sentiment

        observation = self._build_observation(customer_reply)
        info["state_summary"] = {
            "classification_correct": state.classification_correct,
            "responded_helpfully": state.responded_helpfully,
            "resolved": state.resolved,
        }
        normalized_reward = self._normalize_reward(reward)
        return observation, Reward(score=normalized_reward), done, info

    def _normalize_reward(self, reward: float) -> float:
        return clamp_open_unit_interval(reward)

    def _handle_classify(self, action: Action) -> Tuple[float, bool]:
        state = self.current_state
        task = self.current_task
        state.classification_attempts += 1

        if action.predicted_issue == task["expected_issue"]:
            if not state.classification_correct:
                state.classification_correct = True
                state.issue_type = action.predicted_issue
                state.sentiment = min(1.0, state.sentiment + 0.1)
                done = (
                    not task.get("valid_solutions")
                    and not task.get("requires_resolution")
                    and not task.get("requires_escalation")
                )
                return 0.3, done
            return -0.02, False

        state.issue_type = action.predicted_issue
        state.classification_correct = False
        state.sentiment = max(-1.0, state.sentiment - 0.15)
        return -0.25, False

    def _handle_ask(self, action: Action) -> float:
        state = self.current_state
        task = self.current_task
        content = action.content.lower()

        if task.get("requires_clarification"):
            if not state.asked_clarification and any(
                token in content for token in ["detail", "details", "when", "error", "what happens"]
            ):
                state.asked_clarification = True
                state.sentiment = min(1.0, state.sentiment + 0.1)
                return 0.18
            state.sentiment = max(-1.0, state.sentiment - 0.05)
            return -0.08

        state.sentiment = max(-1.0, state.sentiment - 0.03)
        return -0.05

    def _handle_respond(self, action: Action) -> Tuple[float, bool]:
        state = self.current_state
        task = self.current_task
        content = action.content.lower()

        if not state.classification_correct:
            state.sentiment = max(-1.0, state.sentiment - 0.1)
            return -0.18, False

        expected_solutions = task.get("valid_solutions", [])
        matched = [solution for solution in expected_solutions if solution in content]
        state.knowledge_used.append(action.content)

        if matched:
            merged = set(state.matched_solution_keywords)
            merged.update(matched)
            state.matched_solution_keywords = sorted(merged)
            coverage = len(state.matched_solution_keywords) / len(expected_solutions)
            state.helpful_response_score = max(
                state.helpful_response_score,
                clamp_open_unit_interval(coverage),
            )
            state.responded_helpfully = coverage >= (2 / 3) if expected_solutions else True
            state.sentiment = min(1.0, state.sentiment + 0.2)
            done = (
                state.responded_helpfully
                and not task.get("requires_resolution")
                and not task.get("requires_escalation")
            )
            return 0.12 + (0.28 * coverage), done

        fallback_match = any(
            phrase in content
            for phrase in [
                "bank hold",
                "retry",
                "24 hours",
                "freeze card",
                "recent orders",
                "password",
            ]
        )
        if fallback_match:
            state.helpful_response_score = max(state.helpful_response_score, 0.34)
            state.sentiment = min(1.0, state.sentiment + 0.05)
            return 0.08, False

        state.sentiment = max(-1.0, state.sentiment - 0.12)
        return -0.15, False

    def _handle_resolve(self) -> Tuple[float, bool]:
        state = self.current_state
        task = self.current_task

        resolution_ready = state.classification_correct and (
            state.responded_helpfully or not task.get("valid_solutions")
        )
        if task.get("requires_clarification"):
            resolution_ready = resolution_ready and state.asked_clarification

        if resolution_ready:
            state.resolved = True
            state.sentiment = min(1.0, state.sentiment + 0.15)
            return 0.35, True

        state.premature_resolution_attempts += 1
        state.sentiment = max(-1.0, state.sentiment - 0.15)
        return -0.22, False

    def _handle_escalate(self) -> Tuple[float, bool]:
        state = self.current_state
        task = self.current_task

        escalation_ready = state.classification_correct and (
            state.responded_helpfully or not task.get("valid_solutions")
        )
        if task.get("requires_clarification"):
            escalation_ready = escalation_ready and state.asked_clarification

        if task.get("requires_escalation") and escalation_ready:
            state.resolved = True
            state.sentiment = min(1.0, state.sentiment + 0.15)
            return 0.32, True

        if task.get("requires_escalation"):
            state.premature_resolution_attempts += 1
            state.sentiment = max(-1.0, state.sentiment - 0.15)
            return -0.24, False

        state.sentiment = max(-1.0, state.sentiment - 0.1)
        return -0.18, True

    def _generate_customer_reply(self, action: Action) -> str:
        state = self.current_state
        task = self.current_task

        if action.action_type == "ask" and task.get("requires_clarification") and state.asked_clarification:
            return task["clarification_response"]

        if action.action_type == "resolve" and state.resolved:
            return task.get("resolution_confirmation", "Thanks, the issue looks fixed now.")

        if action.action_type == "escalate" and state.resolved:
            return task.get("escalation_confirmation", "Thanks, please escalate this case to the specialist team.")

        if action.action_type == "respond" and state.responded_helpfully:
            return "That sounds useful. I will try those steps now."

        if action.action_type == "classify" and state.classification_correct:
            return "Yes, that sounds like the problem I am facing."

        if state.sentiment <= -0.1:
            return "That did not help. Can you be more specific?"

        return "I am still waiting for a concrete next step."

    def _build_observation(self, latest_message: str) -> Observation:
        state = self.current_state
        return Observation(
            latest_customer_message=latest_message,
            conversation_history=state.conversation_history,
            steps_remaining=max(state.max_steps - state.steps_taken, 0),
            sentiment=round(state.sentiment, 2),
            task_id=state.task_id,
        )
