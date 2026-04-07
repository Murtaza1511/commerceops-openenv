import unittest

from app.models.schemas import Action, Observation, State


class SchemaTests(unittest.TestCase):
    def test_action_model_accepts_supported_issue_label(self):
        action = Action(
            action_type="classify",
            content="This looks like an account access issue.",
            predicted_issue="account_access",
        )
        self.assertEqual(action.predicted_issue, "account_access")

    def test_observation_exposes_available_actions(self):
        observation = Observation(
            latest_customer_message="Help me reset my password.",
            conversation_history=["customer: Help me reset my password."],
            steps_remaining=3,
            sentiment=0.0,
            task_id="task_1",
        )
        self.assertIn("classify", observation.available_actions)
        self.assertIn("resolve", observation.available_actions)

    def test_state_defaults_are_isolated_per_instance(self):
        first = State(ticket_id="1", customer_query="Issue one")
        second = State(ticket_id="2", customer_query="Issue two")

        first.knowledge_used.append("shared?")

        self.assertEqual(second.knowledge_used, [])


if __name__ == "__main__":
    unittest.main()
