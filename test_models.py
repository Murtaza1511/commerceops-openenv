import unittest

from app.models.schemas import Action, Observation, State


class SchemaTests(unittest.TestCase):
    def test_action_model_accepts_diagnosis_label(self):
        action = Action(
            action_type="analyze",
            content="Missing qty in JSON body.",
            predicted_diagnosis="missing_required_field",
        )
        self.assertEqual(action.predicted_diagnosis, "missing_required_field")

    def test_observation_exposes_available_actions(self):
        observation = Observation(
            artifact="GET / HTTP/1.1",
            latest_feedback="405 Method Not Allowed",
            conversation_history=["[scenario]"],
            steps_remaining=3,
            sentiment=0.0,
            task_id="task_1",
        )
        self.assertIn("analyze", observation.available_actions)
        self.assertIn("apply_fix", observation.available_actions)

    def test_state_defaults_are_isolated_per_instance(self):
        first = State(request_id="1", artifact="a")
        second = State(request_id="2", artifact="b")

        first.matched_fix_markers.append("x")

        self.assertEqual(second.matched_fix_markers, [])


if __name__ == "__main__":
    unittest.main()
