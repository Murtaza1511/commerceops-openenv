import unittest

from app.api.routes import env, get_tasks, grader, index, reset, run_baseline, step
from app.models.schemas import Action


class ApiHandlerTests(unittest.TestCase):
    def setUp(self):
        reset("task_1")

    def test_tasks_handler_returns_task_list_and_action_schema(self):
        payload = get_tasks()

        self.assertEqual(len(payload["tasks"]), 3)
        self.assertIn("action_schema", payload)
        self.assertIn("properties", payload["action_schema"])

    def test_health_handler_reports_service_status(self):
        from app.api.routes import health

        payload = health()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], "commerceops-openenv")

    def test_root_handler_reports_service_status(self):
        payload = index()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["health"], "/health")

    def test_reset_step_and_grader_workflow(self):
        observation = reset("task_1")
        self.assertEqual(observation.task_id, "task_1")

        response = step(
            Action(
                action_type="classify",
                content="This is an account access issue.",
                predicted_issue="account_access",
            )
        )
        self.assertTrue(response["done"])

        result = grader("task_1")
        self.assertGreaterEqual(result["score"], 0.9)

    def test_baseline_handler_returns_structured_summary(self):
        summary = run_baseline()

        self.assertIn("results", summary)
        self.assertIn("average_score", summary)
        self.assertEqual(len(summary["results"]), 3)
        self.assertIsNotNone(env.state())


if __name__ == "__main__":
    unittest.main()
