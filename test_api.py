import unittest

from app.api.routes import env, get_tasks, grader, grader_get, index, reset, run_baseline, step
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
        self.assertGreater(result["score"], 0.0)
        self.assertLess(result["score"], 1.0)

    def test_grader_without_task_id_uses_current_state_task(self):
        reset("task_2")
        result = grader()
        self.assertEqual(result["task_id"], "task_2")
        self.assertGreater(result["score"], 0.0)
        self.assertLess(result["score"], 1.0)

    def test_grader_accepts_common_external_task_id_keys(self):
        reset({"taskId": "task_3"})
        result = grader(payload={"task": "task_3"})
        self.assertEqual(result["task_id"], "task_3")
        self.assertGreater(result["score"], 0.0)
        self.assertLess(result["score"], 1.0)

    def test_grader_before_reset_still_returns_in_range_score(self):
        env.current_state = None
        env.current_task = None

        result = grader(payload={"task_id": "task_1"})
        self.assertEqual(result["task_id"], "task_1")
        self.assertGreater(result["score"], 0.0)
        self.assertLess(result["score"], 1.0)

    def test_get_grader_path_is_supported_and_in_range(self):
        reset("task_1")
        result = grader_get()
        self.assertEqual(result["task_id"], "task_1")
        self.assertGreater(result["score"], 0.0)
        self.assertLess(result["score"], 1.0)

    def test_reset_with_invalid_task_falls_back_to_default(self):
        observation = reset({"task_id": "unknown-task"})
        self.assertEqual(observation.task_id, "task_1")

    def test_step_before_reset_bootstraps_state(self):
        env.current_state = None
        env.current_task = None
        response = step(
            Action(
                action_type="classify",
                content="Account access issue",
                predicted_issue="account_access",
            )
        )
        self.assertIn("reward", response)
        self.assertGreater(response["reward"].score, 0.0)
        self.assertLess(response["reward"].score, 1.0)

    def test_baseline_handler_returns_structured_summary(self):
        summary = run_baseline()

        self.assertIn("results", summary)
        self.assertIn("average_score", summary)
        self.assertEqual(len(summary["results"]), 3)
        self.assertGreater(summary["average_score"], 0.0)
        self.assertLess(summary["average_score"], 1.0)
        for item in summary["results"]:
            self.assertGreater(item["score"], 0.0)
            self.assertLess(item["score"], 1.0)
        self.assertIsNotNone(env.state())


if __name__ == "__main__":
    unittest.main()
