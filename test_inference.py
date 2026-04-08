import io
import json
import unittest
from contextlib import redirect_stdout

from inference import _log_end, _log_results


class InferenceLoggingTests(unittest.TestCase):
    def test_log_end_includes_task_score_in_open_interval(self):
        buffer = io.StringIO()
        result_payload = {
            "task_id": "task_1",
            "difficulty": "easy",
            "steps": 2,
            "score": 1.0,
            "rewards": [0.0, 1.0],
        }

        with redirect_stdout(buffer):
            _log_end(result_payload)

        line = buffer.getvalue().strip()
        self.assertTrue(line.startswith("[END] "))
        payload = json.loads(line[len("[END] ") :])

        self.assertIn("task_score", payload)
        self.assertIn("score", payload)
        self.assertGreater(payload["task_score"], 0.0)
        self.assertLess(payload["task_score"], 1.0)
        self.assertGreater(payload["score"], 0.0)
        self.assertLess(payload["score"], 1.0)

    def test_log_results_emits_normalized_per_task_scores(self):
        buffer = io.StringIO()
        results = [
            {"task_id": "task_1", "difficulty": "easy", "score": 0.0},
            {"task_id": "task_2", "difficulty": "medium", "score": 1.0},
            {"task_id": "task_3", "difficulty": "hard", "score": 0.57},
        ]

        with redirect_stdout(buffer):
            _log_results(results)

        line = buffer.getvalue().strip()
        self.assertTrue(line.startswith("[RESULTS] "))
        payload = json.loads(line[len("[RESULTS] ") :])

        self.assertIn("task_scores", payload)
        self.assertEqual(len(payload["task_scores"]), 3)
        for value in payload["task_scores"]:
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1.0)

        self.assertIn("per_task_scores", payload)
        self.assertEqual(len(payload["per_task_scores"]), 3)
        for item in payload["per_task_scores"]:
            self.assertIn("task_score", item)
            self.assertIn("score", item)
            self.assertGreater(item["task_score"], 0.0)
            self.assertLess(item["task_score"], 1.0)
            self.assertGreater(item["score"], 0.0)
            self.assertLess(item["score"], 1.0)

        self.assertIn("average_task_score", payload)
        self.assertGreater(payload["average_task_score"], 0.0)
        self.assertLess(payload["average_task_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
