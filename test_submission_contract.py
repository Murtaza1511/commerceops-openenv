import unittest
from typing import Any

from fastapi.testclient import TestClient

from app.main import app


def _collect_score_values(payload: Any) -> list[float]:
    values: list[float] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "score" and isinstance(value, (int, float)):
                values.append(float(value))
            values.extend(_collect_score_values(value))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_collect_score_values(item))
    return values


class SubmissionContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_common_validator_calls_keep_all_score_fields_inside_open_interval(self):
        responses = []
        responses.append(self.client.get("/tasks"))
        responses.append(self.client.get("/grader"))
        responses.append(self.client.post("/grader", json={}))
        responses.append(self.client.post("/reset", json={"task_id": "task_1"}))
        responses.append(
            self.client.post(
                "/step",
                json={
                    "action_type": "analyze",
                    "content": "Request body is missing a required JSON field.",
                    "predicted_diagnosis": "missing_required_field",
                },
            )
        )
        responses.append(self.client.post("/grader", json={"task_id": "task_1"}))
        responses.append(self.client.get("/baseline"))
        responses.append(self.client.post("/reset", json={"task_id": "invalid-task"}))
        responses.append(self.client.get("/grader"))

        score_values: list[float] = []
        for response in responses:
            self.assertEqual(response.status_code, 200)
            score_values.extend(_collect_score_values(response.json()))

        self.assertGreater(len(score_values), 0)
        for value in score_values:
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1.0)


if __name__ == "__main__":
    unittest.main()
