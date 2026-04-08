import unittest

from app.scoring import (
    MAX_OPEN_SCORE,
    MIN_OPEN_SCORE,
    average_open_scores,
    clamp_closed_unit_interval,
    clamp_open_unit_interval,
    sanitize_score_fields,
)


class ScoringTests(unittest.TestCase):
    def test_open_interval_clamp_blocks_boundary_values(self):
        self.assertEqual(clamp_open_unit_interval(0.0), MIN_OPEN_SCORE)
        self.assertEqual(clamp_open_unit_interval(1.0), MAX_OPEN_SCORE)
        self.assertEqual(clamp_open_unit_interval(0.999), MAX_OPEN_SCORE)
        self.assertEqual(clamp_open_unit_interval(0.001), MIN_OPEN_SCORE)

    def test_closed_interval_clamp_preserves_reward_boundaries(self):
        self.assertEqual(clamp_closed_unit_interval(-0.5), 0.0)
        self.assertEqual(clamp_closed_unit_interval(1.5), 1.0)
        self.assertEqual(clamp_closed_unit_interval(0.456), 0.46)

    def test_average_open_scores_stays_inside_interval(self):
        self.assertEqual(average_open_scores([]), MIN_OPEN_SCORE)
        self.assertEqual(average_open_scores([0.99, 0.99, 0.99]), MAX_OPEN_SCORE)
        self.assertEqual(average_open_scores([0.91, 0.92, 0.93]), 0.92)

    def test_sanitize_score_fields_clamps_nested_boundary_values(self):
        payload = {
            "score": 1.0,
            "nested": {"score": 0.0, "helpful_response_score": 1.0},
            "items": [{"score": 0.0}, {"value": 7}],
        }
        sanitized = sanitize_score_fields(payload)
        self.assertEqual(sanitized["score"], 0.99)
        self.assertEqual(sanitized["nested"]["score"], 0.01)
        self.assertEqual(sanitized["nested"]["helpful_response_score"], 0.99)
        self.assertEqual(sanitized["items"][0]["score"], 0.01)


if __name__ == "__main__":
    unittest.main()
