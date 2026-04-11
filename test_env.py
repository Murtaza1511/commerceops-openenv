import unittest

from app.baseline_runner import choose_action
from app.env.environment import ApiRepairEnv
from app.env.grader import _clamp_task_score, grade_task
from app.env.tasks import TASKS
from app.models.schemas import Action


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = ApiRepairEnv()

    def test_easy_task_completes_with_baseline_policy(self):
        task = next(t for t in TASKS if t["id"] == "task_1")
        observation = self.env.reset_with_task(task)
        done = False
        step_count = 0
        while not done and step_count < task.get("max_steps", 8):
            payload = choose_action(task, observation, step_count)
            observation, _, done, _ = self.env.step(Action.model_validate(payload))
            step_count += 1

        self.assertTrue(done)
        self.assertTrue(self.env.state().resolved)
        score = grade_task(self.env.state(), task)
        self.assertGreaterEqual(score, 0.85)

    def test_hard_task_rejects_confirm_before_clarification(self):
        task = next(t for t in TASKS if t["id"] == "task_3")
        self.env.reset_with_task(task)

        self.env.step(
            Action(
                action_type="confirm_done",
                content="Closing without context.",
            )
        )
        self.assertFalse(self.env.state().resolved)

    def test_grader_rewards_complete_medium_workflow(self):
        task = next(t for t in TASKS if t["id"] == "task_2")
        observation = self.env.reset_with_task(task)
        done = False
        step_count = 0
        while not done and step_count < task.get("max_steps", 8):
            payload = choose_action(task, observation, step_count)
            observation, _, done, _ = self.env.step(Action.model_validate(payload))
            step_count += 1

        score = grade_task(self.env.state(), task)
        self.assertGreaterEqual(score, 0.85)

    def test_step_reward_is_normalized_to_strict_open_interval(self):
        task = next(t for t in TASKS if t["id"] == "task_3")
        self.env.reset_with_task(task)

        _, reward, _, _ = self.env.step(
            Action(
                action_type="confirm_done",
                content="Too early.",
            )
        )

        self.assertGreater(reward.score, 0.0)
        self.assertLess(reward.score, 1.0)

    def test_all_task_scores_stay_strictly_inside_zero_one(self):
        for task in TASKS:
            observation = self.env.reset_with_task(task)
            done = False
            step_count = 0

            while not done and step_count < task.get("max_steps", 8):
                action = Action.model_validate(choose_action(task, observation, step_count))
                observation, _, done, _ = self.env.step(action)
                step_count += 1

            score = grade_task(self.env.state(), task)
            self.assertGreater(score, 0.0)
            self.assertLess(score, 1.0)

    def test_task_score_clamp_stays_strictly_inside_open_interval(self):
        self.assertEqual(_clamp_task_score(0.0), 0.01)
        self.assertEqual(_clamp_task_score(1.0), 0.99)
        self.assertEqual(_clamp_task_score(0.994), 0.99)
        self.assertEqual(_clamp_task_score(0.004), 0.01)


if __name__ == "__main__":
    unittest.main()
