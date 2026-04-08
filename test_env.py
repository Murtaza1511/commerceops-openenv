import unittest

from app.env.environment import CustomerSupportEnv
from app.env.grader import grade_task
from app.env.tasks import TASKS
from app.models.schemas import Action


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = CustomerSupportEnv()

    def test_easy_task_finishes_after_correct_classification(self):
        task = next(task for task in TASKS if task["id"] == "task_1")
        self.env.reset_with_task(task)

        observation, reward, done, _ = self.env.step(
            Action(
                action_type="classify",
                content="This is an account access issue caused by a 2FA device problem.",
                predicted_issue="account_access",
            )
        )

        self.assertTrue(done)
        self.assertGreater(reward.score, 0)
        self.assertEqual(observation.task_id, "task_1")

    def test_hard_task_requires_clarification_before_resolution(self):
        task = next(task for task in TASKS if task["id"] == "task_3")
        self.env.reset_with_task(task)

        _, _, _, _ = self.env.step(
            Action(
                action_type="classify",
                content="This seems like a fraud risk.",
                predicted_issue="fraud_risk",
            )
        )
        _, reward, done, _ = self.env.step(
            Action(
                action_type="escalate",
                content="Sending this to the fraud team now.",
            )
        )

        self.assertFalse(done)
        self.assertEqual(reward.score, 0.0)

    def test_grader_rewards_complete_medium_workflow(self):
        task = next(task for task in TASKS if task["id"] == "task_2")
        self.env.reset_with_task(task)

        self.env.step(
            Action(
                action_type="classify",
                content="This is a payment failure.",
                predicted_issue="payment_issue",
            )
        )
        self.env.step(
            Action(
                action_type="respond",
                content="Please check bank hold, avoid repeated retries, and wait 24 hours before retrying.",
            )
        )

        score = grade_task(self.env.state(), task)
        self.assertGreaterEqual(score, 0.99)

    def test_step_reward_is_normalized_to_zero_one_range(self):
        task = next(task for task in TASKS if task["id"] == "task_3")
        self.env.reset_with_task(task)

        _, reward, _, _ = self.env.step(
            Action(
                action_type="escalate",
                content="Escalating immediately without clarification.",
            )
        )

        self.assertGreaterEqual(reward.score, 0.0)
        self.assertLessEqual(reward.score, 1.0)

    def test_all_task_scores_stay_strictly_inside_zero_one(self):
        from app.baseline_runner import choose_action

        for task in TASKS:
            observation = self.env.reset_with_task(task)
            done = False
            step_count = 0

            while not done and step_count < task.get("max_steps", 5):
                action = Action.model_validate(choose_action(task, observation, step_count))
                observation, _, done, _ = self.env.step(action)
                step_count += 1

            score = grade_task(self.env.state(), task)
            self.assertGreater(score, 0.0)
            self.assertLess(score, 1.0)


if __name__ == "__main__":
    unittest.main()
