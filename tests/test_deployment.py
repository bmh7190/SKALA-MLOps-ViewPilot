import unittest

from serving_app.deployment import evaluate_deployment, naive_predictions
from tests.support import video


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        # 3일 합계 300 → 단순 예측 700, 실제 7일 조회수 1000
        self.rows = [video(0, day1_views=100, day2_views=100, day3_views=100, target_views_day7=1000)]

    def test_naive_uses_daily_views_without_target(self):
        inputs = [{"day1_views": 100, "day2_views": 200, "day3_views": 300},
                  {"day1_views": 0, "day2_views": 0, "day3_views": 0}]
        self.assertEqual(naive_predictions(inputs), [1400, 0])

    def test_initial_without_absolute_gate_still_must_beat_naive(self):
        for predictions, promoted in [([900], True), ([700], False), ([500], False)]:
            with self.subTest(predictions=predictions):
                result = evaluate_deployment(self.rows, predictions)
                self.assertEqual(result["promoted"], promoted)
                self.assertIsNone(result["current_rmsle"])

    def test_initial_absolute_limit_is_an_additional_inclusive_check(self):
        score = evaluate_deployment(self.rows, [900])["rmsle"]
        self.assertTrue(evaluate_deployment(self.rows, [900], absolute_gate=score)["promoted"])
        result = evaluate_deployment(self.rows, [900], absolute_gate=score / 2)
        self.assertFalse(result["promoted"])
        self.assertTrue(result["deployment_checks"]["beats_naive"])
        self.assertFalse(result["deployment_checks"]["within_absolute_gate"])

    def test_improving_current_is_insufficient_when_naive_is_better(self):
        result = evaluate_deployment(self.rows, [600], current_predictions=[400])
        self.assertTrue(result["deployment_checks"]["beats_current"])
        self.assertFalse(result["deployment_checks"]["beats_naive"])
        self.assertFalse(result["promoted"])

    def test_retraining_must_strictly_improve_both_comparators(self):
        for candidate, current, expected in [(900, 800, True), (800, 800, False), (800, 900, False), (700, 500, False)]:
            with self.subTest(candidate=candidate, current=current):
                result = evaluate_deployment(self.rows, [candidate], [current], absolute_gate=0)
                self.assertEqual(result["promoted"], expected)
                self.assertNotIn("within_absolute_gate", result["deployment_checks"])

    def test_invalid_prediction_cannot_pass_gate(self):
        for predictions in [[float("nan")], [float("inf")], [-1], []]:
            with self.subTest(predictions=predictions):
                with self.assertRaises(ValueError):
                    evaluate_deployment(self.rows, predictions)
