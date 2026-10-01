import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tests.support import video
from data import observations
from serving_app.monitoring.drift_detector import compute_rmsle, drift_threshold, evaluate_block
from serving_app.monitoring.retrain_trigger import check_and_trigger


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.patch = patch.object(observations, "OBSERVATION_DB", Path(self.directory.name) / "observations.db")
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def add(self, start, count, version="1", mode="simulation", bad=True):
        rows = [video(i) for i in range(start, start + count)]
        predictions = [8400 if bad else 12000] * count
        return observations.save_predictions(rows, predictions, version, mode)

    def test_metric_and_threshold(self):
        self.assertAlmostEqual(compute_rmsle([0, 99], [0, 9]), math.log(10) / math.sqrt(2))
        self.assertAlmostEqual(drift_threshold(0.2), 0.3)
        self.assertAlmostEqual(drift_threshold(1.0), 1.3)
        self.assertEqual(evaluate_block([1] * 30, [1] * 30, 0.2, 2)["consecutive_exceeds"], 0)
        with self.assertRaises(ValueError):
            compute_rmsle([], [])

    def test_nonoverlapping_blocks_two_exceeds_reset_and_replay(self):
        self.add(0, 29)
        check = observations.evaluate_pending("1", "simulation", 0.1)
        self.assertEqual(check["pending_count"], 29)
        self.assertFalse(check["new_blocks"])
        self.add(29, 1)
        first = observations.evaluate_pending("1", "simulation", 0.1)
        self.assertEqual(first["status"], "warning")
        self.add(30, 30)
        second = observations.evaluate_pending("1", "simulation", 0.1)
        self.assertEqual(second["status"], "retrain_review")
        self.assertTrue(set(first["video_ids"]).isdisjoint(second["video_ids"]))
        self.add(30, 30)
        self.assertFalse(observations.evaluate_pending("1", "simulation", 0.1)["new_blocks"])
        self.add(60, 30, bad=False)
        reset = observations.evaluate_pending("1", "simulation", 0.1)
        self.assertEqual(reset["consecutive_exceeds"], 0)

    def test_versions_and_simulation_are_isolated(self):
        self.add(0, 30)
        observations.evaluate_pending("1", "simulation", 0.1)
        self.add(30, 29, version="2")
        self.assertEqual(observations.evaluate_pending("2", "simulation", 0.1)["status"], "collecting")
        self.add(0, 30, mode="live")
        self.assertEqual(observations.evaluate_pending("1", "live", 0.1)["pending_count"], 0)

    def test_missing_baseline_keeps_pending_and_later_processes_all(self):
        self.add(0, 61)
        self.assertEqual(observations.evaluate_pending("1", "simulation", None)["pending_count"], 61)
        check = observations.evaluate_pending("1", "simulation", 0.1)
        self.assertEqual(len(check["new_blocks"]), 2)
        self.assertEqual(check["pending_count"], 1)
        with self.assertRaises(ValueError):
            observations.evaluate_pending("1", "simulation", 0.2)

    def test_original_forecast_and_actual_are_immutable(self):
        original = self.add(0, 1, mode="live")[0]
        repeated = observations.save_predictions([video()], [9900], "1", "live")[0]
        self.assertEqual(original, repeated)
        observations.save_actual(original["prediction_id"], 12000)
        observations.save_actual(original["prediction_id"], 12000)
        self.assertEqual(len(observations.labeled_rows("live")), 1)
        with self.assertRaises(ValueError):
            observations.save_actual(original["prediction_id"], 15000)
        with self.assertRaises(ValueError):
            observations.save_predictions([video(day1_views=2000)], [9900], "1", "live")
        with self.assertRaises(LookupError):
            observations.save_actual("missing", 12000)

    def test_early_actual_rejected(self):
        row = video(published_at=datetime.now(timezone.utc).isoformat())
        result = observations.save_predictions([row], [10000], "1", "live")[0]
        with self.assertRaises(ValueError):
            observations.save_actual(result["prediction_id"], 12000)

    def test_old_version_does_not_trigger_new_model_training(self):
        self.add(0, 60, version="old")
        with patch.dict("os.environ", {"BASELINE_RMSLE": "0.1", "BASELINE_MODEL_VERSION": "old", "AUTO_RETRAIN": "true"}), \
             patch("serving_app.monitoring.retrain_trigger.check_configured_baseline"), \
             patch("serving_app.model_loader.get_model") as get_model:
            get_model.return_value.version = "new"
            check = check_and_trigger("old", "simulation")
        self.assertEqual(check["retraining"]["status"], "skipped_old_version")
