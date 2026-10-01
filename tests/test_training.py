import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.support import video
from data import baselines, observations
from serving_app.config import baseline_rmsle, deployment_gate
from serving_app.train_and_register import prepare_data, train_and_register
from serving_app.monitoring.retrain_trigger import check_and_trigger


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for module, attribute, path in [(observations, "OBSERVATION_DB", "observations.db"),
                                         (baselines, "BASELINE_DIR", "baselines")]:
            p = patch.object(module, attribute, Path(self.directory.name) / path)
            p.start()
            self.addCleanup(p.stop)

    def test_explicit_splits_use_all_training_rows(self):
        training, validation = prepare_data([video(i) for i in range(40)], [video(i) for i in range(50, 60)])
        self.assertEqual((len(training), len(validation)), (40, 10))

    def test_retraining_rejects_previously_used_videos(self):
        current = SimpleNamespace(metadata={"used_video_ids": ["video_2"], "last_published_at": video(4)["published_at"]})
        with self.assertRaisesRegex(ValueError, "새 영상"):
            prepare_data([video(i) for i in range(40)], [video(i) for i in range(50, 60)], current)

    def test_test_videos_cannot_be_reused_for_training(self):
        observations.save_predictions([video(2)], [10000], "1", "simulation")
        with self.assertRaisesRegex(ValueError, "운영 테스트"):
            prepare_data([video(i) for i in range(40)], [video(i) for i in range(50, 60)])

    def test_initial_selection_needs_no_baseline(self):
        with patch.dict("os.environ", {"BASELINE_RMSLE": "", "DEPLOY_RMSLE_GATE": ""}):
            self.assertIsNone(deployment_gate())

    def test_baseline_does_not_transfer_to_new_model_version(self):
        with patch.dict("os.environ", {"BASELINE_RMSLE": "0.2", "BASELINE_MODEL_VERSION": "1", "DEPLOY_RMSLE_GATE": ""}):
            self.assertIsNone(deployment_gate("1"))
            self.assertIsNone(baseline_rmsle("2"))
            self.assertIsNone(deployment_gate("2"))

    def test_stale_retrain_trigger_is_rejected_inside_training_lock(self):
        with patch("serving_app.train_and_register.PRODUCTION_FILE") as pointer, \
             patch("serving_app.model_loader.get_model", return_value=SimpleNamespace(version="2")):
            pointer.exists.return_value = True
            with self.assertRaisesRegex(ValueError, "모델 버전"):
                train_and_register(mode="fine_tune", expected_version="1")

    def trigger(self, retraining_data):
        rows = [video(i) for i in range(60)]
        observations.save_predictions(rows, [8400] * 60, "1", "simulation")
        with patch.dict("os.environ", {"BASELINE_RMSLE": "0.1", "BASELINE_MODEL_VERSION": "1", "AUTO_RETRAIN": "true"}), \
             patch("serving_app.monitoring.retrain_trigger.check_configured_baseline"), \
             patch("serving_app.model_loader.get_model", return_value=SimpleNamespace(version="1")), \
             patch("serving_app.monitoring.retrain_trigger.retraining_data", **retraining_data):
            return check_and_trigger("1", "simulation")

    def test_drift_without_separate_data_defers_training(self):
        check = self.trigger({"side_effect": FileNotFoundError("별도 재학습 데이터 필요")})
        self.assertEqual(check["status"], "retrain_review")
        self.assertEqual(check["retraining"]["status"], "deferred")

    def test_drift_with_separate_data_reports_promotion_and_rejection(self):
        training, validation = [video(i) for i in range(100, 140)], [video(i) for i in range(150, 160)]
        with patch("serving_app.train_and_register.fine_tune", return_value={"promoted": True, "version": "2"}) as fit:
            check = self.trigger({"return_value": (training, validation)})
        fit.assert_called_once_with(training, validation, expected_version="1")
        self.assertTrue(check["retraining"]["promoted"])
        observations.save_predictions([video(i) for i in range(60, 90)], [8400] * 30, "1", "simulation")
        with patch("serving_app.train_and_register.fine_tune", return_value={"promoted": False}), \
             patch.dict("os.environ", {"BASELINE_RMSLE": "0.1", "BASELINE_MODEL_VERSION": "1", "AUTO_RETRAIN": "true"}), \
             patch("serving_app.monitoring.retrain_trigger.check_configured_baseline"), \
             patch("serving_app.model_loader.get_model", return_value=SimpleNamespace(version="1")), \
             patch("serving_app.monitoring.retrain_trigger.retraining_data", return_value=(training, validation)):
            check = check_and_trigger("1", "simulation")
        self.assertFalse(check["retraining"]["promoted"])

    def test_labeled_observations_can_train_but_not_validate(self):
        training = [video(i) for i in range(40)]
        validation = [video(i) for i in range(50, 60)]
        current = SimpleNamespace(metadata={"used_video_ids": [], "last_published_at": "2019-01-01T00:00:00+00:00"})
        observations.save_predictions(training, [10000] * 40, "1", "simulation")
        self.assertEqual(len(prepare_data(training, validation, current)[0]), 40)
        changed = [{**row, "target_views_day7": 15000} for row in training]
        with self.assertRaisesRegex(ValueError, "실제값"):
            prepare_data(changed, validation, current)
        observations.save_predictions(validation, [10000] * 10, "1", "simulation")
        with self.assertRaisesRegex(ValueError, "검증"):
            prepare_data(training, validation, current)

    def test_drift_threshold_is_not_a_retraining_deployment_gate(self):
        with patch.dict("os.environ", {"BASELINE_RMSLE": "0.1", "BASELINE_MODEL_VERSION": "1", "DEPLOY_RMSLE_GATE": "0"}):
            self.assertEqual(deployment_gate(), 0)
            self.assertIsNone(deployment_gate("1"))

    def test_scenario_retrains_only_on_two_detected_blocks(self):
        validation = [video(i) for i in range(70, 130)]
        observations.save_predictions([video(i) for i in range(60)], [8400] * 60, "1", "simulation")
        with patch.dict("os.environ", {"BASELINE_RMSLE": "0.1", "BASELINE_MODEL_VERSION": "1", "AUTO_RETRAIN": "true"}), \
             patch("serving_app.monitoring.retrain_trigger.check_configured_baseline"), \
             patch("serving_app.model_loader.get_model", return_value=SimpleNamespace(version="1")), \
             patch("serving_app.train_and_register.fine_tune", return_value={"promoted": True, "version": "2"}) as fit:
            check = check_and_trigger("1", "simulation", validation_rows=validation)
        training = fit.call_args.args[0]
        self.assertEqual([r["video_id"] for r in training], [f"video_{i}" for i in range(60)])
        fit.assert_called_once_with(training, validation, expected_version="1", retrospective=True)
        self.assertTrue(check["retraining"]["baseline_required"])
        self.assertTrue({r["video_id"] for r in validation}.isdisjoint(observations.recorded_video_ids()))
