import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from tests.support import csv_text, video
from fastapi.testclient import TestClient
from data import observations, storage
from serving_app.main import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.patches = [patch("serving_app.monitoring.retrain_trigger.check_configured_baseline"), patch.object(observations, "OBSERVATION_DB", Path(self.directory.name) / "observations.db"),
                        patch.object(storage, "UPLOAD_DIR", Path(self.directory.name) / "uploads"),
                        patch.dict("os.environ", {"BASELINE_RMSLE": "0.1", "BASELINE_MODEL_VERSION": "1", "AUTO_RETRAIN": "false"})]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.model = Mock(version="1", metadata={"used_video_ids": []})
        self.model.predict_many.side_effect = lambda rows: [8400] * len(rows)
        self.loader = patch("serving_app.model_loader.get_model", return_value=self.model)
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.client = TestClient(app)

    def test_upload_predict_and_observe(self):
        text = csv_text([video()])
        self.assertEqual(self.client.post("/data/upload", files={"file": ("videos.csv", text)}).status_code, 200)
        status = self.client.get("/data/status").json()
        self.assertEqual(status["rows"], 1)
        prediction = self.client.post("/predict/csv", files={"file": ("videos.csv", text)}).json()["predictions"][0]
        result = self.client.post("/observations", json={"prediction_id": prediction["prediction_id"], "target_views_day7": 12000})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["drift_check"]["pending_count"], 1)

    def test_simulation_uses_video_features_and_does_not_change_training_upload(self):
        self.client.post("/data/upload", files={"file": ("training.csv", csv_text([video(500)]))})
        original = self.client.get("/data/status").json()
        response = self.client.post("/predict/batch-test", json={"videos": [video(i) for i in range(60)]})
        self.assertEqual(response.status_code, 200)
        check = response.json()["drift_check"]
        self.assertEqual(check["status"], "retrain_review")
        self.assertEqual(check["retraining"]["status"], "manual_review")
        self.assertEqual(self.client.get("/data/status").json(), original)
        self.assertEqual(observations.labeled_rows("live"), [])
        replay = self.client.post("/predict/batch-test", json={"videos": [video(i) for i in range(60)]})
        self.assertEqual(replay.json()["drift_check"]["new_blocks"], [])

    def test_missing_model_has_actionable_response(self):
        self.model.predict_many.side_effect = FileNotFoundError("모델을 먼저 학습하세요.")
        row = video()
        row.pop("target_views_day7")
        response = self.client.post("/predict", json=row)
        self.assertEqual(response.status_code, 503)
        self.assertIn("학습", response.json()["detail"])

    def test_bad_csv_and_future_prediction(self):
        response = self.client.post("/data/upload", files={"file": ("invalid.csv", csv_text([video(target_views_day7=1)]))})
        self.assertEqual(response.status_code, 400)
        row = video(published_at=datetime.now(timezone.utc).isoformat())
        row.pop("target_views_day7")
        self.assertEqual(self.client.post("/predict", json=row).status_code, 400)
        self.assertEqual(self.client.post("/observations", json={"prediction_id": "none", "target_views_day7": 10}).status_code, 404)

    def test_training_video_is_excluded_from_drift(self):
        self.model.metadata = {"used_video_ids": ["video_0"]}
        response = self.client.post("/predict/batch-test", json={"videos": [video()]})
        self.assertEqual(response.status_code, 400)

    def test_duplicate_video_batch_is_rejected(self):
        response = self.client.post("/predict/batch-test", json={"videos": [video(), video()]})
        self.assertEqual(response.status_code, 422)

    def test_scenario_validation_stays_out_of_drift_records(self):
        validation = [video(i) for i in range(100, 160)]
        with patch('serving_app.train_and_register.fine_tune', return_value={'promoted': True, 'version': '2'}) as fit, \
             patch.dict('os.environ', {'AUTO_RETRAIN': 'true'}):
            first = self.client.post('/predict/batch-test', json={'videos': [video(i) for i in range(30)]})
            self.assertEqual(first.json()['drift_check']['status'], 'warning')
            response = self.client.post('/predict/batch-test', json={
                'videos': [video(i) for i in range(30, 60)], 'retraining_validation': validation})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(fit.call_args.args[0]), 60)
        from serving_app.schemas import LabeledVideo
        self.assertEqual(fit.call_args.args[1], [LabeledVideo.model_validate(r).model_dump(mode='json') for r in validation])
        self.assertEqual(len(observations.recorded_video_ids()), 60)
        self.assertTrue(response.json()['drift_check']['retraining']['baseline_required'])
