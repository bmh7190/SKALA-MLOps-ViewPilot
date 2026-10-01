import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.support import video
from data import baselines, observations


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for module, name, path in [(baselines, "BASELINE_DIR", "baselines"),
                                   (observations, "OBSERVATION_DB", "observations.db")]:
            p = patch.object(module, name, Path(self.directory.name) / path)
            p.start()
            self.addCleanup(p.stop)
        self.model = SimpleNamespace(version="1", metadata={"used_video_ids": [], "run_id": "run1"},
                                     predict_many=lambda rows: [12000] * len(rows))
        self.rows = [video(i) for i in range(30)]

    def test_calculation_is_separate_from_operating_observations(self):
        result = baselines.calculate_baseline(self.model, self.rows)
        self.assertEqual(result["baseline_rmsle"], 0)
        baselines.save_baseline(result)
        self.assertEqual(len(baselines.baseline_video_ids()), 30)
        self.assertFalse(observations.OBSERVATION_DB.exists())
        baselines.check_configured_baseline("1", 0)
        with self.assertRaises(ValueError):
            baselines.check_configured_baseline("1", 0.1)

    def test_baseline_rejects_training_videos_and_wrong_size(self):
        with self.assertRaises(ValueError):
            baselines.calculate_baseline(self.model, self.rows[:29])
        self.model.metadata["used_video_ids"] = ["video_0"]
        with self.assertRaises(ValueError):
            baselines.calculate_baseline(self.model, self.rows)

    def test_saved_baseline_cannot_be_overwritten(self):
        result = baselines.calculate_baseline(self.model, self.rows)
        baselines.save_baseline(result)
        baselines.save_baseline(result)
        with self.assertRaises(ValueError):
            baselines.save_baseline({**result, "baseline_rmsle": 0.2})
