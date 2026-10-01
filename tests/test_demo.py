import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serving_app.demo import reset_demo_runtime


class DemoResetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        bundle = self.root / "models" / "seed-run"
        bundle.mkdir(parents=True)
        (bundle / "model.keras").write_text("seed weights")
        (bundle / "preprocessor.json").write_text("{}")
        (bundle / "metadata.json").write_text(json.dumps({"run_id": "seed-run"}))
        (bundle.parent / "production.json").write_text(json.dumps({"bundle_id": "seed-run", "version": "1"}))
        (self.root / "baselines").mkdir()
        (self.root / "baselines" / "1.json").write_text(json.dumps({
            "model_version": "1", "run_id": "seed-run", "baseline_rmsle": 0.1}))
        (self.root / "observations.db").write_text("original observations")
        with sqlite3.connect(self.root / "mlflow.db") as db:
            db.execute("CREATE TABLE experiments (name TEXT, artifact_location TEXT)")
            db.execute("INSERT INTO experiments VALUES (?, ?)", ("viewpilot", "file:///original"))
        self.environment = patch.dict(os.environ, {})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_restart_restores_model_and_clears_only_demo_state(self):
        originals = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        reset_demo_runtime(self.root)
        demo = self.root / 'demo'
        (demo / 'models/production.json').write_text('{"version":"2"}')
        (demo / 'observations.db').write_text('demo observations')
        (demo / 'logs').mkdir()
        (demo / 'logs/aiops.log').write_text('previous log')
        os.environ['BASELINE_MODEL_VERSION'] = '2'
        reset_demo_runtime(self.root)
        self.assertEqual(json.loads((demo / 'models/production.json').read_text())['version'], '1')
        self.assertFalse((demo / 'observations.db').exists())
        self.assertFalse((demo / 'logs').exists())
        self.assertEqual(os.environ['BASELINE_MODEL_VERSION'], '1')
        self.assertEqual(os.environ['BASELINE_RMSLE'], '0.1')
        for path, content in originals.items():
            self.assertEqual((self.root / path).read_bytes(), content)
        with sqlite3.connect(demo / 'mlflow.db') as db:
            location = db.execute('SELECT artifact_location FROM experiments').fetchone()[0]
        self.assertEqual(location, (demo / 'mlruns').resolve().as_uri())

    def test_missing_seed_does_not_erase_previous_demo(self):
        reset_demo_runtime(self.root)
        marker = self.root / 'demo/keep.txt'
        marker.write_text('previous demo')
        (self.root / 'baselines/1.json').unlink()
        with self.assertRaises(ValueError):
            reset_demo_runtime(self.root)
        self.assertEqual(marker.read_text(), 'previous demo')

    def test_unknown_folder_and_symlink_are_not_deleted(self):
        demo = self.root / 'demo'
        demo.mkdir()
        with self.assertRaises(ValueError):
            reset_demo_runtime(self.root)
        demo.rmdir()
        demo.symlink_to(self.root / 'models', target_is_directory=True)
        with self.assertRaises(ValueError):
            reset_demo_runtime(self.root)
        self.assertTrue((self.root / 'models/production.json').exists())
