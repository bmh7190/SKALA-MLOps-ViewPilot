import unittest
from pathlib import Path

import tests  # 임시 runtime 설정
from data.features import VideoPreprocessor, load_rows, validate_splits


class DataBundleTests(unittest.TestCase):
    def test_provided_splits_are_disjoint_and_compatible(self):
        root = Path(__file__).parents[1] / "data" / "synthetic"
        counts = {"train": 3000, "validation": 600, "baseline_normal": 30,
                  "test_normal": 120, "test_drift": 120}
        datasets = {}
        seen = set()
        for name, count in counts.items():
            rows = load_rows(root / f"{name}.csv")
            self.assertEqual(len(rows), count)
            ids = {row["video_id"] for row in rows}
            self.assertFalse(ids & seen)
            seen.update(ids)
            datasets[name] = rows
        training, validation = validate_splits(datasets["train"], datasets["validation"])
        self.assertEqual((len(training), len(validation)), (3000, 600))
        processor = VideoPreprocessor.fit(training)
        self.assertEqual(processor.categories, ["gaming", "education", "entertainment", "lifestyle"])
        self.assertEqual((len(processor.transform(training[0])), processor.n_features), (3, 8))
