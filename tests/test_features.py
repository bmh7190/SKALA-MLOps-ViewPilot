import math
import tempfile
import unittest
from pathlib import Path

from tests.support import csv_text, video
from data.features import VideoPreprocessor, build_sequences, load_rows, parse_csv, restore_prediction, validate_splits
from serving_app.schemas import LabeledVideo


class FeatureTests(unittest.TestCase):
    def test_user_csv_and_three_step_shape(self):
        rows = load_rows(Path(__file__).parents[1] / "data" / "sample_videos.csv")
        preprocessor = VideoPreprocessor.fit(rows)
        X, y = build_sequences(rows, preprocessor)
        self.assertEqual((len(X), len(X[0]), len(X[0][0])), (3, 3, 7))
        self.assertAlmostEqual(y[0], math.log1p(12000))
        self.assertNotEqual(X[0][0][0], X[0][1][0])
        self.assertEqual(X[0][0][3:], X[0][2][3:])

    def test_target_cannot_change_input(self):
        row = video()
        processor = VideoPreprocessor.fit([row])
        changed = {**row, "target_views_day7": 9999999, "video_id": "another_id"}
        self.assertEqual(processor.transform(row), processor.transform(changed))
        self.assertNotIn("target_views_day7", LabeledVideo.model_validate(row).features_dict())

    def test_unknown_category_and_save_round_trip(self):
        rows = [video(0), video(1, category="education")]
        processor = VideoPreprocessor.fit(rows)
        unknown = video(2, category="new_category")
        self.assertEqual(processor.transform(unknown)[0][-2:], [0.0, 0.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preprocessor.json"
            processor.save(path)
            loaded = VideoPreprocessor.load(path)
        self.assertEqual(loaded.transform(unknown), processor.transform(unknown))

    def test_explicit_validation_does_not_change_training_preprocessor(self):
        train, validation = validate_splits([video(i) for i in range(40)],
                                         [video(i, category="future") for i in range(50, 60)])
        processor = VideoPreprocessor.fit(train)
        self.assertEqual(len(train), 40)
        self.assertEqual(len(validation), 10)
        self.assertNotIn("future", processor.categories)
        with self.assertRaises(ValueError):
            validate_splits([video(i) for i in range(40)], [video(i) for i in range(38, 48)])
        with self.assertRaises(ValueError):
            validate_splits([video(i) for i in range(40)], [video(i) for i in range(40, 50)])

    def test_prediction_csv_does_not_need_label_and_ignores_supplied_label(self):
        self.assertNotIn("target_views_day7", parse_csv(csv_text([video()], target=False), False)[0])
        self.assertNotIn("target_views_day7", parse_csv(csv_text([video(target_views_day7="")]), False)[0])

    def test_reject_bad_csv(self):
        bad_rows = [video(day1_views=-1), video(target_views_day7=1), video(day1_avg_view_percentage="NaN"),
                    video(published_at="2020-01-01"), video(category=" ")]
        for row in bad_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                parse_csv(csv_text([row]))
        with self.assertRaises(ValueError):
            parse_csv(csv_text([video(), video()]))
        with self.assertRaises(ValueError):
            parse_csv("video_id,video_id\na,a\n")

    def test_percentage_above_100_is_allowed_and_output_has_cumulative_floor(self):
        rows = parse_csv(csv_text([video(day1_avg_view_percentage=120)]))
        self.assertEqual(rows[0]["day1_avg_view_percentage"], 120)
        self.assertEqual(restore_prediction(-5, rows[0]), 8400)
        self.assertEqual(restore_prediction(math.log1p(12000), rows[0]), 12000)
        with self.assertRaises(ValueError):
            restore_prediction(float("inf"), rows[0])
