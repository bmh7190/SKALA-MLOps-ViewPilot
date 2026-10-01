import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from tests.support import video
from data import observations
from serving_app.main import app


class HistoryApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db = Path(self.directory.name) / 'observations.db'
        patched = patch.object(observations, 'OBSERVATION_DB', self.db)
        patched.start()
        self.addCleanup(patched.stop)
        self.client = TestClient(app)

    def test_empty_history_does_not_create_database_or_load_model(self):
        with patch('serving_app.model_loader.get_model', side_effect=AssertionError('must not load')):
            for path in ['/api/v1/videos/predictions', '/api/v1/monitoring/drift']:
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {'items': [], 'total': 0, 'page': 1, 'page_size': 20})
        self.assertFalse(self.db.exists())

    def test_prediction_pagination_order_null_and_actual_update(self):
        forecasts = observations.save_predictions([video(i) for i in range(3)], [10000] * 3, '1', 'live')
        observations.save_actual(forecasts[0]['prediction_id'], 12000)
        first = self.client.get('/api/v1/videos/predictions?page_size=2').json()
        self.assertEqual(first['total'], 3)
        self.assertEqual([r['video_id'] for r in first['items']], ['video_2', 'video_1'])
        self.assertIsNone(first['items'][0]['target_views_day7'])
        self.assertIsNone(first['items'][0]['observed_at'])
        self.assertIsNone(first['items'][0]['block_id'])
        second = self.client.get('/api/v1/videos/predictions?page_size=2&page=2').json()
        self.assertEqual(second['items'][0]['target_views_day7'], 12000)
        self.assertEqual(second['items'][0]['predicted_views_day7'], 10000)
        self.assertIsNotNone(second['items'][0]['observed_at'])
        # 새 TestClient도 같은 DB 이력을 읽는다.
        reopened = TestClient(app).get('/api/v1/videos/predictions?page=3&page_size=2').json()
        self.assertEqual(reopened['total'], 3)
        self.assertEqual(reopened['items'], [])

    def test_mode_and_model_filters_are_combined_and_parameterized(self):
        for version, mode in [('1', 'live'), ('1', 'simulation'), ('2', 'simulation')]:
            observations.save_predictions([video()], [10000], version, mode)
        filtered = self.client.get('/api/v1/videos/predictions', params={
            'mode': 'simulation', 'model_version': '2'}).json()
        self.assertEqual(filtered['total'], 1)
        self.assertEqual(filtered['items'][0]['model_version'], '2')
        self.assertEqual(filtered['items'][0]['mode'], 'simulation')
        self.assertEqual(self.client.get('/api/v1/videos/predictions?mode=simulation').json()['total'], 2)
        query = self.client.get('/api/v1/videos/predictions', params={'model_version': "1' OR 1=1 --"})
        self.assertEqual(query.json()['total'], 0)
        self.assertEqual(self.client.get('/api/v1/videos/predictions').json()['total'], 3)

    def test_drift_history_keeps_original_baseline_and_ignores_pending_rows(self):
        observations.save_predictions([video(i) for i in range(61)], [8400] * 61, '1', 'simulation')
        observations.evaluate_pending('1', 'simulation', 0.1)
        observations.save_predictions([video(i) for i in range(30)], [12000] * 30, '2', 'simulation')
        observations.evaluate_pending('2', 'simulation', 0.3)
        live = observations.save_predictions([video(i) for i in range(30)], [12000] * 30, '1', 'live')
        for row in live:
            observations.save_actual(row['prediction_id'], 12000)
        observations.evaluate_pending('1', 'live', 0.2)
        with patch.dict('os.environ', {'BASELINE_RMSLE': '9', 'BASELINE_MODEL_VERSION': '2'}), \
             patch('data.observations.evaluate_pending', side_effect=AssertionError('must not evaluate')):
            result = self.client.get('/api/v1/monitoring/drift', params={
                'mode': 'simulation', 'model_version': '1', 'page_size': 1}).json()
            self.assertEqual(result['total'], 2)
            row = result['items'][0]
            self.assertEqual(row['status'], 'retrain_review')
            self.assertEqual(row['consecutive_exceeds'], 2)
            self.assertEqual(row['baseline_rmsle'], 0.1)
            self.assertEqual(row['threshold'], 0.2)
            self.assertEqual(len(row['video_ids']), 30)
            earlier = self.client.get('/api/v1/monitoring/drift', params={
                'mode': 'simulation', 'model_version': '1', 'page_size': 1, 'page': 2}).json()
            self.assertEqual(earlier['items'][0]['status'], 'warning')
            self.assertLess(earlier['items'][0]['block_id'], row['block_id'])
        self.assertEqual(self.client.get('/api/v1/monitoring/drift').json()['total'], 4)
        self.assertEqual(observations.evaluate_pending('1', 'simulation', 0.1)['pending_count'], 1)

    def test_history_reads_leave_database_bytes_unchanged_and_reset_shows_empty(self):
        observations.save_predictions([video(i) for i in range(30)], [10000] * 30, '1', 'simulation')
        observations.evaluate_pending('1', 'simulation', 0.1)
        before = self.db.read_bytes()
        for endpoint in ['/api/v1/videos/predictions', '/api/v1/monitoring/drift']:
            self.assertEqual(self.client.get(endpoint).status_code, 200)
        self.assertEqual(before, self.db.read_bytes())
        self.db.unlink()
        self.assertEqual(self.client.get('/api/v1/monitoring/drift').json()['total'], 0)
        self.assertFalse(self.db.exists())

    def test_invalid_filters_and_pagination_are_rejected(self):
        for path in ['/api/v1/videos/predictions', '/api/v1/monitoring/drift']:
            for params in [{'page': 0}, {'page_size': 0}, {'page_size': 101},
                           {'mode': 'unknown'}, {'model_version': ''}]:
                with self.subTest(path=path, params=params):
                    self.assertEqual(self.client.get(path, params=params).status_code, 422)
