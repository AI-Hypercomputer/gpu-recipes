"""Tests to unit test GoodputCache class."""

import datetime
from unittest import mock

from cloud_goodput.ml_goodput_measurement.src import goodput_cache
from cloud_goodput.ml_goodput_measurement.src import goodput_utils
from cloud_goodput.ml_goodput_measurement.src.goodput_utils import BadputType, GoodputInfo

from google3.testing.pybase import googletest


class GoodputCacheTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.goodput_cache = goodput_cache.GoodputCache()

  def test_update_cached_entries(self):
    mock_entries = [
        {'time': 1, 'step': 1},
        {'time': 2, 'step': 2},
        {'time': 3, 'step': 3},
    ]
    self.goodput_cache.update_cached_entries(mock_entries)
    self.assertFalse(self.goodput_cache.is_cache_empty())
    self.assertEqual(self.goodput_cache.get_cached_entries(), mock_entries)

  def test_update_goodput_info(self):
    goodput_info = GoodputInfo(
        total_productive_time=100,
        total_elapsed_time_since_start=200,
        total_unproductive_time={
            BadputType.TPU_INITIALIZATION: 10,
            BadputType.TRAINING_PREP: 10,
            BadputType.DATA_LOADING_SYNC: 30,
            BadputType.PROGRAM_STARTUP: 10,
            BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME: 20,
            BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME: 10,
            BadputType.WASTED_PROGRESS_FROM_DISRUPTION: 10,
            BadputType.OTHER: 10,
        },
        last_recorded_step=3,
    )
    self.goodput_cache.update_goodput_info(goodput_info)
    self.assertEqual(self.goodput_cache._goodput_info, goodput_info)

  def test_clear_cache(self):
    mock_entries = [
        {'time': 1, 'step': 1},
        {'time': 2, 'step': 2},
        {'time': 3, 'step': 3},
    ]
    self.goodput_cache.update_cached_entries(mock_entries)
    self.goodput_cache.update_goodput_info(
        GoodputInfo(
            total_productive_time=100,
            total_elapsed_time_since_start=200,
            total_unproductive_time={
                BadputType.TPU_INITIALIZATION: 10,
                BadputType.TRAINING_PREP: 10,
                BadputType.DATA_LOADING_SYNC: 30,
                BadputType.PROGRAM_STARTUP: 10,
                BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME: 20,
                BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME: 10,
                BadputType.WASTED_PROGRESS_FROM_DISRUPTION: 10,
                BadputType.OTHER: 10,
            },
            last_recorded_step=3,
        )
    )
    self.goodput_cache.clear_cache()
    self.assertEqual(self.goodput_cache.get_cached_entries(), [])
    self.assertIsNone(self.goodput_cache._goodput_info)
    self.assertIsNone(self.goodput_cache._last_entry_timestamp)

  def test_is_cache_empty(self):
    self.assertTrue(self.goodput_cache.is_cache_empty())
    self.goodput_cache.update_cached_entries([
        {'time': 1, 'step': 1},
        {'time': 2, 'step': 2},
        {'time': 3, 'step': 3},
    ])
    self.assertFalse(self.goodput_cache.is_cache_empty())

  def test_get_last_entry_timestamp(self):
    self.assertIsNone(self.goodput_cache._last_entry_timestamp)
    self.goodput_cache.update_cached_entries([
        {'time': 1, 'step': 1},
        {'time': 2, 'step': 2},
        {'time': 3, 'step': 3},
    ])
    self.assertFalse(self.goodput_cache.is_cache_empty())
    self.assertEqual(
        self.goodput_cache._last_entry_timestamp,
        datetime.datetime.fromtimestamp(3, tz=datetime.timezone.utc),
    )

  def test_get_step_info(self):
    step_info = goodput_utils.StepInfo(
        step_deviations={1: 1.0, 2: 2.0},
        ideal_step_time=1.0,
    )
    self.goodput_cache.update_step_info(step_info)
    self.assertEqual(self.goodput_cache._step_info, step_info)

  def test_update_job_start_time(self):
    self.assertIsNone(self.goodput_cache._job_start_time)
    self.goodput_cache.update_cached_entries([
        {'step_start_time': 2, 'step': 1},
        {'step_start_time': 3, 'step': 2},
        {'job_end_time': 4},
    ])
    self.assertIsNone(self.goodput_cache._job_start_time)
    self.goodput_cache.update_cached_entries([
        {'job_start_time': 1},
        {'job_start_time': 9},
        {'step_start_time': 2, 'step': 1},
        {'step_start_time': 3, 'step': 2},
        {'job_end_time': 4},
    ])
    self.assertEqual(
        self.goodput_cache._job_start_time,
        datetime.datetime.fromtimestamp(1, tz=datetime.timezone.utc),
    )

  def test_update_job_end_time(self):
    self.assertIsNone(self.goodput_cache._job_end_time)
    self.goodput_cache.update_cached_entries([
        {'job_end_time': 1},
        {'job_end_time': 2},
        {'job_end_time': 3},
    ])
    self.assertEqual(
        self.goodput_cache._job_end_time,
        datetime.datetime.fromtimestamp(3, tz=datetime.timezone.utc),
    )


if __name__ == '__main__':
  googletest.main()
