# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Tests for checkpointing metrics processing (NeMo 2 format)."""

import io
import re
import sys
import os
import unittest
from unittest import mock

# Add the module directory to sys.path so we can import the modules.
sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)

# Mock google.cloud.storage before importing modules that depend on it.
sys.modules["google"] = mock.MagicMock()
sys.modules["google.cloud"] = mock.MagicMock()
sys.modules["google.cloud.storage"] = mock.MagicMock()

import log_patterns
import utils
import calculate_checkpoint_metrics


# Sample NeMo 2 log lines for testing.
SAMPLE_NEMO2_LOG = """\
[NeMo I 2026-02-17 17:58:40 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Iteration: 24 : Start time: 1771351120.135s : Save duration: 25.537s
[NeMo I 2026-02-17 17:58:41 nemo_logging:393] Scheduled async checkpoint save for /ckpt/step=24.ckpt
[NeMo I 2026-02-17 17:58:41 nemo_logging:393] Async finalization time took 0.018 s
[NeMo I 2026-02-17 17:58:50 nemo_logging:393] Successfully saved checkpoint from iteration 24 to /ckpt/step=24.ckpt
[NeMo I 2026-02-17 17:58:50 nemo_logging:393] Async checkpoint save for step 25 (/ckpt/step=24.ckpt) finalized successfully.
[NeMo I 2026-02-17 17:59:05 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Iteration: 49 : Start time: 1771351145.879s : Save duration: 5.330s
[NeMo I 2026-02-17 17:59:06 nemo_logging:393] Scheduled async checkpoint save for /ckpt/step=49.ckpt
[NeMo I 2026-02-17 17:59:15 nemo_logging:393] Successfully saved checkpoint from iteration 49 to /ckpt/step=49.ckpt
[NeMo I 2026-02-17 17:59:15 nemo_logging:393] Async checkpoint save for step 50 (/ckpt/step=49.ckpt) finalized successfully.
"""


class TestLogPatterns(unittest.TestCase):
  """Tests for NeMo 2 log pattern regex matching."""

  def test_checkpoint_write_start_matches_nemo2(self):
    line = (
        "[NeMo I 2026-02-17 17:58:40 nemo_logging:393] Global Checkpoint"
        " Save : Rank: 0 : Iteration: 24 : Start time: 1771351095.135s"
        " : Save duration: 25.537s"
    )
    match = re.search(log_patterns.CHECKPOINT_WRITE_START, line)
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "24")
    self.assertEqual(match.group(2), "1771351095.135")
    self.assertEqual(match.group(3), "25.537")

  def test_checkpoint_write_start_multi_digit_rank(self):
    line = (
        "[NeMo I 2026-02-17 17:58:40 nemo_logging:393] Global Checkpoint"
        " Save : Rank: 63 : Iteration: 99 : Start time: 1771351190.030s"
        " : Save duration: 1.453s"
    )
    match = re.search(log_patterns.CHECKPOINT_WRITE_START, line)
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "99")
    self.assertEqual(match.group(2), "1771351190.030")
    self.assertEqual(match.group(3), "1.453")

  def test_checkpoint_write_start_does_not_match_nemo1(self):
    line = (
        "2024-08-15 10:30:00 Checkpoint save for step 100 started"
    )
    match = re.search(log_patterns.CHECKPOINT_WRITE_START, line)
    self.assertIsNone(match)

  def test_checkpoint_write_end_matches_nemo2(self):
    line = (
        "[NeMo I 2026-02-17 17:58:50 nemo_logging:393] Async checkpoint"
        " save for step 25 (/ckpt/step=24.ckpt) finalized successfully."
    )
    match = re.search(log_patterns.CHECKPOINT_WRITE_END, line)
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "25")

  def test_nemo_log_file_name_pattern(self):
    path = "logs/nemo_log_globalrank-0_localrank-0.txt"
    match = re.search(log_patterns.NEMO_LOG_FILE_NAME, path)
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "0")
    self.assertEqual(match.group(2), "0")

  def test_nemo_timestamp_pattern(self):
    line = "[NeMo I 2026-02-17 17:58:50 nemo_logging:393] Some message"
    match = re.search(log_patterns.NEMO_LOG_TIMESTAMP, line)
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "2026-02-17 17:58:50")


class TestParseNemoTimestamp(unittest.TestCase):
  """Tests for timestamp parsing from NeMo 2 log lines."""

  def test_parse_valid_timestamp(self):
    line = "[NeMo I 2026-02-17 17:58:50 nemo_logging:393] Some message"
    timestamp = utils.parse_nemo_timestamp(line)
    self.assertIsInstance(timestamp, float)
    self.assertGreater(timestamp, 0)

  def test_parse_invalid_timestamp_raises(self):
    line = "no timestamp here"
    with self.assertRaises(ValueError):
      utils.parse_nemo_timestamp(line)


class TestProcessCkptWriteTimes(unittest.TestCase):
  """Tests for processing checkpoint write times from NeMo 2 logs."""

  def setUp(self):
    # Suppress warnings during tests.
    calculate_checkpoint_metrics.generate_warnings = False

  @mock.patch.object(
      calculate_checkpoint_metrics, "generate_warnings", False
  )
  def test_process_nemo2_log_extracts_checkpoints(self):
    """Verify that NeMo 2 log lines are correctly parsed."""
    mock_bucket = mock.MagicMock()
    mock_blob = mock.MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_blob.download_as_string.return_value = (
        SAMPLE_NEMO2_LOG.encode("utf-8")
    )

    file_path = "logs/nemo_log_globalrank-0_localrank-0.txt"
    results = calculate_checkpoint_metrics.process_ckpt_write_times(
        mock_bucket, file_path
    )

    self.assertIsNotNone(results)
    self.assertEqual(len(results), 2)

    # First checkpoint: iteration 24.
    self.assertEqual(results[0]["global_rank"], 0)
    self.assertEqual(results[0]["local_rank"], 0)
    self.assertEqual(results[0]["checkpoint_step"], "24")
    self.assertEqual(results[0]["start_time"], 1771351120.135)
    self.assertGreater(results[0]["end_time"], results[0]["start_time"])
    self.assertGreater(results[0]["checkpoint_write_duration"], 0)

    # Second checkpoint: iteration 49.
    self.assertEqual(results[1]["checkpoint_step"], "49")
    self.assertEqual(results[1]["start_time"], 1771351145.879)

  def test_process_invalid_file_path(self):
    mock_bucket = mock.MagicMock()
    mock_blob = mock.MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_blob.download_as_string.return_value = b""

    file_path = "logs/invalid_file_name.txt"
    result = calculate_checkpoint_metrics.process_ckpt_write_times(
        mock_bucket, file_path
    )
    # Should print an error and return None.
    self.assertIsNone(result)


class TestComputeWriteDurationPerStep(unittest.TestCase):
  """Tests for computing write duration per step."""

  def test_single_rank_multiple_steps(self):
    write_times = [
        {
            "global_rank": 0,
            "local_rank": 0,
            "checkpoint_step": "24",
            "checkpoint_write_duration": 10.0,
            "start_time": 100.0,
            "end_time": 110.0,
        },
        {
            "global_rank": 0,
            "local_rank": 0,
            "checkpoint_step": "49",
            "checkpoint_write_duration": 12.0,
            "start_time": 200.0,
            "end_time": 212.0,
        },
    ]
    # Should print stats without error.
    calculate_checkpoint_metrics.compute_write_duration_per_step(
        write_times
    )

  def test_multi_rank_multiple_steps(self):
    write_times = [
        {
            "global_rank": 0,
            "local_rank": 0,
            "checkpoint_step": "24",
            "checkpoint_write_duration": 10.0,
            "start_time": 100.0,
            "end_time": 110.0,
        },
        {
            "global_rank": 1,
            "local_rank": 1,
            "checkpoint_step": "24",
            "checkpoint_write_duration": 12.0,
            "start_time": 99.0,
            "end_time": 111.0,
        },
        {
            "global_rank": 0,
            "local_rank": 0,
            "checkpoint_step": "49",
            "checkpoint_write_duration": 8.0,
            "start_time": 200.0,
            "end_time": 208.0,
        },
        {
            "global_rank": 1,
            "local_rank": 1,
            "checkpoint_step": "49",
            "checkpoint_write_duration": 9.0,
            "start_time": 199.0,
            "end_time": 208.0,
        },
    ]
    # Duration per step:
    #   step 24: max(110,111) - min(100,99) = 12
    #   step 49: max(208,208) - min(200,199) = 9
    calculate_checkpoint_metrics.compute_write_duration_per_step(
        write_times
    )

  def test_empty_write_times(self):
    # Should print a warning and not crash.
    calculate_checkpoint_metrics.compute_write_duration_per_step([])

  def test_missing_fields(self):
    write_times = [{"global_rank": 0}]
    # Should print a warning about missing fields.
    calculate_checkpoint_metrics.compute_write_duration_per_step(
        write_times
    )


if __name__ == "__main__":
  unittest.main()
