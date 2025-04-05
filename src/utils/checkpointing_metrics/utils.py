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


"""Utilities for checkpointing benchmark results processing."""

import datetime
import itertools
import multiprocessing.pool
import re
from google.cloud import storage
import log_patterns


def process_logs_files(
    logs_bucket: storage.bucket.Bucket,
    match_glob: str = None,
    process_logs_file=None,
    filename_val=None,
):
  """Iterate through the log files to process raw metrics.

  Args:
      logs_bucket: The bucket which contains the logs from
        the benchmark run.
      match_glob: The pattern to filter the paths of the log files.
      process_logs_file: The function to process raw metrics in each log file.
          The process_file function should adhere to:
              func(
                  logs_bucket: storage.bucket.Bucket,
                  file_path: str,
              ) -> list
      filename_val: The function to validate file names. The filename_val
          function should adhere to:
              func(
                file_path: str,
              ) -> bool

  Returns:
      A list of dictionaries, representing metrics data per global_rank.

  """
  storage_client = storage.Client()

  try:
    blobs = storage_client.list_blobs(logs_bucket, match_glob=match_glob)
    files = [blob.name for blob in blobs if filename_val(blob.name)]

    with multiprocessing.pool.ThreadPool(multiprocessing.cpu_count()) as pool:
      data = pool.map(
          lambda x: process_logs_file(logs_bucket, x),
          files,
      )

    filtered_data = [item for item in data if item is not None]
    return list(itertools.chain.from_iterable(filtered_data))

  except Exception as e:
    print(f"Error: Failed to process the logs files for raw metrics: {e}")
    raise


def parse_nemo_timestamp(line: str):
  """Parse the timestamp from a NeMo log line.

  Args:
      line: A line from NeMo logs.

  Returns:
      The timestamp for the NeMo log line.
  """
  time_match = re.search(log_patterns.NEMO_LOG_TIMESTAMP, line)
  if time_match is None:
    raise ValueError(f"Failed to fetch the timestamp from line {line}.")

  try:
    timestamp = datetime.datetime.strptime(
        time_match.group(1), "%Y-%m-%d %H:%M:%S"
    )
    return float(timestamp.timestamp())

  except Exception as e:
    print(f"Error: Failed to parse the timestamp from line {line}: {e}")
    raise