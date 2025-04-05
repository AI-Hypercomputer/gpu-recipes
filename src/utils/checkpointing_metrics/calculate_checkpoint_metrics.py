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


"""Tool to process checkpointing metrics from logs."""

import argparse
import io
import os
import re
import statistics
from google.cloud import storage
import log_patterns
import utils


def process_metrics_from_logs(
    gcs_logs_path: str,
):
  """Process NeMo logs stored in a GCS bucket and calculate checkpointing
  metrics.

  Args:
    gcs_logs_path: The path to the NeMo logs in a GCS bucket.
  """

  storage_client = storage.Client()
  logs_bucket_name = gcs_logs_path.split("/")[2]
  match_glob = f'{"/".join(gcs_logs_path.split("/")[3:])}/**'
  logs_bucket = storage_client.bucket(logs_bucket_name)

  ckpt_write_times = utils.process_logs_files(
      logs_bucket=logs_bucket,
      match_glob=match_glob,
      process_logs_file=process_ckpt_write_times,
      filename_val=lambda file_path: re.search(
          log_patterns.NEMO_LOG_FILE_NAME, file_path
      )
      is not None,
  )

  compute_write_duration_per_step(ckpt_write_times)


def process_ckpt_write_times(
    logs_bucket: storage.bucket.Bucket,
    file_path: str,
):
  """Process checkpoint write times from NeMo logs.

  Args:
      logs_bucket: The bucket which contains the logs from
        the benchmark run.
      file_path: The path to the NeMo log file.

  Returns:
      A list of dictionaries, representing ckpt write data per global_rank.

  """
  global generate_warnings

  ckpt_write_results = []
  ckpt_write_times = {}
  blob = logs_bucket.blob(file_path)

  try:
    content = blob.download_as_string()
    stream = io.TextIOWrapper(io.BytesIO(content), encoding="utf-8")

    file_path_match = re.search(log_patterns.NEMO_LOG_FILE_NAME, file_path)
    if not file_path_match:
      raise ValueError(
          f"Invalid file path: {file_path}. Valid pattern:"
          "nemo_log_globalrank-[idx]_localrank-[idx]"
      )

    global_rank, local_rank = map(int, file_path_match.groups())
    print(
        f"Analyzing file: {file_path}, Global rank: {global_rank}, Local rank:"
        f" {local_rank}"
    )

    for line in stream:
      start_match = re.search(log_patterns.CHECKPOINT_WRITE_START, line)
      if start_match:
        step = start_match.group(1)
        start_time = utils.parse_nemo_timestamp(line)

        if ckpt_write_times.get(step, {}).get("start_time"):
          if generate_warnings:
            print(
                f"Warning: Duplicate checkpoint write start time at step {step}"
                f" in file {file_path}. We only keep the first occurence."
            )
          continue

        ckpt_write_times[step] = {"start_time": start_time}
        continue

      end_match = re.search(log_patterns.CHECKPOINT_WRITE_END, line)
      if end_match:
        step = end_match.group(1)
        end_time = utils.parse_nemo_timestamp(line)

        if ckpt_write_times.get(step, {}).get("start_time") is None:
          raise ValueError(
              f"Checkpoing write at step {step} has the end time"
              f" reported prior to its start time in file {file_path}"
          )

        if ckpt_write_times.get(step, {}).get("end_time"):
          if generate_warnings:
            print(
                f"Warning: Duplicate checkpoing write end time at step {step}"
                f" in file {file_path}. We only keep the first occurence."
            )
          continue

        start_time = ckpt_write_times[step]["start_time"]
        ckpt_write_results.append({
            "global_rank": global_rank,
            "local_rank": local_rank,
            "checkpoint_step": step,
            "checkpoint_write_duration": end_time - start_time,
            "start_time": start_time,
            "end_time": end_time,
        })
        ckpt_write_times[step]["end_time"] = end_time

    return ckpt_write_results

  except Exception as e:
    print(f"Error: Failed to process {file_path}: {e}")


def compute_write_duration_per_step(write_times: list[dict[str, any]]):
  """Calculate and print out the checkpoint write duration for each step.

  We use the difference between the earliest start time and the latest end
  time across all ranks to calculate the checkpoint write duration.

  Args:
    write_times: Checkpoint write start and end times per step, by rank.

  """

  write_time_dict = {}

  for time in write_times:
    step = time.get("checkpoint_step")
    start_time = time.get("start_time")
    end_time = time.get("end_time")

    if any(key is None for key in (step, start_time, end_time)):
      print(
          "Warning: Missing checkpoint step, start time, or end time in"
          " write times list."
      )
      continue

    if step not in write_time_dict:
      write_time_dict[step] = {"start_times": [], "end_times": []}

    write_time_dict[step]["start_times"].append(start_time)
    write_time_dict[step]["end_times"].append(end_time)

  write_duration_per_step = []
  for step, times in write_time_dict.items():
    write_duration_per_step.append(
        max(times["end_times"]) - min(times["start_times"])
    )

  if not write_duration_per_step:
    print(
        "Warning: Write time list is empty, cannnot process checkpoint"
        " write time results."
    )
    return

  print(f"min checkpoint write duration: {min(write_duration_per_step)}s")
  print(f"max checkpoint write duration: {max(write_duration_per_step)}s")
  print(
      "average checkpoint write duration:"
      f" {statistics.mean(write_duration_per_step)}s"
  )
  print(
      "checkpoint write time standard deviation:"
      f" {statistics.stdev(write_duration_per_step)}"
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Process checkpointing metrics from the logs."
  )
  parser.add_argument(
      "--gcs_logs_path",
      required=True,
      help=(
          "The path to the NeMo logs in a GCS bucket"
      ),
  )

  args = parser.parse_args()

  generate_warnings = os.getenv("GENERATE_LOG_WARNINGS", "False").lower() == "true"

  process_metrics_from_logs(args.gcs_logs_path)