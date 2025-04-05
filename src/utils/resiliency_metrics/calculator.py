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

import argparse
import datetime
import json
import logging
import os
import sys
from typing import Dict, List, Optional
from dateutil import parser as date_parser
import pandas as pd
from tabulate import tabulate

# Attempt to import Google Cloud libraries; fall back if not available
try:
  from google.cloud import logging as gcloud_logging
  from google import auth as google_auth
except ImportError:
  gcloud_logging = None
  google_auth = None

from constant import (
    USER_SCHEDULED,
    USER_TERMINATED,
    JOB_STARTED,
    JOB_TERMINATED,
    CHECKPOINT_LOADED,
    CHECKPOINT_SAVED,
    EVENT_TYPE_ORDER,
)


def _event_sort_key(event):
  # Sort events first by timestamp and then by the defined event type order.
  return event.get("timestamp", ""), EVENT_TYPE_ORDER.get(
      event.get("event_type"), float("inf")
  )


class GoodputCalculator:
  """Calculates training job goodput by analyzing logged events."""

  def __init__(
      self,
      job_name: str,
      local_log_path: Optional[str] = None,
      verbose: bool = False,
      gcloud_logging_lookback_days: float = 7.0,
  ):
    """Initialize the goodput calculator.

    Args:
        job_name: Name of the training job.
        use_file_tracking: Whether to use local file tracking instead of GCloud
          logging.
        local_log_path: Path to the local log file (if file tracking is used).
        verbose: Whether to enable verbose logging.
    """
    self.job_name = job_name
    self.log_name = f"{job_name}-goodput"
    self.verbose = verbose
    self.local_log_path = local_log_path
    self.gcloud_logging_lookback_days = gcloud_logging_lookback_days

    # Set up logger
    self.logger = logging.getLogger("goodput_calculator")
    self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    if not self.logger.handlers:
      self.logger.addHandler(handler)

    # Initialize Google Cloud Logging if not using file tracking
    self.logging_client = None
    if self.local_log_path is None:
      try:
        _, project_id = google_auth.default()
        self.logging_client = gcloud_logging.Client(project=project_id)
        self.logger.info(
            f"Using Google Cloud Logging with project ID: {project_id}"
        )
      except Exception as e:
        self.logger.error(f"Failed to initialize Google Cloud Logging: {e}")
        raise ValueError("gcloud logging is not accessible")
    else:
      self.logger.info(
          f"Using file-based tracking with log file: {self.local_log_path}"
      )

  def load_events(self) -> List[Dict]:
    """Load events from Google Cloud Logging or a local file.

    Returns:
        List of event dictionaries.
    """
    events = []
    if self.local_log_path:
      try:
        with open(self.local_log_path, "r") as f:
          for line in f:
            try:
              event = json.loads(line.strip())
              events.append(event)
            except json.JSONDecodeError as e:
              self.logger.warning(
                  f"Failed to parse log line: {line.strip()}, error: {e}"
              )
      except FileNotFoundError:
        raise FileNotFoundError(f"Log file not found: {self.local_log_path}")

    elif self.logging_client is not None:
      try:
        # Calculate the timestamp for 7 days ago in UTC
        now = datetime.datetime.utcnow()
        seven_days_ago = now - datetime.timedelta(
            days=self.gcloud_logging_lookback_days
        )
        # Format the timestamp in ISO 8601 format with a "Z" suffix to indicate UTC time
        time_filter = seven_days_ago.isoformat("T") + "Z"
        filter_str = (
            "severity>=INFO AND "
            f'logName="projects/{self.logging_client.project}/logs/{self.log_name}"'
            " AND "
            f'timestamp>"{time_filter}"'
        )
        self.logger.info(f"Fetching logs with filter: {filter_str}")
        entries = self.logging_client.list_entries(
            filter_=filter_str, page_size=500
        )
        for entry in entries:
          if hasattr(entry, "payload") and entry.payload:
            events.append(entry.payload)
      except Exception as e:
        self.logger.error(f"Failed to fetch logs from Google Cloud: {e}")

    events.sort(key=lambda x: x.get("timestamp", ""))
    self.logger.info(f"Loaded {len(events)} events")
    return events

  @staticmethod
  def preprocess_events(
      events: List[Dict], job_name: Optional[str] = None
  ) -> List[Dict]:
    """Preprocess events to create a clear timeline with deduplication.

    Args:
        events: List of raw event dictionaries.
        job_name: Optional job name to filter events.

    Returns:
        List of preprocessed event dictionaries.
    """
    if not events:
      return []

    logger = logging.getLogger("goodput_calculator")
    if job_name:
      filtered_events = [e for e in events if e.get("job_name") == job_name]
    else:
      if "job_name" in events[0]:
        job_name = events[0]["job_name"]
        filtered_events = [e for e in events if e.get("job_name") == job_name]
      else:
        filtered_events = events

    if not filtered_events:
      logger.warning(f"No events found for job name: {job_name}")
      return []

    filtered_events.sort(key=_event_sort_key)

    if filtered_events[0]["event_type"] != USER_SCHEDULED:
      proxy_user_scheduled = filtered_events[0].copy()
      proxy_user_scheduled["event_type"] = USER_SCHEDULED
      proxy_user_scheduled["is_proxy"] = True
      filtered_events.append(proxy_user_scheduled)

    filtered_events.sort(key=_event_sort_key)

    dedup_events = []
    for event in filtered_events:
      # For multiple JOB_STARTED or JOB_TERMINATED events in a row, keep only the last one.
      if (
          event["event_type"] == JOB_STARTED
          and dedup_events
          and dedup_events[-1]["event_type"] == JOB_STARTED
      ):
        dedup_events.pop()
      if (
          event["event_type"] == JOB_TERMINATED
          and dedup_events
          and dedup_events[-1]["event_type"] == JOB_TERMINATED
      ):
        dedup_events.pop()
      dedup_events.append(event)

    return dedup_events

  def calculate_goodput(
      self, events: List[Dict], reference_step_time: float = None
  ) -> Dict:
    """Calculate goodput metrics from event data.

    Args:
        events: List of (preprocessed) event dictionaries.
        reference_step_time: Time per step (in seconds) for computing effective
          time.

    Returns:
        Dictionary containing goodput metrics.
    """
    if not events:
      return {
          "error": "No events found",
          "goodput_percentage": 0,
          "total_runtime": 0,
          "useful_runtime": 0,
      }

    # Ensure events are preprocessed.
    if not any("is_proxy" in event for event in events):
      job_name = events[0].get("job_name") if events else None
      events = GoodputCalculator.preprocess_events(events, job_name)

    metrics = {
        "total_events": len(events),
        "job_started_count": 0,
        "checkpoints_loaded": 0,
        "checkpoints_saved": 0,
        "total_runtime_seconds": 0,
        "useful_runtime_seconds": 0,
        "effective_computation_time": 0,
        "goodput_percentage": 0,
        "job_intervals": [],
        "checkpoint_intervals": [],
    }
    if events:
      start_time = date_parser.parse(events[0].get("timestamp"))
      end_time = date_parser.parse(events[-1].get("timestamp"))
      metrics["total_runtime_seconds"] = (end_time - start_time).total_seconds()

    user_scheduled_time = None
    user_terminated_time = None
    job_start_time = None
    min_loaded_step = float("inf")
    max_saved_step = 0

    for event in events:
      event_type = event.get("event_type")
      timestamp = event.get("timestamp")
      try:
        event_time = date_parser.parse(timestamp)
        if event_type == JOB_STARTED:
          metrics["job_started_count"] += 1
          job_start_time = event_time
        elif event_type == CHECKPOINT_LOADED:
          step = event.get("step", 0)
          if step > 0:
            metrics["checkpoints_loaded"] += 1
          if step < min_loaded_step:
            min_loaded_step = step
        elif event_type == CHECKPOINT_SAVED:
          metrics["checkpoints_saved"] += 1
          step = event.get("step", 0)
          if step > max_saved_step:
            max_saved_step = step
      except Exception as e:
        self.logger.warning(f"Failed to process event: {event}, error: {e}")

    # Compute total time from the first event (USER_SCHEDULED) to the last event (JOB_TERMINATED)
    if user_scheduled_time and events:
      last_event_time = date_parser.parse(events[-1]["timestamp"])
      total_time = (last_event_time - user_scheduled_time).total_seconds()
      metrics["total_time_seconds"] = total_time

    # Compute effective computation time based on step difference
    if min_loaded_step != float("inf") and max_saved_step > 0:
      step_diff = max_saved_step - min_loaded_step
      if reference_step_time is not None and step_diff > 0:
        effective_time = step_diff * reference_step_time
        metrics["effective_computation_time"] = effective_time
        metrics["step_diff"] = step_diff
        metrics["min_loaded_step"] = min_loaded_step
        metrics["max_saved_step"] = max_saved_step
        if metrics.get("total_runtime_seconds", 0) > 0:
          metrics["goodput_percentage"] = (
              effective_time / metrics["total_runtime_seconds"]
          ) * 100

    return metrics

  def display_metrics(self, metrics: Dict) -> None:
    """Display the calculated goodput metrics in a human-readable format."""
    print("\n=== Goodput Analysis for Job:", self.job_name, "===\n")
    if "error" in metrics:
      print(f"Error: {metrics['error']}")
      return
    summary_data = [
        ["Total Events", metrics["total_events"]],
        ["Job Started Count", metrics["job_started_count"]],
        ["Checkpoints Loaded", metrics["checkpoints_loaded"]],
        ["Checkpoints Saved", metrics["checkpoints_saved"]],
    ]
    if (
        "total_runtime_seconds" in metrics
        and metrics["total_runtime_seconds"] > 0
    ):
      summary_data.append([
          "Total Runtime (hours)",
          round(metrics["total_runtime_seconds"] / 3600, 2),
      ])
    if "step_diff" in metrics:
      summary_data.append(["Min Loaded Step", metrics["min_loaded_step"]])
      summary_data.append(["Max Saved Step", metrics["max_saved_step"]])
      summary_data.append(["Step Difference", metrics["step_diff"]])
    if (
        "effective_computation_time" in metrics
        and metrics["effective_computation_time"] > 0
    ):
      summary_data.append([
          "Effective Computation Time (hours)",
          round(metrics["effective_computation_time"] / 3600, 2),
      ])
    summary_data.append(
        ["Goodput Percentage", f"{metrics['goodput_percentage']:.2f}%"]
    )
    print(tabulate(summary_data, headers=["Metric", "Value"], tablefmt="grid"))
    if (
        "checkpoint_intervals" in metrics
        and metrics["checkpoint_intervals"]
        and self.verbose
    ):
      print("\nCheckpoint Intervals:")
      cp_df = pd.DataFrame(metrics["checkpoint_intervals"])
      cp_df["duration_minutes"] = cp_df["duration_seconds"] / 60
      print(
          tabulate(
              cp_df[["start", "end", "duration_minutes"]],
              headers=["Start Time", "End Time", "Duration (minutes)"],
              tablefmt="grid",
          )
      )

  def export_metrics(self, metrics: Dict, output_file: str) -> None:
    """Export metrics to a JSON file."""
    try:
      with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
      print(f"\nMetrics exported to: {output_file}")
    except Exception as e:
      print(f"Failed to export metrics: {e}")


def get_parser():
  parser = argparse.ArgumentParser(
      description="Goodput Calculator for ML Training Jobs"
  )
  parser.add_argument(
      "--job-name", type=str, help="Name of the training job", required=True
  )
  parser.add_argument("--log-file", type=str, help="Path to local log file")
  parser.add_argument(
      "--export", type=str, help="Export metrics to specified JSON file"
  )
  parser.add_argument(
      "--reference-step-time",
      type=float,
      help="Reference time for a single step in seconds",
      default=None,
  )
  parser.add_argument(
      "--gcloud-logging-lookback-days",
      type=float,
      help="Number of days to lookback to find event logs.",
      default=7.0,
  )
  parser.add_argument(
      "--verbose", action="store_true", help="Enable verbose output"
  )
  return parser

def main():
  args = get_parser().parse_args()

  calculator = GoodputCalculator(
      job_name=args.job_name,
      local_log_path=args.log_file,
      gcloud_logging_lookback_days=args.gcloud_logging_lookback_days,
      verbose=args.verbose,
  )

  events = calculator.load_events()
  preprocessed_events = calculator.preprocess_events(events, args.job_name)
  metrics = calculator.calculate_goodput(
      preprocessed_events, args.reference_step_time
  )
  calculator.display_metrics(metrics)

  if args.export:
    calculator.export_metrics(metrics, args.export)

if __name__ == "__main__":
  main()
