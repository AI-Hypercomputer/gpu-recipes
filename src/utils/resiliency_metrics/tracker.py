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

"""Training Job Goodput Tracker

This script tracks ML training job goodput by monitoring and logging key events
in the training pipeline using Google Cloud Logging.
"""

import argparse
import datetime
import json
import logging
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple
from google import auth as google_auth
from google.cloud import logging as gcloud_logging


class GoodputLogger:
  """Tracks training job goodput using Google Cloud Logging."""

  def __init__(self):
    """Initialize the goodput tracker."""
    self.job_name = "training-job"
    if os.getenv("JOB_IDENTIFIER") is not None:
      self.job_name = os.getenv("JOB_IDENTIFIER")
    self.log_name = f"{self.job_name}-goodput"
    self.use_gcloud_logging = True

    if os.getenv("GOODPUT_USE_FILE_TRACKING") is not None:
      self.use_gcloud_logging = False
      local_log_path = f"{self.log_name}.log"
      self.logger = logging.getLogger(self.log_name)
      while self.logger.handlers:
        self.logger.removeHandler(self.logger.handlers[0])
      self.logger.setLevel(logging.INFO)
      if os.getenv("RANK") == "0":
        print(f"Logging goodput events to {local_log_path}")

      # Add file handler
      file_handler = logging.FileHandler(local_log_path)
      file_handler.setFormatter(logging.Formatter("%(message)s"))
      self.logger.addHandler(file_handler)

    # Initialize logging client
    if self.use_gcloud_logging:
      _, project_id = google_auth.default()
      self.logging_client = gcloud_logging.Client(project=project_id)
      self.logger = self.logging_client.logger(self.log_name)
      if os.getenv("RANK") == "0":
        print(
            "Logging goodput events to gcloud logging"
            f" 'logName=projects/{project_id}/logs/{self.log_name}'"
        )

  def log_event(self, event_type: str, **kwargs) -> None:
    """Log an event to Google Cloud Logging.

    Args:
        event_type: Type of event
        **kwargs: Additional event data
    """
    timestamp = datetime.datetime.now()
    event_data = {
        "timestamp": timestamp.isoformat(),
        "job_name": self.job_name,
        "event_type": event_type,
        **kwargs,
    }

    # Log to Google Cloud Logging
    if self.use_gcloud_logging:
      self.logger.log_struct(event_data, severity="INFO")
    else:
      self.logger.info(f"{json.dumps(event_data)}")


def get_parser():
  parser = argparse.ArgumentParser(description="Goodput Tracker CLI.")

  parser.add_argument(
      "--event-type",
      type=str,
      choices=[USER_SCHEDULED, USER_TERMINATED],
      help="Event type.",
      required=True,
  )
  return parser


def main():
  logging.basicConfig(level=logging.INFO)
  args = get_parser().parse_args()
  goodput_logger = GoodputLogger()
  goodput_logger.log_event(event_type=args.event_type)


if __name__ == "__main__":
  main()
