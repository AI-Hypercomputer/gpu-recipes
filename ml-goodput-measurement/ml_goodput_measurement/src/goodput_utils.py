"""Goodput Utility Classes and Helpers."""

import dataclasses
import datetime
import enum
import logging
import math
from typing import Any, Optional

import numpy as np
import requests
from scipy import stats
from urllib3.util import retry


Retry = retry.Retry
_TIME_ENTRY = 'time'
_METADATA_SERVER_URL = 'http://metadata.google.internal/computeMetadata/v1/'
_METADATA_HEADERS = {'Metadata-Flavor': 'Google'}

MACHINE_TYPE_TO_ACCELERATOR_TYPE_MAPPING = {
    'ct6e': 'TPU-v6e',
    'ct5p': 'TPU-v5p',
    'ct5lp': 'TPU-v5e',
    'ct5l': 'TPU-v5e',
    'ct4p': 'TPU-v4p',
    'ct3p': 'TPU-v3',
    'ct3': 'TPU-v3',
    'tpu-v2': 'TPU-v2',
    'tpu': 'TPU',
    'a3-edgegpu': 'NVIDIA-H100',
    'a3-highgpu': 'NVIDIA-H100',
    'a3-megagpu': 'NVIDIA-H100',
    'a3-ultragpu': 'NVIDIA-H200',
    'a2': 'NVIDIA-A100',
    'gpu': 'GPU',
}


@dataclasses.dataclass
class GCPOptions:
  project_id: Optional[str] = None
  location: Optional[str] = None
  replica_id: str = '0'
  acc_type: Optional[str] = None
  enable_gcp_goodput_metrics: bool = True
  enable_gcp_step_deviation_metrics: bool = True


# Productive time is not broken down by activities yet. As such, we only have
# one type of Goodput which contributes to the total productive time.
class GoodputType(enum.Enum):
  """The type of Goodput."""

  TOTAL = 1


class BadputType(enum.Enum):
  """The type of Badput."""

  TPU_INITIALIZATION = 1
  TRAINING_PREP = 2
  PROGRAM_STARTUP = 3
  DATA_LOADING_SYNC = 4
  DATA_LOADING_ASYNC = 5
  UNPRODUCTIVE_CHECKPOINT_SAVE_TIME = 6
  UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME = 7
  WASTED_PROGRESS_FROM_DISRUPTION = 8
  CUSTOM_BADPUT_EVENTS = 9
  OTHER = 10


ACTIVITY_EXCLUSION_LIST = [
    # DATA_LOADING_ASYNC is not a non-productive activity as it is not
    # blocking. Hence, we exclude it from calculating Goodput.
    'DATA_LOADING_ASYNC',
]


class GoodputInfo:
  """Goodput Information."""

  def __init__(
      self,
      total_productive_time: float = 0.0,
      total_elapsed_time_since_start: float = 0.0,
      total_unproductive_time: Optional[dict[BadputType, float]] = None,
      last_recorded_step: int = 0,
      last_updated_timestamp: datetime.datetime = datetime.datetime.now(
          datetime.timezone.utc
      ),
  ):
    self.total_productive_time = total_productive_time
    self.total_elapsed_time_since_start = total_elapsed_time_since_start

    # We cannot use {} as the default argument directly because it's a mutable
    # default argument.  Mutable default arguments are shared between all
    # instances of the class.  If one instance modifies the default
    # dictionary, it will affect all other instances.  Instead, we use
    # None as a sentinel value and create a new dictionary inside the
    # __init__ method if no dictionary is provided. This ensures each
    # instance gets its own dictionary.
    self.total_unproductive_time = (
        total_unproductive_time or {}
    )
    self.last_recorded_step = last_recorded_step
    self.last_updated_timestamp = last_updated_timestamp


class StepInfo:
  """Step Information."""

  def __init__(
      self,
      ideal_step_time: float,
      step_deviations: dict[int, float],
  ):
    self.ideal_step_time = ideal_step_time
    self.step_deviations = step_deviations


def compute_ideal_step_time(step_times: list[float]) -> Optional[float]:
  """Helper function to compute the ideal step time."""
  # Filter out step times that may be less than 1 second.
  step_times = [step_time for step_time in step_times if step_time >= 1.0]
  if not step_times:
    return None
  # Compute the median absolute deviation (MAD) and median of the step times
  mad = stats.median_abs_deviation(step_times)
  med = np.median(step_times)

  # Normalize the step times to the median + 3 * MAD.
  normal_step_times = [
      step_time for step_time in step_times if step_time <= (med + mad * 3)
  ]
  return np.mean(normal_step_times) if normal_step_times else None


def get_anomalous_and_normal_step_times(
    step_times: list[Any],
) -> tuple[list[Any], list[Any]]:
  """Helper function to get anomalous and normal step times."""
  mad = stats.median_abs_deviation(step_times)
  med = np.median(step_times)

  anomalous_step_times = []
  normal_step_times = []
  for step_time in step_times:
    if step_time > (med + mad * 3):
      anomalous_step_times.append(step_time)
    else:
      normal_step_times.append(step_time)

  return anomalous_step_times, normal_step_times


def get_extra_time_from_anomalous_steps(step_times: list[Any]) -> float:
  anomalous_step_times, normal_step_times = get_anomalous_and_normal_step_times(
      step_times
  )
  normal_step_mean = np.mean(normal_step_times)
  return sum(anomalous_step_times) - (
      len(anomalous_step_times) * normal_step_mean
  )


def get_timestamp_from_log_entry(
    entry: dict[str, Any],
) -> Optional[datetime.datetime]:
  """Helper function to get the timestamp from a log entry."""
  timestamp_posix_time = [
      entry_value
      for entry_label, entry_value in entry.items()
      if _TIME_ENTRY in entry_label
  ]
  if timestamp_posix_time:
    return datetime.datetime.fromtimestamp(
        timestamp_posix_time[0], datetime.timezone.utc
    )
  return None


def get_gcp_metadata(category: str, attribute: str, timeout=5, retries=3):
  """Fetch the specified attribute from GCP metadata server.

  Args:
    category (str): The high-level metadata category (ex: 'instance',
      'project').
    attribute (str): The attribute to fetch under this category (ex: 'id',
      'zone').
    timeout (int): Timeout for the request in seconds.
    retries (int): Number of retry attempts for transient failures.

  Returns:
    str: The metadata value as a string, or None if the request fails.
  """
  target_url = f'{_METADATA_SERVER_URL}{category}/{attribute}'

  session = requests.Session()
  retry_strategy = Retry(
      total=retries,
      backoff_factor=0.5,
      # Retry on the following status codes
      status_forcelist=[429, 500, 502, 503, 504],
  )
  adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
  session.mount('http://', adapter)

  try:
    response = session.get(
        target_url, headers=_METADATA_HEADERS, timeout=timeout
    )
    response.raise_for_status()
    return response.text
  except requests.exceptions.RequestException as e:
    logging.warning(
        'Failed to retrieve metadata for %s/%s: %s', category, attribute, e
    )
    return None


def get_gcp_project_id():
  """Returns the project id of the current GCP project."""
  return get_gcp_metadata('project', 'project-id')


def get_node_zone():
  """Returns the zone of the GCE instance."""
  zone_path = get_gcp_metadata('instance', 'zone')
  # example zone_path: "projects/123456789/zones/us-central1-a"
  return zone_path.rsplit('/', 1)[-1] if zone_path else None


def get_accelerator_type():
  """Retrieves the accelerator type from GCP metadata.

  For GKE TPU VMs, it extracts the type from the 'machine-type' metadata.

  Returns:
    str: The accelerator type, or 'UNKNOWN' if not found.
  """
  machine_type_url = get_gcp_metadata('instance', 'machine-type')
  # example machine_type_url: "projects/123456789/machineTypes/a3-highgpu-8g"
  machine_type_name = (
      machine_type_url.split('/')[-1] if machine_type_url else None
  )

  if not machine_type_name:
    return 'UNKNOWN'

  for (
      prefix,
      accelerator_type,
  ) in MACHINE_TYPE_TO_ACCELERATOR_TYPE_MAPPING.items():
    if prefix.lower() in machine_type_name.lower():
      return accelerator_type

  return 'UNKNOWN'
