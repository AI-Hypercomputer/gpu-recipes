"""Goodput package API implementations.

This file contains all the core implementations of the ml_goodput_measurement
library for users to measure and monitor Goodput, Badput and Step Time
Deviation.
"""

import datetime
import logging
import threading
from typing import Any, Optional, Union

from cloud_goodput.ml_goodput_measurement.src import checkpoint_badput_calculator
from cloud_goodput.ml_goodput_measurement.src import goodput_cache
from cloud_goodput.ml_goodput_measurement.src import goodput_utils


get_timestamp_from_log_entry = goodput_utils.get_timestamp_from_log_entry
get_extra_time_from_anomalous_steps = (
    goodput_utils.get_extra_time_from_anomalous_steps
)
compute_ideal_step_time = goodput_utils.compute_ideal_step_time

BadputType = goodput_utils.BadputType
CheckpointLoggerOptions = checkpoint_badput_calculator.CheckpointLoggerOptions
CheckpointBadputCalculator = (
    checkpoint_badput_calculator.CheckpointBadputCalculator
)
GoodputType = goodput_utils.GoodputType
GoodputCache = goodput_cache.GoodputCache
GoodputInfo = goodput_utils.GoodputInfo
StepInfo = goodput_utils.StepInfo
# Data structure to store the type of unproductive time (BadputType) and the
# corresponding time in seconds. If the BadputType is CUSTOM_BADPUT_EVENTS, the
# value is a dictionary of user defined event type and the corresponding time
# in seconds.
UnproductiveTimeDict = dict[
    BadputType, Union[float, dict[str, float]]
]

_JOB_NAME = 'job_name'
_STEP_COUNT = 'step_count'
_STEP_START_TIME = 'step_start_time'
_JOB_START_TIME = 'job_start_time'
_JOB_END_TIME = 'job_end_time'
_TPU_INIT_START_TIME = 'tpu_init_start_time'
_TPU_INIT_END_TIME = 'tpu_init_end_time'
_TRAINING_PREPARATION_START_TIME = 'training_prep_start_time'
_TRAINING_PREPARATION_END_TIME = 'training_prep_end_time'
_DATA_LOADING_START_TIME = 'data_loading_start_time'
_DATA_LOADING_END_TIME = 'data_loading_end_time'
_CUSTOM_BADPUT_EVENT_TYPE = 'custom_badput_event_type'
_CUSTOM_BADPUT_EVENT_START_TIME = 'custom_badput_event_start_time'
_CUSTOM_BADPUT_EVENT_END_TIME = 'custom_badput_event_end_time'

_CLOUD_LOGGING_PAGE_SIZE = 1000000

logger = logging.getLogger(__name__)


class _CloudLogger:
  """A helper class for reading and writing to Cloud Logging.

  Attributes:
    job_name: Name of a specific job.
    logger: The Cloud Logging logger object.
    job_start_time: Start time of the job run.
  """

  def __init__(self, job_name: str, log_name: str):
    """_CloudLogger constructor.

    Args:
      job_name: Name of the job the _CloudLogger is for.
      log_name: Name of the log being written.
    """
    import google.cloud.logging  # pylint: disable=g-import-not-at-top

    self.job_name = job_name
    logging_client = google.cloud.logging.Client()
    self.logger = logging_client.logger(log_name)
    self.job_start_time = None

  def write_cloud_logging_entry(self, entry) -> None:
    """Writes an entry to the Cloud Logging logger at INFO level.

    Args:
      entry: JSON-serializable structured log dictionary.
    """
    if entry is None:
      return
    if entry[_JOB_NAME] == self.job_name:
      self.logger.log_struct(
          entry,
          severity='INFO',
      )

  def _get_filter_msg(
      self,
      start_time: Optional[datetime.datetime],
      end_time: Optional[datetime.datetime],
  ) -> str:
    """Gets the filter message for the Cloud Logging query."""
    filter_entries = [
        'severity=INFO',
        f'jsonPayload.job_name="{self.job_name}"',
    ]
    # Add a filter to bind an end-time to the query window.
    if end_time is None:
      end_time = datetime.datetime.now(datetime.timezone.utc)
    elif end_time.tzinfo is None:
      end_time = end_time.replace(tzinfo=datetime.timezone.utc)

    filter_entries.append(f'timestamp<="{end_time.isoformat()}"')

    # Add a filter to bind a start-time to the query window (if available).
    if start_time is None:
      if self.job_start_time is not None:
        start_time = self.job_start_time - datetime.timedelta(days=1)

    if start_time is not None:
      if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=datetime.timezone.utc)
      filter_entries.append(f'timestamp>"{start_time.isoformat()}"')
    return ' AND '.join(filter_entries)

  def _update_job_start_time(self, entries: list[Any]):
    if self.job_start_time:
      return
    for entry in entries:
      if _JOB_START_TIME in entry and self.job_start_time is None:
        self.job_start_time = datetime.datetime.fromtimestamp(
            entry[_JOB_START_TIME]
        )
        break

  def read_cloud_logging_entries(
      self,
      start_time: Optional[datetime.datetime] = None,
      end_time: Optional[datetime.datetime] = None,
  ):
    """Queries Cloud Logging entries for the specific job.

    Args:
      start_time: The start time of the query window.
      end_time: The end time of the query window.

    Returns:
      Filtered entries in ascending order of timestamp.
    """
    import google.cloud.logging  # pylint: disable=g-import-not-at-top

    entries = self.logger.list_entries(
        filter_=self._get_filter_msg(start_time, end_time),
        order_by=google.cloud.logging.ASCENDING,
        page_size=_CLOUD_LOGGING_PAGE_SIZE,
    )
    entry_payload = [entry.payload for entry in entries]
    self._update_job_start_time(entry_payload)
    return entry_payload


class GoodputRecorder:
  """The Goodput recorder class, responsible for recording Goodput metrics from the user application.

  Attributes:
    job_name: Name of the job the GoodputRecorder is for.
  """

  def __init__(
      self,
      job_name: str,
      logger_name: str,
      logging_enabled=False,
      cloud_logger: Optional[_CloudLogger] = None,
  ):
    """GoodputRecorder constructor.

    Args:
      job_name: Name of the job the GoodputRecorder is for.
      logger_name: The name of the Cloud Logging logger object that the
        application wants logs to be written to and read from.
      logging_enabled: A boolean value to indicate whether the current process
        should send logs to Cloud Logging or not. The application should set
        this value to True if the Recorder is being called from TPU worker 0 and
        the application's configurations request Goodput logging.
      cloud_logger: Should never be passed directly by the user.
    """
    self.job_name = job_name
    # If logging is disabled for this process, do not create a _cloud_logger
    # object and exit early if any record record_* API is called.
    if not logging_enabled:
      self._cloud_logger = None
      logging.info('Logging is disabled for this process.')
      return

    if cloud_logger is not None:
      self._cloud_logger = cloud_logger
    else:
      self._cloud_logger = _CloudLogger(job_name, logger_name)

  def record_step_start_time(
      self, step: int, start_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log an individual step's start time.

    Args:
      step: The count of the step that timing information is recorded for.
      start_time: Optional backfill start time of the training step. If
        provided, it has to be in UTC time.
    """
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.now(datetime.timezone.utc)

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _STEP_COUNT: int(step),
        _STEP_START_TIME: start_time.timestamp(),
    })

  def record_checkpoint_progress(self, step, checkpoint_start_time):
    """Main recorder function to log information on a successful checkpoint.

    This method is intended to log the progress for a checkpoint (last step
    included in the checkpoint) and when the checkpoint starts. This information
    will be retrieved in the future to determine whether training progress from
    a completed step contributes to Goodput or wasted progress Badput.

    Args:
      step: The step count of the last step included in the saved checkpoint.
      checkpoint_start_time: Timestamp at which the checkpoint containing
        progress upto "step" starts to save.
    """
    pass

  def record_job_start_time(
      self, start_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log a job's start time.

    Args:
      start_time: Optional backfill start time of the job. If provided, it has
        to be in UTC time.
    """
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.now(datetime.timezone.utc)

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _JOB_START_TIME: start_time.timestamp(),
    })

  def record_job_end_time(self, end_time: Optional[datetime.datetime] = None):
    """Main recorder function to log a job's end time.

    Args:
      end_time: Optional backfull end time of the job. If provided, it has to be
        in UTC time.
    """
    if self._cloud_logger is None:
      return
    if end_time is None:
      end_time = datetime.datetime.now(datetime.timezone.utc)

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _JOB_END_TIME: end_time.timestamp(),
    })

  def record_tpu_init_start_time(
      self, start_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log the start time for TPU initialization.

    Note: TPU initialization may include the time spent in completing
    jax.devices() which is responsible for device scanning and Slice Builder
    initialization.

    Args:
      start_time: Start time of TPU initialization.
    """
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.now(datetime.timezone.utc)

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _TPU_INIT_START_TIME: start_time.timestamp(),
    })

  def record_tpu_init_end_time(
      self, end_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log the end time for TPU initialization.

    Args:
      end_time: End time of TPU initialization.
    """
    if self._cloud_logger is None:
      return
    if end_time is None:
      end_time = datetime.datetime.now(datetime.timezone.utc)

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _TPU_INIT_END_TIME: end_time.timestamp(),
    })

  def record_training_preparation_start_time(
      self, start_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log the start time of training preparation before starting a training loop.

    Note: Training preparation may include the time spent in creation of
    checkpoint managers, checkpoint loading, running mesh and model optimizers
    etc.

    Args:
      start_time: Start time of training preparation.
    """
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.now(datetime.timezone.utc)

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _TRAINING_PREPARATION_START_TIME: start_time.timestamp(),
    })

  def record_training_preparation_end_time(
      self, end_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log the end time of training preparation before starting a training loop.

    Args:
      end_time: End time of training preparation.
    """
    if self._cloud_logger is None:
      return
    if end_time is None:
      end_time = datetime.datetime.now(datetime.timezone.utc)

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _TRAINING_PREPARATION_END_TIME: end_time.timestamp(),
    })

  def record_data_loading_start_time(
      self, start_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log the start time of training's data loading.

    Args:
      start_time: Start time of data loading.
    """
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.now(datetime.timezone.utc)

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _DATA_LOADING_START_TIME: start_time.timestamp(),
    })

  def record_data_loading_end_time(
      self, end_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log the end time of training's data loading.

    Args:
      end_time: End time of data loading.
    """
    if self._cloud_logger is None:
      return
    if end_time is None:
      end_time = datetime.datetime.now(datetime.timezone.utc)

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _DATA_LOADING_END_TIME: end_time.timestamp(),
    })

  def record_custom_badput_event_start_time(
      self,
      start_time: Optional[datetime.datetime] = None,
      custom_badput_event_type: str = 'unknown',
  ):
    """Main recorder function to log the start time of a custom badput event.

    Use this function to record the start time of a custom badput event that
    occurs inside the training loop and utilizes the accelerator resources,
    and blocks training.

    For example, use this API to record the start time of the evaluation
    loop or an SDC check if the the event blocks the training loop.

    Args:
      start_time: Start time of the custom badput event.
      custom_badput_event_type: Type of the custom badput event.
    """
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.now(datetime.timezone.utc)

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _CUSTOM_BADPUT_EVENT_TYPE: custom_badput_event_type,
        _CUSTOM_BADPUT_EVENT_START_TIME: start_time.timestamp(),
    })

  def record_custom_badput_event_end_time(
      self,
      end_time: Optional[datetime.datetime] = None,
      custom_badput_event_type: str = 'unknown',
  ):
    """Main recorder function to log the end time of a custom badput event.

    Args:
      end_time: End time of the custom badput event.
      custom_badput_event_type: Type of the custom badput event.
    """
    if self._cloud_logger is None:
      return
    if end_time is None:
      end_time = datetime.datetime.now(datetime.timezone.utc)

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _CUSTOM_BADPUT_EVENT_TYPE: custom_badput_event_type,
        _CUSTOM_BADPUT_EVENT_END_TIME: end_time.timestamp(),
    })


class GoodputCalculator:
  """The Goodput calculator class, responsible for querying necessary information and computing Goodput metrics to return to the user application.

  Attributes:
    job_name: Name of the job the GoodputCalculator is for.
    using_pathways: Whether or not the job uses Pathways.
  """

  def __init__(
      self,
      job_name: str,
      logger_name: str,
      cloud_logger: Optional[_CloudLogger] = None,
      using_pathways: bool = False,
  ):
    """GoodputCalculator constructor.

    Args:
      job_name: Name of the job the GoodputCalculator is for.
      logger_name: Name of the log being written.
      cloud_logger: Should never be passed directly by the user.
      using_pathways: Whether or not the job uses Pathways.
    """
    self.job_name = job_name
    self.using_pathways = using_pathways
    if cloud_logger is not None:
      self._cloud_logger = cloud_logger
    else:
      self._cloud_logger = _CloudLogger(job_name, logger_name)
    self._current_entries = []
    self._goodput_cache = GoodputCache()
    self._goodput_cache_lock = threading.Lock()
    self._interval_entries = []
    self._interval_start_time = None
    self._interval_end_time = None
    self._number_of_interruptions = 0
    self._gcm_last_recorded_timestamp = None
    self._last_disruption_time = None
    self._last_disrupted_step = None

  def _get_total_productive_and_unproductive_time(
      self, new_entries: list[dict[str, Any]]
  ) -> tuple[float, UnproductiveTimeDict, int]:
    """Helper function to compute the total productive and unproductive time.

    Args:
      new_entries: A list of new log entries to process.

    Returns:
      A tuple of:
        - total productive training time
        - total unproductive time
        - last recorded step
    """
    # If no new entries are present, return last computed values.
    if not new_entries:
      cached_values = self._get_cached_productive_and_unproductive_time()
      if cached_values is not None:
        return cached_values

    return self._get_current_productive_and_unproductive_time()

  def _get_cached_productive_and_unproductive_time(
      self,
  ) -> tuple[float, UnproductiveTimeDict, int] | None:
    """Helper function to retrieve the cached productive training time and unproductive time."""
    goodput_info = self._goodput_cache.get_goodput_info()
    if not self._goodput_cache.is_cache_empty() and goodput_info is not None:
      return (
          goodput_info.total_productive_time,
          goodput_info.total_unproductive_time,
          goodput_info.last_recorded_step,
      )
    return None

  def _accumulate_unproductive_time(
      self,
      segment_unproductive_time: UnproductiveTimeDict,
      total_unproductive_time: UnproductiveTimeDict,
  ):
    """Helper function to accumulate the segment unproductive time.

    Args:
      segment_unproductive_time: A dictionary of unproductive time for a
        segment.
      total_unproductive_time: A dictionary of total unproductive time.

    Returns:
      None. The function updates the total_unproductive_time dictionary.
    """

    for badput_type, unproductive_value in segment_unproductive_time.items():
      if isinstance(unproductive_value, dict):
        if badput_type not in total_unproductive_time:
          total_unproductive_time[badput_type] = dict(unproductive_value)
        else:
          existing_value = total_unproductive_time[badput_type]
          if isinstance(existing_value, dict):
            for sub_type, sub_value in unproductive_value.items():
              existing_value[sub_type] = (
                  existing_value.get(sub_type, 0.0) + sub_value
              )
      else:
        if badput_type in total_unproductive_time:
          existing_value = total_unproductive_time[badput_type]
          if isinstance(existing_value, float):
            total_unproductive_time[badput_type] = (
                existing_value + unproductive_value
            )
        else:
          total_unproductive_time[badput_type] = unproductive_value

  def _get_current_productive_and_unproductive_time(
      self, interval_query: Optional[bool] = False
  ) -> tuple[
      float,
      UnproductiveTimeDict,
      int,
  ]:
    """Helper function to compute the current productive training time, unproductive time and the last step recorded till now.

    Args:
      interval_query: A boolean value to indicate whether the current query is
        for an interval or not.

    Returns:
      A tuple of the productive training time, the unproductive time
      (dict of BadputType and unproductive time) and the last step recorded till
      now based on the latest entries retrieved from Cloud Logging.
    """
    def _extract_custom_sync_intervals(
        entries: list[dict[str, Any]],
    ) -> list[tuple[float, float, str]]:
      """Extracts custom badput intervals from Cloud Logging entries.

      This helperfunction scans through a list of Cloud Logging entries to find
      custom
      badput start and end times, pairing them into intervals.

      Args:
          entries: A list of dictionaries representing Cloud Logging entries.
            Each entry may contain keys indicating the start or end of a custom
            badput event.

      Returns:
          A list of tuples, where each tuple consists of:
              - start_time (float): The timestamp when the sync event started.
              - end_time (float): The timestamp when the sync event ended.
              - sync_type (str): The type of custom sync
              event.
      """
      intervals = []
      active_syncs = {}

      for entry in entries:
        if _CUSTOM_BADPUT_EVENT_START_TIME in entry:
          sync_type = entry[_CUSTOM_BADPUT_EVENT_TYPE].upper()
          active_syncs[sync_type] = entry[_CUSTOM_BADPUT_EVENT_START_TIME]
        elif _CUSTOM_BADPUT_EVENT_END_TIME in entry:
          sync_type = entry[_CUSTOM_BADPUT_EVENT_TYPE].upper()
          if sync_type in active_syncs:
            start_time = active_syncs.pop(sync_type)
            end_time = entry[_CUSTOM_BADPUT_EVENT_END_TIME]
            if start_time < end_time:
              intervals.append((start_time, end_time, sync_type))

      return intervals

    def _compute_adjusted_segment_productive_and_unproductive_time(
        step_items: list[tuple[int, float]],
        curr_step: int,
        min_step: int,
        custom_sync_intervals: list[
            tuple[float, float, str]
        ],
    ) -> tuple[
        float,
        float,
        list[float],
        float,
        dict[str, float],
        int,
    ]:
      """Computes adjusted productive and unproductive time for a segment of steps.

      This helper function calculates the total productive time, and the
      breakdown of time lost due to custom badput events, as well as
      wasted progress caused by disruptions.

      Args:
          step_items: A list of tuples, where each tuple contains a step number
            (int) and its start timestamp (float).
          curr_step: The current step number indicating the end of the segment.
          min_step: The minimum step number indicating the start of the segment.
          custom_sync_intervals: A list of tuples, where each tuple consists of:
            - start_time (float): Start timestamp of the sync event.
            - end_time (float): End timestamp of the sync event.
            - sync_type (str): The type of sync event.

      Returns:
          A tuple containing:
            - total_productive_time (float): Adjusted time excluding custom
                sync durations.
            - first_step_time (float): Adjusted duration of the first step in
                the segment.
            - step_times (list[float]): List of adjusted times for steps in the
                segment excluding the first step.
            - wasted_progress (float): Total unproductive time due to possible
                disruptions.
            - custom_sync_breakdown (dict[str, float]):
                Breakdown of time spent in each custom sync type.
            - steps_in_segment (int): Total number of steps considered in the
                segment.
      """
      total_productive_time = 0.0
      first_step_time = 0.0
      step_times = []
      wasted_progress = 0.0
      custom_sync_breakdown: dict[str, float] = {}

      steps_in_segment = 0

      for i in range(1, len(step_items)):
        prev_step, prev_time = step_items[i - 1]
        curr_step_num, curr_time = step_items[i]

        raw_delta = curr_time - prev_time
        if curr_step_num <= curr_step:
          if curr_step_num - 1 != prev_step:
            continue

          custom_sync_in_interval = 0.0
          for sync_start, sync_end, sync_type in custom_sync_intervals:
            if prev_time <= sync_start and sync_end <= curr_time:
              sync_duration = sync_end - sync_start
              custom_sync_in_interval += sync_duration
              custom_sync_breakdown[sync_type] = (
                  custom_sync_breakdown.get(sync_type, 0.0) + sync_duration
              )

          adjusted_delta = max(0.0, raw_delta - custom_sync_in_interval)
          total_productive_time += adjusted_delta

          if prev_step == min_step:
            first_step_time = adjusted_delta
          else:
            step_times.append(adjusted_delta)

          steps_in_segment += 1

        else:
          # These steps are after curr_step, they are lost due to disruption.
          wasted_progress += raw_delta

      return (
          total_productive_time,
          first_step_time,
          step_times,
          wasted_progress,
          custom_sync_breakdown,
          steps_in_segment,
      )

    def _compute_segment_final_metrics(
        adjusted_productive_time: float,
        first_step_time: float,
        step_times: list[float],
        wasted_progress: float,
        custom_sync_breakdown: dict[str, float],
    ) -> tuple[
        float,
        UnproductiveTimeDict,
    ]:
      """Computes final metrics for a segment, separating productive and unproductive time.

      This function takes adjusted productive time and calculates additional
      badput sources such as program startup and wasted progress due to
      disruptions. It returns the final productive time and a breakdown of all
      unproductive time sources.

      Args:
          adjusted_productive_time: Total productive time for the segment
          first_step_time: Productive time for the first step in the segment.
          step_times: Productive times for non-first steps in the segment.
          wasted_progress: Total time lost due to step discontinuities.
          custom_sync_breakdown: A dictionary mapping each custom sync type to
            the total badput time it accounted for during the segment.

      Returns:
          A tuple containing:
              - final_productive_time (float)
              - total_segment_unproductive_time (dict)
      """
      steps_in_segment = len(step_times) + 1  # Including first step

      if steps_in_segment == 1:
        return first_step_time, {
            BadputType.WASTED_PROGRESS_FROM_DISRUPTION: wasted_progress,
            BadputType.CUSTOM_BADPUT_EVENTS: custom_sync_breakdown,
            BadputType.PROGRAM_STARTUP: 0.0,
        }

      non_first_steps = steps_in_segment - 1
      non_first_total_time = adjusted_productive_time - first_step_time
      average_step_time = (
          non_first_total_time / non_first_steps if non_first_steps > 0 else 0.0
      )
      first_step_extra_time = max(0.0, first_step_time - average_step_time)
      final_productive_time = adjusted_productive_time - first_step_extra_time

      total_segment_unproductive_time = {
          BadputType.PROGRAM_STARTUP: first_step_extra_time,
          BadputType.WASTED_PROGRESS_FROM_DISRUPTION: wasted_progress,
          BadputType.CUSTOM_BADPUT_EVENTS: custom_sync_breakdown,
      }

      return final_productive_time, total_segment_unproductive_time

    def _get_segment_productive_and_unproductive_time(
        step_start_data: dict[int, float],
        curr_step: int,
        entries_to_process: list[Any],
    ) -> tuple[
        float,
        UnproductiveTimeDict,
    ]:
      if curr_step == 0:
        return 0.0, {}

      step_items = list(step_start_data.items())
      min_step = min(step_start_data.keys())

      # Extract custom sync intervals
      custom_sync_intervals = _extract_custom_sync_intervals(entries_to_process)

      # Compute adjusted segmentproductive and unproductive times
      (
          total_productive_time,
          first_step_time,
          step_times,
          wasted_progress_from_disruption,
          custom_sync_breakdown,
          steps_in_segment,
      ) = _compute_adjusted_segment_productive_and_unproductive_time(
          step_items, curr_step, min_step, custom_sync_intervals
      )

      if steps_in_segment == 0:
        return 0.0, {
            BadputType.WASTED_PROGRESS_FROM_DISRUPTION: (
                wasted_progress_from_disruption
            )
        }

      # Compute adjusted averages and unproductive breakdown
      (
          final_adjusted_productive_time,
          total_segment_unproductive_time,
      ) = _compute_segment_final_metrics(
          total_productive_time,
          first_step_time,
          step_times,
          wasted_progress_from_disruption,
          custom_sync_breakdown,
      )

      return final_adjusted_productive_time, total_segment_unproductive_time

    # Build a deserialized dictionary from cloud logging entries to store step
    # start times. The dictionary maps from step count to start time and will be
    # used to each step's productive time by looking for its completion in the
    # next step's start.
    # Note in the instance where progress is lost due to a disruption and the
    # last successful checkpoint did not include all the steps, the last set of
    # records of the step information will be kept and the previous set will be
    # overwritten by design so as to correct for the the previously computed
    # additional time that was counted as productive but lost due to a
    # disruption.
    productive_training_time = 0.0
    total_unproductive_time = {}
    step_start_data = {}
    job_start_time = None
    job_end_time = None
    tpu_init_start_time = None
    training_prep_start_time = None
    data_loading_start_time = None
    tpu_initialization_badput = 0.0
    training_prep_badput = 0.0
    data_loading_badput = 0.0
    sync_data_loading = True
    current_sync_data_loading = None
    if interval_query:
      entries_to_process = self._interval_entries
    else:
      with self._goodput_cache_lock:
        entries_to_process = list(self._goodput_cache.get_cached_entries())

    self._number_of_interruptions = 0
    for payload in entries_to_process:
      if _JOB_START_TIME in payload:
        # Keep track of the latest start to compute badput due to disruption.
        job_start_time = payload[_JOB_START_TIME]
      if _STEP_START_TIME in payload:
        curr_step = int(payload[_STEP_COUNT])
        if curr_step not in step_start_data:
          step_start_data[curr_step] = payload[_STEP_START_TIME]
        else:
          # In this case, the job restarted from Step (curr_step). It means that
          # all progress till Step (curr_step - 1) has been preserved. So we
          # can get the productive time since the previous start/restart and
          # then clear the step_start_data dict.
          self._number_of_interruptions += 1
          self._last_disrupted_step = list(step_start_data.keys())[-1]
          self._last_disruption_time = step_start_data[
              self._last_disrupted_step
          ]

          # Compute segment productive and unproductive time.
          segment_productive_time, segment_unproductive_time = (
              _get_segment_productive_and_unproductive_time(
                  step_start_data, curr_step, entries_to_process
              )
          )
          # Accumulate the segment productive time.
          productive_training_time += segment_productive_time

          # When the job restarts, data loading is synchronous.
          sync_data_loading = True
          if current_sync_data_loading is not None:
            segment_unproductive_time[BadputType.DATA_LOADING_SYNC] = (
                segment_unproductive_time.get(BadputType.DATA_LOADING_SYNC, 0)
                + current_sync_data_loading
            )
            current_sync_data_loading = None

          # Since the current step has been recorded again, the progress
          # between the previously recorded curr_step and recently recorded
          # curr_step has been lost to a disruption and partially recovered
          # due to a checkpoint of curr_step - 1. Accumulate the lost time in
          # this segment as unproductive time.
          # Note this unproductive time is divided into two buckets:
          #   1. Wasted training progress after the last successfully
          #      checkpointed step and the disruption time until the job
          #      restarts.
          #   2. TPU re-init, training prep, data loading, program startup,
          #      checkpoint loading etc. after the job restarts and before
          #      training progress resumes.

          # The first bucket can be calculated as the time between the start
          # time of curr_step and the job restart time immediately prior.
          if (
              job_start_time is not None
              and self._last_disruption_time is not None
              and job_start_time > self._last_disruption_time
          ):
            # Add the additional time it took for the job to restart after last
            # interruption. These conditions are only met when the job is
            # restarted after a disruption.
            # TODO(dishaw): This is the infrastructure disruption Badput and can
            # go into a separate bucket.
            disruption_badput = job_start_time - self._last_disruption_time
            if (
                BadputType.WASTED_PROGRESS_FROM_DISRUPTION
                in segment_unproductive_time
            ):
              segment_unproductive_time[
                  BadputType.WASTED_PROGRESS_FROM_DISRUPTION
              ] += disruption_badput
            else:
              segment_unproductive_time[
                  BadputType.WASTED_PROGRESS_FROM_DISRUPTION
              ] = disruption_badput

          # The second bucket is individually computed either from recorded
          # logs (TPU initialization, training preparation, data loading) or
          # computed from the first step time after start or restart
          # (segment unproductive time). All unproductive time is accumulated
          # as we go.
          self._accumulate_unproductive_time(
              segment_unproductive_time, total_unproductive_time
          )
          step_start_data = {curr_step: payload[_STEP_START_TIME]}

      if _JOB_END_TIME in payload:
        # Locate the last instance of job's end time if the job has completed.
        job_end_time = payload[_JOB_END_TIME]

      # Compute badput due to TPU initialization.
      if _TPU_INIT_START_TIME in payload:
        tpu_init_start_time = payload[_TPU_INIT_START_TIME]
      elif _TPU_INIT_END_TIME in payload and tpu_init_start_time is not None:
        tpu_initialization_badput += (
            payload[_TPU_INIT_END_TIME] - tpu_init_start_time
        )
        tpu_init_start_time = None

      # Compute badput due to training preparation.
      elif _TRAINING_PREPARATION_START_TIME in payload:
        training_prep_start_time = payload[_TRAINING_PREPARATION_START_TIME]
      elif (
          _TRAINING_PREPARATION_END_TIME in payload
          and training_prep_start_time is not None
      ):
        training_prep_badput += (
            payload[_TRAINING_PREPARATION_END_TIME] - training_prep_start_time
        )
        training_prep_start_time = None

      # Compute badput due to data loading.
      elif _DATA_LOADING_START_TIME in payload:
        data_loading_start_time = payload[_DATA_LOADING_START_TIME]
      elif (
          _DATA_LOADING_END_TIME in payload
          and data_loading_start_time is not None
      ):
        data_loading_end_time = payload[_DATA_LOADING_END_TIME]
        current_sync_data_loading = (
            data_loading_end_time - data_loading_start_time
        )
        data_loading_badput += current_sync_data_loading
        if sync_data_loading:
          # When the job starts, data loading is synchronous.
          total_unproductive_time[BadputType.DATA_LOADING_SYNC] = (
              total_unproductive_time.get(BadputType.DATA_LOADING_SYNC, 0)
              + current_sync_data_loading
          )
          sync_data_loading = False
        data_loading_start_time = None

    # Compute unproductive time from checkpoint manager save and restore.
    checkpoint_logger_options = CheckpointLoggerOptions(use_goodput_logger=True)
    checkpoint_badput_calc = CheckpointBadputCalculator(
        checkpoint_logger_options
    )
    checkpoint_badput_calc.entries = entries_to_process
    checkpoint_manager_save_stats = (
        checkpoint_badput_calc.calculate_save_operation_checkpoint_manager_blocking_time()
    )
    checkpoint_manager_save_badput = (
        checkpoint_manager_save_stats.total_checkpoint_manager_blocking_time
    )
    checkpoint_manager_restore_stats = (
        checkpoint_badput_calc.calculate_restore_operation_checkpoint_manager_blocking_time()
    )
    checkpoint_manager_restore_badput = (
        checkpoint_manager_restore_stats.total_checkpoint_manager_time
    )

    # Populate some Badput buckets in total_unproductive_time.
    total_unproductive_time[BadputType.TPU_INITIALIZATION] = (
        tpu_initialization_badput
    )
    total_unproductive_time[BadputType.TRAINING_PREP] = training_prep_badput

    # Populate async data loading badput.
    async_data_loading_badput = (
        data_loading_badput
        - total_unproductive_time.get(BadputType.DATA_LOADING_SYNC, 0)
    )
    total_unproductive_time[BadputType.DATA_LOADING_ASYNC] = (
        async_data_loading_badput
    )

    # Populate checkpoint manager save and restore badput.
    total_unproductive_time[BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME] = (
        checkpoint_manager_save_badput
    )
    total_unproductive_time[BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME] = (
        checkpoint_manager_restore_badput
    )

    if not step_start_data:
      return 0.0, total_unproductive_time, 0

    last_step = max(list(step_start_data.keys()))
    segment_productive_time, segment_unproductive_time = (
        _get_segment_productive_and_unproductive_time(
            step_start_data, last_step, entries_to_process
        )
    )
    productive_training_time += segment_productive_time
    self._accumulate_unproductive_time(
        segment_unproductive_time, total_unproductive_time
    )

    # Only consider the last step productive if the job has completed.
    if job_end_time is not None:
      productive_training_time += job_end_time - step_start_data[last_step]

    # Remove blocking checkpoint manager save time from productive time.
    productive_training_time -= checkpoint_manager_save_badput

    # Return a tuple of the total productive training time, the total
    # unproductive time (dict of BadputType and unproductive time) and the last
    # step recorded.
    return productive_training_time, total_unproductive_time, last_step

  def _get_total_job_time(self, query_time: datetime.datetime) -> float:
    """Helper function to compute the current job runtime.

    Args:
      query_time: The time at which the query is being made.

    Returns:
      The job's total runtime computed based on the last retrieved logs.
    """
    # Find the job's original start time from the cache.
    start_time = self._goodput_cache.get_job_start_time()
    end_time = self._goodput_cache.get_job_end_time()
    if start_time:
      if not end_time:
        end_time = query_time
      return end_time.timestamp() - start_time.timestamp()

    # De-serealize job start and end times from cloud logging entries. These
    # will be used to compute total runtime of the job.
    job_start_time = None
    job_end_time = None
    with self._goodput_cache_lock:
      cached_entries = list(self._goodput_cache.get_cached_entries())
    for payload in cached_entries:
      # Locate the earliest timestamp recorded for the job's start.
      if _JOB_START_TIME in payload and job_start_time is None:
        job_start_time = payload[_JOB_START_TIME]
      # Locate the latest timestamp recorded for the job's end.
      if _JOB_END_TIME in payload:
        job_end_time = payload[_JOB_END_TIME]

    if job_start_time is not None:
      if job_end_time is not None:
        return job_end_time - job_start_time
      # If the job's end time is missing then job has not yet completed, use
      # current query time to compute total job time.
      return query_time.timestamp() - job_start_time
    # The the job's start time is missing so the total job time cannot be
    # calculated. Caller of this function should raise an error if this happens.
    return 0.0

  def _fetch_new_entries(self, query_time: datetime.datetime) -> list[Any]:
    """Thread-safe helper function to update and return new log entries."""
    with self._goodput_cache_lock:
      if not self._goodput_cache.is_cache_empty():
        last_entry_timestamp = self._goodput_cache.get_last_entry_timestamp()
        if query_time <= last_entry_timestamp:
          return []
        new_entries = self._cloud_logger.read_cloud_logging_entries(
            last_entry_timestamp, query_time
        )
      else:
        new_entries = self._cloud_logger.read_cloud_logging_entries()

      # Update the cache with the new log entries.
      self._goodput_cache.update_cached_entries(new_entries)
      return new_entries

  def _get_interval_log_entries(
      self, start_time: datetime.datetime, end_time: datetime.datetime
  ):
    """Helper function to get log entries from an interval window."""
    if start_time is None or end_time is None:
      raise ValueError(
          'Start and end times are required to get log entries from an interval'
          ' window.'
      )
    self._interval_entries = self._cloud_logger.read_cloud_logging_entries(  # type: ignore
        start_time, end_time
    )
    logging.info(
        'Inspecting interval entries between %s and %s', start_time, end_time
    )

    if not self._interval_entries:
      raise ValueError(
          'No log entries found within the interval window between %s and %s.'
          % (start_time, end_time)
      )

  def _sanitize_unproductive_times(
      self,
      unproductive_times: UnproductiveTimeDict,
      max_allowed: float,
  ) -> None:
    """Helper function to sanitize unproductive times."""
    for badput_type, value in unproductive_times.items():
      if isinstance(value, float):
        if value < 0.0 or value > max_allowed:
          logging.warning(
              'Unproductive time for %s could not be computed.', badput_type
          )
          unproductive_times[badput_type] = 0.0
      elif isinstance(value, dict):
        for sub_type, sub_value in value.items():
          if sub_value < 0.0 or sub_value > max_allowed:
            logging.warning(
                'Unproductive time for %s[%s] could not be computed.',
                badput_type,
                sub_type,
            )
            value[sub_type] = 0.0

  def _calculate_total_flat_unproductive_time(
      self,
      unproductive_time_dict: UnproductiveTimeDict,
  ) -> float:
    """Helper function to calculate total flat unproductive time."""
    total = 0.0
    for badput_type, value in unproductive_time_dict.items():
      if badput_type in {BadputType.DATA_LOADING_ASYNC, BadputType.OTHER}:
        continue
      if isinstance(value, float):
        total += value
      elif isinstance(value, dict):
        total += sum(value.values())
    return total

  def _compute_other_unproductive_time(
      self,
      total_job_time: float,
      productive_training_time: float,
      unproductive_time_dict: UnproductiveTimeDict,
  ) -> float:
    """Helper function to compute the "Unknown/Other" unproductive time."""
    other_unproductive_time = (
        total_job_time
        - productive_training_time
        - self._calculate_total_flat_unproductive_time(unproductive_time_dict)
    )
    return max(0.0, other_unproductive_time)

  def _get_total_job_time_from_interval(
      self, start_interval: datetime.datetime, end_interval: datetime.datetime
  ) -> float:
    """Helper function to compute the total job runtime from interval entries."""
    # Get the first and last entry's timestamps in the window
    first_entry_timestamp = get_timestamp_from_log_entry(
        self._interval_entries[0]
    )
    last_entry_timestamp = get_timestamp_from_log_entry(
        self._interval_entries[-1]
    )

    # Calculate effective start_time and end_time
    self._interval_start_time = (
        max(start_interval, first_entry_timestamp)
        if first_entry_timestamp
        else start_interval
    )
    self._interval_end_time = (
        min(end_interval, last_entry_timestamp)
        if last_entry_timestamp
        else end_interval
    )

    # Ensure start_time is not after end_time
    if self._interval_start_time >= self._interval_end_time:
      raise ValueError(
          'Start time is on or after end time, cannot compute total job time.'
      )

    return (
        self._interval_end_time.timestamp()
        - self._interval_start_time.timestamp()
    )

  def get_job_goodput(self, include_badput_breakdown=False) -> tuple[
      float,
      UnproductiveTimeDict,
      int,
  ]:
    """Method to get the cumulative Goodput and Badput breakdown of the job computed until now.

    If the application is interested in retrieving the overall Goodput of the
    job throughout its lifetime, this method provides the singular Goodput
    computation for the entire job.

    This method also returns the Badput breakdown of the job if
    `include_badput_breakdown` is set to True.

    Additionaly, this method returns the last step recorded for the job. This is
    primarily used for improving monitoring and observability of the job's
    overall Goodput as a function of number of executed steps.

    Args:
      include_badput_breakdown: Whether or not to return the badput breakdown.
        If False, returns {} for the badput breakdown.

    Returns:
      A tuple of the job's Goodput, optionally the Badput breakdown and the last
      step recorded for the job.

    Raises:
      ValueError if computed total job time is zero. In this case, Goodput
      cannot be computed.
      ValueError if productive training time is invalid.
    """
    query_time = datetime.datetime.now(datetime.timezone.utc)

    # Update the logs used to compute Goodput.
    new_entries = self._fetch_new_entries(query_time)

    total_job_time = self._get_total_job_time(query_time)
    # No calculations can be made if total job time is zero. This can happen if
    # logs for the job are not present, sent to an invalid location or contain
    # bad data. Raise a ValueError if this happens.
    if total_job_time == 0.0:
      raise ValueError(
          'Total job time is zero, Goodput cannot be calculated. Please fix the'
          ' logging entries.'
      )
    productive_training_time, total_unproductive_time, last_step = (
        self._get_total_productive_and_unproductive_time(new_entries)
    )
    if (
        productive_training_time < 0.0
        or productive_training_time > total_job_time
    ):
      raise ValueError(
          'Productive training time is invalid. Please fix the logging entries.'
      )

    # Sanitize the unproductive times.
    self._sanitize_unproductive_times(total_unproductive_time, total_job_time)

    # Compute the "Unknown/Other" unproductive time.
    total_unproductive_time[BadputType.OTHER] = (
        self._compute_other_unproductive_time(
            total_job_time, productive_training_time, total_unproductive_time
        )
    )

    # Compute the job Goodput and Badput breakdown.
    job_goodput = (float(productive_training_time) / total_job_time) * 100
    job_badput_breakdown = (
        self._get_job_badput_breakdown(total_unproductive_time, total_job_time)
        if include_badput_breakdown
        else {}
    )

    # Update the Goodput cache with new information.
    self._goodput_cache.update_goodput_info(
        GoodputInfo(
            total_productive_time=productive_training_time,
            total_elapsed_time_since_start=total_job_time,
            total_unproductive_time=total_unproductive_time,
            last_recorded_step=last_step,
            last_updated_timestamp=datetime.datetime.now(datetime.timezone.utc),
        )
    )
    return job_goodput, job_badput_breakdown, last_step

  def get_job_goodput_interval(
      self, interval_start: datetime.datetime, interval_end: datetime.datetime
  ) -> tuple[
      float,
      UnproductiveTimeDict,
      int,
      float,
      int,
  ]:
    """Method to get the Goodput and Badput breakdown of the job within an interval window.

    If the application is interested in retrieving the Goodput of the job within
    a specific window of time, this method provides the metrics computed between
    the start and end of this window.

    Additionaly, this method returns the last step recorded for the job. This is
    primarily used for improving monitoring and observability of the job's
    overall Goodput as a function of number of executed steps.

    Args:
      interval_start: The start time of the window for which Goodput is to be
        computed.
      interval_end: The end time of the window for which Goodput is to be
        computed.

    Returns:
      A tuple containing:
        - The job's Goodput percentage with respect to the total job time within
          the interval window.
        - The Badput Breakdown percentages with respect to the total job time
          within the interval window.
        - The last step recorded for the job within the interval window.
        - The total job time within the interval window.
        - The number of disruptions within the interval window.

    Raises:
      ValueError if computed total job time is zero. In this case, Goodput
      cannot be computed.
      ValueError if productive training or unproductive time is invalid.
    """
    # Get the logs for the interval and validate the interval window.
    self._get_interval_log_entries(interval_start, interval_end)

    total_job_time = self._get_total_job_time_from_interval(
        interval_start, interval_end
    )

    productive_training_time, total_unproductive_time, last_step = (
        self._get_current_productive_and_unproductive_time(interval_query=True)
    )
    if (
        productive_training_time < 0.0
        or productive_training_time > total_job_time
    ):
      raise ValueError(
          'Productive training time is invalid. Please fix the logging entries.'
      )

    # Sanitize unproductive times
    self._sanitize_unproductive_times(total_unproductive_time, total_job_time)

    # Compute the "Unknown/Other" unproductive time
    total_unproductive_time[BadputType.OTHER] = (
        self._compute_other_unproductive_time(
            total_job_time, productive_training_time, total_unproductive_time
        )
    )

    # Compute the job Goodput and Badput breakdown.
    job_goodput = (float(productive_training_time) / total_job_time) * 100
    job_badput_breakdown = self._get_job_badput_breakdown(
        total_unproductive_time, total_job_time
    )

    return (
        job_goodput,
        job_badput_breakdown,
        last_step,
        total_job_time,
        self._number_of_interruptions,
    )

  def _get_step_times(self, entries: list[Any]):
    """Helper function to compute step times from the log entries."""
    step_times = {}
    previous_step_start_time = None
    previous_step_count = None
    for payload in entries:
      if _STEP_START_TIME in payload:
        step_start_time = payload[_STEP_START_TIME]
        step_count = int(payload[_STEP_COUNT])
        if (
            previous_step_start_time is not None
            and previous_step_count is not None
            and step_count == previous_step_count + 1
        ):
          step_times[previous_step_count] = (
              step_start_time - previous_step_start_time
          )
        previous_step_count = step_count
        previous_step_start_time = step_start_time
    return step_times

  def _contains_step_entries(self, entries: list[Any]) -> bool:
    return any(_STEP_START_TIME in entry for entry in entries)

  def get_step_deviation(
      self, configured_ideal_step_time: Optional[float] = None
  ) -> dict[int, float]:
    """Method to get the step deviation of the current step based on the ideal step time.

    This method computes the ideal step time if one is not provided by the user
    and returns the step deviation of the current step.

    Args:
      configured_ideal_step_time: Optional user-defined ideal step time.

    Returns:
      A dictionary of step deviation for each step.
    """
    query_time = datetime.datetime.now(datetime.timezone.utc)
    new_entries = self._fetch_new_entries(query_time)
    with self._goodput_cache_lock:
      step_info = self._goodput_cache.get_step_info()

    if (
        not self._contains_step_entries(new_entries)
        and step_info
        and step_info.step_deviations
    ):
      return step_info.step_deviations

    with self._goodput_cache_lock:
      process_entries = self._goodput_cache.get_step_entries()

    step_times = self._get_step_times(process_entries)

    if not step_times:
      raise ValueError(
          'No step times available and no previous step deviations found.'
      )

    # Compute ideal step time.
    ideal_step_time = (
        configured_ideal_step_time
        if configured_ideal_step_time is not None
        else compute_ideal_step_time(list(step_times.values()))
    )
    if not ideal_step_time:
      raise ValueError(
          'No ideal step time available and no previous step deviations found.'
      )

    # Compute step deviation.
    step_deviations = {
        step_count: abs(step_time - ideal_step_time)
        for step_count, step_time in step_times.items()
    }
    # Update the step information in the cache.
    with self._goodput_cache_lock:
      self._goodput_cache.update_step_info(
          StepInfo(
              ideal_step_time=ideal_step_time,
              step_deviations=step_deviations,
          )
      )
    return step_deviations

  def _get_job_badput_breakdown(
      self, total_unproductive_time, total_job_time
  ) -> UnproductiveTimeDict:
    """Method to get the the Badput breakdown as percentage of total job time.

    This method provides a granular breakdown of the known components of Badput.

    Args:
      total_unproductive_time: A dictionary of computed unproductive time of
        each BadputType.
      total_job_time: The total job time.

    Returns:
      A dictionary of badput components and their percentage breakdown within
      total job time.
    """
    badput_breakdown: dict[
        BadputType, float | dict[str, float]
    ] = {}
    if total_job_time == 0.0:
      raise ValueError(
          'Total job time is zero, Badput cannot be calculated. Please fix the'
          ' logging entries.'
      )

    # TPU initialization badput.
    tpu_init_badput = total_unproductive_time.get(
        BadputType.TPU_INITIALIZATION, 0.0
    )
    badput_breakdown[BadputType.TPU_INITIALIZATION] = (
        (tpu_init_badput / total_job_time) * 100
        if 0 < tpu_init_badput < total_job_time
        else 0.0
    )

    # Training preparation badput.
    training_prep_badput = total_unproductive_time.get(
        BadputType.TRAINING_PREP, 0.0
    )
    badput_breakdown[BadputType.TRAINING_PREP] = (
        (training_prep_badput / total_job_time) * 100
        if 0 < training_prep_badput < total_job_time
        else 0.0
    )

    # Only synchronous data loading is badput.
    # Sync data loading is accumulated after start and reset of the job and is
    # blocking.
    sync_data_loading_badput = total_unproductive_time.get(
        BadputType.DATA_LOADING_SYNC, 0.0
    )
    # Async data loading is accumulated overlapping with training and is
    # non-blocking, therefore is not unproductive time.
    async_data_loading_badput = total_unproductive_time.get(
        BadputType.DATA_LOADING_ASYNC, 0.0
    )
    badput_breakdown[BadputType.DATA_LOADING_SYNC] = (
        (sync_data_loading_badput / total_job_time) * 100
        if 0 < sync_data_loading_badput < total_job_time
        else 0.0
    )
    badput_breakdown[BadputType.DATA_LOADING_ASYNC] = (
        (async_data_loading_badput / total_job_time) * 100
        if 0 < async_data_loading_badput < total_job_time
        else 0.0
    )

    # Unproductive checkpoint save time badput.
    checkpoint_save_badput = total_unproductive_time.get(
        BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME, 0.0
    )
    badput_breakdown[BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME] = (
        (checkpoint_save_badput / total_job_time) * 100
        if 0 < checkpoint_save_badput < total_job_time
        else 0.0
    )

    # Unproductive checkpoint restore time badput.
    checkpoint_restore_badput = total_unproductive_time.get(
        BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME, 0.0
    )
    badput_breakdown[BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME] = (
        (checkpoint_restore_badput / total_job_time) * 100
        if 0 < checkpoint_restore_badput < total_job_time
        else 0.0
    )

    # Program startup badput.
    program_startup_badput = total_unproductive_time.get(
        BadputType.PROGRAM_STARTUP, 0.0
    )
    badput_breakdown[BadputType.PROGRAM_STARTUP] = (
        (program_startup_badput / total_job_time) * 100
        if 0 < program_startup_badput < total_job_time
        else 0.0
    )

    # Wasted progress from disruption badput.
    wasted_progress_from_disruption_badput = total_unproductive_time.get(
        BadputType.WASTED_PROGRESS_FROM_DISRUPTION, 0.0
    )
    badput_breakdown[BadputType.WASTED_PROGRESS_FROM_DISRUPTION] = (
        (wasted_progress_from_disruption_badput / total_job_time) * 100
        if 0 < wasted_progress_from_disruption_badput < total_job_time
        else 0.0
    )

    # Custom events badput.
    badput_breakdown[BadputType.CUSTOM_BADPUT_EVENTS] = {}
    custom_events_badput = total_unproductive_time.get(
        BadputType.CUSTOM_BADPUT_EVENTS, {}
    )

    if isinstance(custom_events_badput, dict):
      nested_breakdown = {}
      for (
          custom_badput_type,
          custom_events_badput_value,
      ) in custom_events_badput.items():
        nested_breakdown[custom_badput_type] = (
            (custom_events_badput_value / total_job_time) * 100
            if 0 < custom_events_badput_value < total_job_time
            else 0.0
        )
      badput_breakdown[BadputType.CUSTOM_BADPUT_EVENTS] = (
          nested_breakdown
      )

    # Populate the 'Other/Unknown' badput bucket.
    other_badput = total_unproductive_time.get(BadputType.OTHER, 0.0)
    badput_breakdown[BadputType.OTHER] = (
        (other_badput / total_job_time) * 100
        if 0 < other_badput < total_job_time
        else 0.0
    )

    return badput_breakdown

  def get_job_goodput_details(
      self,
  ) -> dict[
      str,
      dict[
          Union[BadputType, GoodputType],
          float | dict[str, float],
      ],
  ]:
    """Method to get the productive and non-productive time with breakdown of the job computed until now."""

    goodput_info = self._goodput_cache.get_goodput_info()
    if goodput_info is None:
      logger.warning(
          'Goodput information unavailable and will not be uploaded to GCM'
      )
      return {
          'goodput_time_dict': {},
          'badput_time_dict': {},
      }

    (
        productive_training_time,
        total_unproductive_time,
        cache_last_updated_timestamp,
    ) = (
        goodput_info.total_productive_time,
        goodput_info.total_unproductive_time,
        goodput_info.last_updated_timestamp,
    )

    if (
        self._gcm_last_recorded_timestamp is not None  # Ignore the first entry.
        and self._gcm_last_recorded_timestamp >= cache_last_updated_timestamp
    ):
      logger.warning(
          'No new data, skipping upload to GCM. Cache Timestamp: %s, GCM'
          ' Timestamp: %s', cache_last_updated_timestamp,
          self._gcm_last_recorded_timestamp,
      )
      return {
          'goodput_time_dict': {},
          'badput_time_dict': {},
      }

    self._gcm_last_recorded_timestamp = datetime.datetime.now(
        datetime.timezone.utc
    )

    # Currently productive_time is not split based on productive activities, it
    # is just the total productive time. We will modify this to follow the same
    # format as badput_breakdown. Please update this code accordingly in the
    # future when we have more granular breakdown of productive time.

    total_productive_time = {GoodputType.TOTAL: productive_training_time}

    return {
        'goodput_time_dict': total_productive_time,
        'badput_time_dict': total_unproductive_time,
    }

  def get_job_goodput_interval_details(
      self, interval_start: datetime.datetime, interval_end: datetime.datetime
  ) -> dict[
      str,
      dict[
          Union[BadputType, GoodputType],
          float | dict[str, float],
      ],
  ]:
    """Method to get the productive and non-productive time with breakdown of the job computed within an interval window."""
    try:
      goodput, badput_breakdown, _, total_job_time, _ = (
          self.get_job_goodput_interval(interval_start, interval_end)
      )
      productive_time = goodput * total_job_time / 100
      total_unproductive_time = {}
      for badput_type, badput_value in badput_breakdown.items():
        total_unproductive_time[badput_type] = (
            badput_value * total_job_time / 100
        )
      total_productive_time = {GoodputType.TOTAL: productive_time}

      return {
          'goodput_time_dict': total_productive_time,
          'badput_time_dict': total_unproductive_time,
      }
    except ValueError as e:
      logger.warning('Failed to get job goodput interval details: %s', e)
      return {
          'goodput_time_dict': {},
          'badput_time_dict': {},
      }
