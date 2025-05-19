"""Checkpoint Badput Calculator class."""

import argparse
import dataclasses
import statistics
from typing import Dict, List, Optional

import google.cloud.logging as google_cloud_logging


_JOB_NAME = 'checkpoint_job'
_LOGGER_NAME = 'checkpoint_logger'

_STEP = 'step'
_EVENT_TYPE = 'event_type'
_DIRECTORY = 'directory'

_WAIT_FOR_PREV_DURATION_SECS = 'wait_for_prev_duration_secs'

_CHECKPOINTER_SAVE_DURATION_SECS = 'checkpointer_blocking_duration_secs'
_CHECKPOINTER_RESTORE_DURATION_SECS = 'checkpointer_duration_secs'

_GET_OLD_STEPS_DURATION_SECS = 'get_old_steps_duration_secs'

_CHECKPOINT_MANAGER_SAVE_DURATION_SECS = 'checkpoint_manager_blocking_duration_secs'
_CHECKPOINT_MANAGER_RESTORE_DURATION_SECS = 'checkpoint_manager_duration_secs'

_BROADCAST_DURATION_SECS = 'broadcast_duration_secs'

OPERATION_TYPE_SAVE = 'save'
OPERATION_TYPE_RESTORE = 'restore'
OPERATION_TYPE_EMERGENCY_RESTORE = 'emergency_restore'

OPERATION_TYPE_LOCAL = 'local'
OPERATION_TYPE_PERSISTENT = 'persistent'
OPERATION_TYPE_PERSISTENT_AND_LOCAL = 'persistent_and_local'

_CLOUD_LOGGING_PAGE_SIZE = 10000


@dataclasses.dataclass
class SaveCheckpointManagerVerticalStepStats:
  """Vertical step statistics for save operation."""
  total_checkpoint_manager_blocking_time: float = 0.0
  average_checkpoint_manager_blocking_time: float = 0.0
  minimum_checkpoint_manager_blocking_time: float = 0.0
  maximum_checkpoint_manager_blocking_time: float = 0.0
  standard_deviation_checkpoint_manager_blocking_time: float = 0.0

  total_checkpointer_blocking_time: float = 0.0
  average_checkpointer_blocking_time: float = 0.0
  minimum_checkpointer_blocking_time: float = 0.0
  maximum_checkpointer_blocking_time: float = 0.0
  standard_deviation_checkpointer_blocking_time: float = 0.0

  total_wait_for_prev_time: float = 0.0
  average_wait_for_prev_time: float = 0.0
  minimum_wait_for_prev_time: float = 0.0
  maximum_wait_for_prev_time: float = 0.0
  standard_deviation_wait_for_prev_time: float = 0.0

  total_get_old_steps_time: float = 0.0
  average_get_old_steps_time: float = 0.0
  minimum_get_old_steps_time: float = 0.0
  maximum_get_old_steps_time: float = 0.0
  standard_deviation_get_old_steps_time: float = 0.0


@dataclasses.dataclass
class RestoreCheckpointManagerVerticalStepStats:
  """Vertical step statistics for restore operation."""
  total_checkpoint_manager_time: float = 0.0
  average_checkpoint_manager_time: float = 0.0
  minimum_checkpoint_manager_time: float = 0.0
  maximum_checkpoint_manager_time: float = 0.0
  standard_deviation_checkpoint_manager_time: float = 0.0

  total_restore_time: float = 0.0
  average_restore_time: float = 0.0
  minimum_restore_time: float = 0.0
  maximum_restore_time: float = 0.0
  standard_deviation_restore_time: float = 0.0

  total_broadcast_time: float = 0.0
  average_broadcast_time: float = 0.0
  minimum_broadcast_time: float = 0.0
  maximum_broadcast_time: float = 0.0
  standard_deviation_broadcast_time: float = 0.0


@dataclasses.dataclass
class SaveProcessedStep:
  """Horizontal save step stats for a processed step."""
  step: str = ''
  total_checkpoint_manager_blocking_time: float = 0.0
  total_checkpointer_blocking_time: float = 0.0
  total_wait_for_prev_time: float = 0.0
  total_get_old_steps_time: float = 0.0
  occurrence: int = 0


@dataclasses.dataclass
class RestoreProcessedStep:
  """Horizontal restore step stats for a processed step."""
  step: str = ''
  total_checkpoint_manager_time: float = 0.0
  total_restore_time: float = 0.0
  total_broadcast_time: float = 0.0
  broadcast_occurrence: int = 0
  occurrence: int = 0


@dataclasses.dataclass
class CheckpointLoggerOptions:
  """Checkpoint logger options."""
  job_name: str = _JOB_NAME
  logger_name: str = _LOGGER_NAME
  client: Optional[google_cloud_logging.Client] = None
  use_goodput_logger: bool = False


class CheckpointBadputCalculator:
  """Checkpoint Badput Calculator class."""

  def __init__(
      self, options: CheckpointLoggerOptions = CheckpointLoggerOptions()
  ):
    self._options = options
    if not options.use_goodput_logger:
      if options.client is None:
        self.logging_client = google_cloud_logging.Client()
      else:
        self.logging_client = options.client
      self._logger = self.logging_client.logger(options.logger_name)
    self._use_goodput_logger = options.use_goodput_logger
    self.entries = []

  def read_entries(self) -> List[Dict[str, str]]:
    """Queries Cloud Logging entries for the specific job.

    Returns:
      Filtered entries in ascending order of timestamp.
    """
    if self._use_goodput_logger:
      return self.entries

    filter_entries = [
        'severity=INFO',
        f'jsonPayload.job_name="{self._options.job_name}"',
    ]

    event_type_filter = (
        '(jsonPayload.event_type=save OR jsonPayload.event_type=restore OR'
        ' jsonPayload.event_type=emergency_restore)'
    )
    filter_entries.append(event_type_filter)

    filter_entries = ' AND '.join(filter_entries)

    entries = self._logger.list_entries(
        filter_=filter_entries,
        order_by=google_cloud_logging.ASCENDING,
        page_size=_CLOUD_LOGGING_PAGE_SIZE,
    )
    entry_payload = [entry.payload for entry in entries]
    return entry_payload

  def _is_local_operation(self, step_stats: Dict[str, str]):
    if (step_stats[_DIRECTORY]).startswith('gs://'):
      return False
    else:
      return True

  def is_valid_save_stats(
      self,
      step_stats: Dict[str, str],
      operation_type: Optional[str] = OPERATION_TYPE_PERSISTENT_AND_LOCAL,
  ):
    """Checks if the step stats is valid.

    Args:
      step_stats: The step stats to check.
      operation_type: whether to check for local or persistent or both.

    Returns:
      Boolean indicating whether the step stats is valid.
    """
    if (
        _EVENT_TYPE not in step_stats
        or step_stats[_EVENT_TYPE] != OPERATION_TYPE_SAVE
    ):
      return False
    elif operation_type == OPERATION_TYPE_LOCAL:
      return self._is_local_operation(step_stats)
    elif operation_type == OPERATION_TYPE_PERSISTENT:
      return not self._is_local_operation(step_stats)
    else:
      return True

  def is_valid_restore_stats(
      self,
      step_stats: Dict[str, str],
      operation_type: Optional[str] = OPERATION_TYPE_PERSISTENT_AND_LOCAL,
  ):
    """Checks if the step stats is valid.

    Args:
      step_stats: The step stats to check.
      operation_type: whether to check for local or persistent or both.

    Returns:
      Boolean indicating whether the step stats is valid.

    """
    if _EVENT_TYPE not in step_stats:
      return False
    elif step_stats[_EVENT_TYPE] not in [
        OPERATION_TYPE_RESTORE,
        OPERATION_TYPE_EMERGENCY_RESTORE,
    ]:
      return False
    elif operation_type == OPERATION_TYPE_LOCAL:
      return step_stats[_EVENT_TYPE] == OPERATION_TYPE_EMERGENCY_RESTORE
    elif operation_type == OPERATION_TYPE_PERSISTENT:
      return step_stats[_EVENT_TYPE] == OPERATION_TYPE_RESTORE
    else:
      return True

  def _save_statistics(
      self, processed_step_stats: Dict[str, SaveProcessedStep]
  ) -> SaveCheckpointManagerVerticalStepStats:
    """Gets the processed step stats."""
    if not processed_step_stats:
      return SaveCheckpointManagerVerticalStepStats()

    for _, stats in processed_step_stats.items():
      if stats.occurrence > 0:
        stats.total_checkpoint_manager_blocking_time = (
            stats.total_checkpoint_manager_blocking_time / stats.occurrence
        )
        stats.total_checkpointer_blocking_time = (
            stats.total_checkpointer_blocking_time / stats.occurrence
        )
        stats.total_wait_for_prev_time = (
            stats.total_wait_for_prev_time / stats.occurrence
        )
        stats.total_get_old_steps_time = (
            stats.total_get_old_steps_time / stats.occurrence
        )

    vertical_step_stats = SaveCheckpointManagerVerticalStepStats()

    # Record statistics for checkpoint_manager_blocking_time.
    vertical_step_stats.total_checkpoint_manager_blocking_time = sum(
        map(
            lambda stats: stats.total_checkpoint_manager_blocking_time,
            processed_step_stats.values(),
        )
    )
    vertical_step_stats.average_checkpoint_manager_blocking_time = (
        vertical_step_stats.total_checkpoint_manager_blocking_time
        / len(processed_step_stats)
    )
    vertical_step_stats.minimum_checkpoint_manager_blocking_time = min(
        map(
            lambda stats: stats.total_checkpoint_manager_blocking_time,
            processed_step_stats.values(),
        )
    )
    vertical_step_stats.maximum_checkpoint_manager_blocking_time = max(
        map(
            lambda stats: stats.total_checkpoint_manager_blocking_time,
            processed_step_stats.values(),
        )
    )
    if len(processed_step_stats) > 1:
      vertical_step_stats.standard_deviation_checkpoint_manager_blocking_time = (
          statistics.stdev(
              map(
                  lambda stats: stats.total_checkpoint_manager_blocking_time,
                  processed_step_stats.values(),
              )
          )
      )

    # Record statistics for checkpointer_blocking_time.
    vertical_step_stats.total_checkpointer_blocking_time = sum(
        map(
            lambda stats: stats.total_checkpointer_blocking_time,
            processed_step_stats.values(),
        )
    )
    vertical_step_stats.average_checkpointer_blocking_time = (
        vertical_step_stats.total_checkpointer_blocking_time
        / len(processed_step_stats)
    )
    vertical_step_stats.minimum_checkpointer_blocking_time = min(
        map(
            lambda stats: stats.total_checkpointer_blocking_time,
            processed_step_stats.values(),
        )
    )
    vertical_step_stats.maximum_checkpointer_blocking_time = max(
        map(
            lambda stats: stats.total_checkpointer_blocking_time,
            processed_step_stats.values(),
        )
    )
    if len(processed_step_stats) > 1:
      vertical_step_stats.standard_deviation_checkpointer_blocking_time = (
          statistics.stdev(
              map(
                  lambda stats: stats.total_checkpointer_blocking_time,
                  processed_step_stats.values(),
              )
          )
      )

    # Record statistics for wait_for_prev_time.
    vertical_step_stats.total_wait_for_prev_time = sum(
        map(
            lambda stats: stats.total_wait_for_prev_time,
            processed_step_stats.values(),
        )
    )
    vertical_step_stats.average_wait_for_prev_time = (
        vertical_step_stats.total_wait_for_prev_time
        / len(processed_step_stats)
    )
    vertical_step_stats.minimum_wait_for_prev_time = min(
        map(
            lambda stats: stats.total_wait_for_prev_time,
            processed_step_stats.values(),
        )
    )
    vertical_step_stats.maximum_wait_for_prev_time = max(
        map(
            lambda stats: stats.total_wait_for_prev_time,
            processed_step_stats.values(),
        )
    )
    if len(processed_step_stats) > 1:
      vertical_step_stats.standard_deviation_wait_for_prev_time = (
          statistics.stdev(
              map(
                  lambda stats: stats.total_wait_for_prev_time,
                  processed_step_stats.values(),
              )
          )
      )

    # Record statistics for get_old_steps_time.
    vertical_step_stats.total_get_old_steps_time = sum(
        map(
            lambda stats: stats.total_get_old_steps_time,
            processed_step_stats.values(),
        )
    )
    vertical_step_stats.average_get_old_steps_time = (
        vertical_step_stats.total_get_old_steps_time / len(processed_step_stats)
    )
    vertical_step_stats.minimum_get_old_steps_time = min(
        map(
            lambda stats: stats.total_get_old_steps_time,
            processed_step_stats.values(),
        )
    )
    vertical_step_stats.maximum_get_old_steps_time = max(
        map(
            lambda stats: stats.total_get_old_steps_time,
            processed_step_stats.values(),
        )
    )
    if len(processed_step_stats) > 1:
      vertical_step_stats.standard_deviation_get_old_steps_time = (
          statistics.stdev(
              map(
                  lambda stats: stats.total_get_old_steps_time,
                  processed_step_stats.values(),
              )
          )
      )
    return vertical_step_stats

  def calculate_save_operation_checkpoint_manager_blocking_time(
      self, operation_type: Optional[str] = OPERATION_TYPE_PERSISTENT_AND_LOCAL,
  ) -> SaveCheckpointManagerVerticalStepStats:
    """Gets checkpoint manager blocking time breakdown for save operation."""
    self.entries = self.read_entries()

    step_already_processed: dict[str, SaveProcessedStep] = dict()
    for step_stats in self.entries:
      if (
          not self.is_valid_save_stats(step_stats, operation_type)
      ):
        continue

      # Create a step info to identify the step_statistics whether local or
      # persistent.
      if self._is_local_operation(step_stats):
        step_info = str(step_stats[_STEP]) + '-' + OPERATION_TYPE_LOCAL
      else:
        step_info = (
            str(step_stats[_STEP]) + '-' + OPERATION_TYPE_PERSISTENT
        )
      if step_already_processed.get(step_info) is None:
        step_already_processed[step_info] = SaveProcessedStep()
        step_already_processed[step_info].step = step_info
        step_already_processed[
            step_info
        ].total_checkpoint_manager_blocking_time = float(
            step_stats[_CHECKPOINT_MANAGER_SAVE_DURATION_SECS]
        )
        step_already_processed[step_info].total_checkpointer_blocking_time = (
            float(step_stats[_CHECKPOINTER_SAVE_DURATION_SECS])
        )
        step_already_processed[step_info].total_wait_for_prev_time = float(
            step_stats[_WAIT_FOR_PREV_DURATION_SECS]
        )
        step_already_processed[step_info].total_get_old_steps_time = float(
            step_stats[_GET_OLD_STEPS_DURATION_SECS]
        )
        step_already_processed[step_info].occurrence = 1
      else:
        step_already_processed[step_info].step = step_info
        step_already_processed[
            step_info
        ].total_checkpoint_manager_blocking_time += float(
            step_stats[_CHECKPOINT_MANAGER_SAVE_DURATION_SECS]
        )
        step_already_processed[
            step_info
        ].total_checkpointer_blocking_time += float(
            step_stats[_CHECKPOINTER_SAVE_DURATION_SECS]
        )
        step_already_processed[step_info].total_wait_for_prev_time += float(
            step_stats[_WAIT_FOR_PREV_DURATION_SECS]
        )
        step_already_processed[step_info].total_get_old_steps_time += float(
            step_stats[_GET_OLD_STEPS_DURATION_SECS]
        )
        step_already_processed[step_info].occurrence += 1

    # Calculate the vertical step stats for the checkpoint manager blocking
    # time.
    save_statistics = self._save_statistics(
        step_already_processed
    )

    return save_statistics

  def _restore_statistics(
      self, processed_step_stats: Dict[str, RestoreProcessedStep]
  ) -> RestoreCheckpointManagerVerticalStepStats:
    """Calculates the vertical step stats."""
    if not processed_step_stats:
      return RestoreCheckpointManagerVerticalStepStats()
    broadcast_occurrence = 0
    for _, stats in processed_step_stats.items():
      stats.total_checkpoint_manager_time = (
          stats.total_checkpoint_manager_time / stats.occurrence
      )
      stats.total_restore_time = stats.total_restore_time / stats.occurrence
      if stats.broadcast_occurrence > 0:
        stats.total_broadcast_time = (
            stats.total_broadcast_time / stats.broadcast_occurrence
        )
        broadcast_occurrence += 1

    vertical_step_stats = RestoreCheckpointManagerVerticalStepStats()

    # Record statistics for total time checkpoint manager spent on restore.
    vertical_step_stats.total_checkpoint_manager_time = sum(
        map(
            lambda stats: stats.total_checkpoint_manager_time,
            processed_step_stats.values(),
        )
    )
    vertical_step_stats.average_checkpoint_manager_time = (
        vertical_step_stats.total_checkpoint_manager_time
        / len(processed_step_stats)
    )
    vertical_step_stats.minimum_checkpoint_manager_time = min(
        map(
            lambda stats: stats.total_checkpoint_manager_time,
            processed_step_stats.values(),
        )
    )
    vertical_step_stats.maximum_checkpoint_manager_time = max(
        map(
            lambda stats: stats.total_checkpoint_manager_time,
            processed_step_stats.values(),
        )
    )
    if len(processed_step_stats) > 1:
      vertical_step_stats.standard_deviation_checkpoint_manager_time = (
          statistics.stdev(
              map(
                  lambda stats: stats.total_checkpoint_manager_time,
                  processed_step_stats.values(),
              )
          )
      )
    # Record statistics for restore time.
    vertical_step_stats.total_restore_time = sum(
        map(
            lambda stats: stats.total_restore_time,
            processed_step_stats.values(),
        )
    )
    vertical_step_stats.average_restore_time = (
        vertical_step_stats.total_restore_time / len(processed_step_stats)
    )
    vertical_step_stats.minimum_restore_time = min(
        map(
            lambda stats: stats.total_restore_time,
            processed_step_stats.values(),
        )
    )
    vertical_step_stats.maximum_restore_time = max(
        map(
            lambda stats: stats.total_restore_time,
            processed_step_stats.values(),
        )
    )
    if len(processed_step_stats) > 1:
      vertical_step_stats.standard_deviation_restore_time = (
          statistics.stdev(
              map(
                  lambda stats: stats.total_restore_time,
                  processed_step_stats.values(),
              )
          )
      )

    # Record statistics for broadcasting the restored checkpoint(Emergency
    # restore only).
    if broadcast_occurrence > 0:
      vertical_step_stats.total_broadcast_time = sum(
          map(
              lambda stats: stats.total_broadcast_time,
              processed_step_stats.values(),
          )
      )
      vertical_step_stats.average_broadcast_time = (
          vertical_step_stats.total_broadcast_time / broadcast_occurrence
      )
      vertical_step_stats.minimum_broadcast_time = min(
          map(
              lambda stats: stats.total_broadcast_time,
              processed_step_stats.values(),
          )
      )
      vertical_step_stats.maximum_broadcast_time = max(
          map(
              lambda stats: stats.total_broadcast_time,
              processed_step_stats.values(),
          )
      )
      if len(processed_step_stats) > 1:
        vertical_step_stats.standard_deviation_broadcast_time = (
            statistics.stdev(
                map(
                    lambda stats: stats.total_broadcast_time,
                    processed_step_stats.values(),
                )
            )
        )

    return vertical_step_stats

  def calculate_restore_operation_checkpoint_manager_blocking_time(
      self,
      operation_type: Optional[str] = OPERATION_TYPE_PERSISTENT_AND_LOCAL,
  ) -> RestoreCheckpointManagerVerticalStepStats:
    """Gets checkpoint manager blocking time breakdown for restore operation."""
    self.entries = self.read_entries()

    step_already_processed: dict[str, RestoreProcessedStep] = dict()
    for step_stats in self.entries:
      if not self.is_valid_restore_stats(step_stats, operation_type):
        continue

      # Create a step info to identify the step_stats whether local or
      if self._is_local_operation(step_stats):
        step_info = str(step_stats[_STEP]) + '-' + OPERATION_TYPE_LOCAL
      else:
        step_info = str(step_stats[_STEP]) + '-' + OPERATION_TYPE_PERSISTENT

      if step_already_processed.get(step_info) is None:
        step_already_processed[step_info] = RestoreProcessedStep()
        step_already_processed[step_info].step = step_info

        step_already_processed[step_info].total_checkpoint_manager_time = float(
            step_stats[_CHECKPOINT_MANAGER_RESTORE_DURATION_SECS]
        )
        step_already_processed[step_info].total_restore_time = float(
            step_stats[_CHECKPOINTER_RESTORE_DURATION_SECS]
        )
        if (
            step_stats.get(_BROADCAST_DURATION_SECS)
            and step_stats[_BROADCAST_DURATION_SECS] is not None
        ):
          step_already_processed[step_info].total_broadcast_time = float(
              step_stats[_BROADCAST_DURATION_SECS]
          )
          step_already_processed[step_info].broadcast_occurrence = 1
        step_already_processed[step_info].occurrence = 1
      else:
        step_already_processed[step_info].step = step_info
        step_already_processed[
            step_info
        ].total_checkpoint_manager_time += float(
            step_stats[_CHECKPOINT_MANAGER_RESTORE_DURATION_SECS]
        )
        step_already_processed[step_info].total_restore_time += float(
            step_stats[_CHECKPOINTER_RESTORE_DURATION_SECS]
        )
        if (
            step_stats.get(_BROADCAST_DURATION_SECS)
            and step_stats[_BROADCAST_DURATION_SECS] is not None
        ):
          step_already_processed[step_info].total_broadcast_time += float(
              step_stats[_BROADCAST_DURATION_SECS]
          )
          step_already_processed[step_info].broadcast_occurrence += 1
        step_already_processed[step_info].occurrence += 1

    # Calculate the vertical step stats for the checkpoint manager blocking
    # time.
    restore_statistics = self._restore_statistics(step_already_processed)

    return restore_statistics

if  __name__ == '__main__':
  parser = argparse.ArgumentParser()
  options = CheckpointLoggerOptions()
  parser.add_argument(
      '--job_name',
      type=str,
      default=options.job_name,
      help='The name of the job.',
  )
  parser.add_argument(
      '--logger_name',
      type=str,
      default=options.logger_name,
      help='The name of the logger.',
  )
  parser.add_argument(
      '--client',
      type=str,
      default=options.client,
      help='The name of the client.',
  )
  parser.add_argument(
      '--operation_type',
      type=str,
      default=OPERATION_TYPE_PERSISTENT_AND_LOCAL,
      help='The operation type.',
  )
  args = parser.parse_args()
  options = CheckpointLoggerOptions(
      job_name=args.job_name,
      logger_name=args.logger_name,
      client=args.client,
  )
  checkpoint_badput_calculator = (
      CheckpointBadputCalculator(options)
  )
  checkpoint_badput_calculator.calculate_save_operation_checkpoint_manager_blocking_time(
      args.operation_type
  )



