"""Tests for checkpoint badput calculator."""

import dataclasses
from typing import Optional

from absl.testing import absltest
from cloud_goodput.ml_goodput_measurement.src import checkpoint_badput_calculator
import google.cloud.logging as google_cloud_logging
import mock


_JOB_NAME = 'checkpoint_job'
_LOGGER_NAME = 'checkpoint_logger'


@dataclasses.dataclass
class MockSaveStepStatistics:
  """Attributes for save step statistics.

  Attributes:
    step: The step number.
    event_type: The event type.
    checkpoint_manager_blocking_start_time: The start time of checkpoint manager
      blocking section.
    directory: The directory of the checkpoint.
    reached_preemption: Whether the event reached preemption.
    preemption_received_at: The time when preemption was received.
    wait_for_prev_start_time: The start time of waiting for previous checkpoint.
    checkpointer_blocking_start_time: The start time of blocking time introduced
      by checkpointer.
    get_old_steps_start_time: The start time of getting old steps.
    synchronous: Whether the event is synchronous.
    wait_for_prev_duration_secs: The duration of waiting for previous
      checkpoint.
    checkpointer_blocking_duration_secs: The duration of blocking time
      introduced by checkpointer.
    get_old_steps_duration_secs: The duration of getting old steps.
    checkpoint_manager_blocking_duration_secs: The duration of checkpoint
      manager blocking section.
  """

  step: Optional[int] = None
  event_type: Optional[str] = 'save'
  directory: Optional[str] = None
  reached_preemption: Optional[bool] = False
  preemption_received_at: Optional[float] = None
  synchronous: Optional[bool] = False
  wait_for_prev_start_time: Optional[float] = None
  wait_for_prev_duration_secs: Optional[float] = None
  checkpointer_blocking_start_time: Optional[float] = None
  checkpointer_blocking_duration_secs: Optional[float] = None
  get_old_steps_start_time: Optional[float] = None
  get_old_steps_duration_secs: Optional[float] = None
  checkpoint_manager_blocking_start_time: Optional[float] = None
  checkpoint_manager_blocking_duration_secs: Optional[float] = None


@dataclasses.dataclass
class MockRestoreStepStatistics:
  """Attributes for restore step statistics.

  Attributes:
    step: The step number.
    event_type: The event type.
    directory: The directory of the checkpoint.
    checkpointer_start_time: The start time of restoring the checkpoint, while
      using the checkpointer.
    checkpointer_duration_secs: The total duration for restoring the checkpoint,
      while using the checkpointer.
    checkpoint_manager_start_time: The start time for restoring the checkpoint,
      while using the checkpoint manager.
    checkpoint_manager_duration_secs: The total duration for restoring the
      checkpoint, while using the checkpoint manager.
  """

  step: Optional[int] = None
  event_type: Optional[str] = 'restore'
  directory: Optional[str] = None
  checkpointer_start_time: Optional[float] = None
  checkpointer_duration_secs: Optional[float] = None
  checkpoint_manager_start_time: Optional[float] = None
  checkpoint_manager_duration_secs: Optional[float] = None


@dataclasses.dataclass
class MockEmergencyRestoreStepStatistics:
  """Attributes for emergency restore step statistics.

  Attributes:
    step: The step number.
    event_type: The event type.
    checkpoint_manager_start_time: The start time of checkpoint manager
      restore event.
    directory: The directory of the checkpoint.
    is_restoring_slice: Whether the event takes place on the slice responsible
      for reading from the storage location. (Note that in_primary_slice=True
      necessarily implies is_restoring_slice=True.)
    in_primary_slice: Whether the event takes place on the slice designated as
      primary (responsible for restoring from persistent storage).
    checkpointer_start_time: The start time of restoring the checkpoint, while
      using the checkpointer.
    checkpointer_duration_secs: The total duration for restoring the checkpoint,
      while using the checkpointer.
    broadcast_start_time: The start time of broadcasting(Restore).The broadcast
      operation performed by SingleReplicaArrayHandler won't be captured in this
      context.
    broadcast_duration_secs: The duration of broadcasting(Restore).
    checkpoint_manager_duration_secs: The total duration of checkpoint
      manager restore event.
  """

  step: Optional[int] = None
  event_type: Optional[str] = 'emergency_restore'
  checkpoint_manager_start_time: Optional[float] = None
  directory: Optional[str] = None
  is_restoring_slice: Optional[bool] = False
  in_primary_slice: Optional[bool] = False
  checkpointer_start_time: Optional[float] = None
  checkpointer_duration_secs: Optional[float] = None
  broadcast_start_time: Optional[float] = None
  broadcast_duration_secs: Optional[float] = None
  checkpoint_manager_duration_secs: Optional[float] = None


class CheckpointBadputCalculatorTest(absltest.TestCase):

  def setUp(self):
    """Setup for the test."""
    super().setUp()
    mock_gcloud_client = mock.create_autospec(google_cloud_logging.Client)
    options = checkpoint_badput_calculator.CheckpointLoggerOptions(
        job_name=_JOB_NAME,
        logger_name=_LOGGER_NAME,
        client=mock_gcloud_client,
        use_goodput_logger=True,
    )
    self.checkpoint_badput_calculator = (
        checkpoint_badput_calculator.CheckpointBadputCalculator(options)
    )

  def test_checkpoint_badput_calculator_persistent_save_operation(self):
    """Test for persistent save operation."""
    step_count = 4
    default_cm_blocking_duration_secs = 4
    default_ckptr_blocking_duration_secs = 1
    default_gos_duration_secs = 1
    default_wfp_duration_secs = 2
    for i in range(1, step_count+1):
      persistent_save_entry = dataclasses.asdict(
          MockSaveStepStatistics(
              step=i,
              event_type='save',
              directory='gs://bucket/path',
              wait_for_prev_start_time=i * 10.0,
              wait_for_prev_duration_secs=default_wfp_duration_secs,
              checkpointer_blocking_start_time=i * 10.0 + 2,
              checkpointer_blocking_duration_secs=default_ckptr_blocking_duration_secs,
              get_old_steps_start_time=i * 10.0 + 3,
              get_old_steps_duration_secs=default_gos_duration_secs,
              checkpoint_manager_blocking_start_time=i * 10.0,
              checkpoint_manager_blocking_duration_secs=default_cm_blocking_duration_secs,
              reached_preemption=True,
              preemption_received_at=i * 10.0,
              synchronous=True,
          )
      )
      self.checkpoint_badput_calculator.entries.append(persistent_save_entry)

    expected_breakdown = (
        checkpoint_badput_calculator.SaveCheckpointManagerVerticalStepStats()
    )
    expected_breakdown.total_checkpoint_manager_blocking_time = (
        step_count * default_cm_blocking_duration_secs
    )
    expected_breakdown.average_checkpoint_manager_blocking_time = (
        default_cm_blocking_duration_secs
    )
    expected_breakdown.minimum_checkpoint_manager_blocking_time = (
        default_cm_blocking_duration_secs
    )
    expected_breakdown.maximum_checkpoint_manager_blocking_time = (
        default_cm_blocking_duration_secs
    )
    expected_breakdown.standard_deviation_checkpoint_manager_blocking_time = 0
    expected_breakdown.total_checkpointer_blocking_time = (
        step_count * default_ckptr_blocking_duration_secs
    )
    expected_breakdown.average_checkpointer_blocking_time = (
        default_ckptr_blocking_duration_secs
    )
    expected_breakdown.minimum_checkpointer_blocking_time = (
        default_ckptr_blocking_duration_secs
    )
    expected_breakdown.maximum_checkpointer_blocking_time = (
        default_ckptr_blocking_duration_secs
    )
    expected_breakdown.standard_deviation_checkpointer_blocking_time = 0
    expected_breakdown.total_wait_for_prev_time = (
        step_count * default_wfp_duration_secs
    )
    expected_breakdown.average_wait_for_prev_time = default_wfp_duration_secs
    expected_breakdown.minimum_wait_for_prev_time = default_wfp_duration_secs
    expected_breakdown.maximum_wait_for_prev_time = default_wfp_duration_secs
    expected_breakdown.standard_deviation_wait_for_prev_time = 0
    expected_breakdown.total_get_old_steps_time = (
        step_count * default_gos_duration_secs
    )
    expected_breakdown.average_get_old_steps_time = default_gos_duration_secs
    expected_breakdown.minimum_get_old_steps_time = default_gos_duration_secs
    expected_breakdown.maximum_get_old_steps_time = default_gos_duration_secs
    expected_breakdown.standard_deviation_get_old_steps_time = 0

    cm_breakdown = (
        self.checkpoint_badput_calculator.calculate_save_operation_checkpoint_manager_blocking_time(
            checkpoint_badput_calculator.OPERATION_TYPE_PERSISTENT
        )
    )
    for field in dataclasses.fields(cm_breakdown):
      value1 = getattr(cm_breakdown, field.name)
      value2 = getattr(expected_breakdown, field.name)
      if value1 != value2:
        raise ValueError(
            f"Mismatch in field '{field.name}':\n"
            f"  Actual: {value1}\n"
            f"  Expected: {value2}"
        )

  def test_checkpoint_badput_calculator_local_save_operation(self):
    """Test for local save operation."""
    step_count = 4
    default_cm_blocking_duration_secs = 4
    default_ckptr_blocking_duration_secs = 1
    default_gos_duration_secs = 1
    default_wfp_duration_secs = 2
    for i in range(1, step_count+1):
      local_save_entry = dataclasses.asdict(
          MockSaveStepStatistics(
              step=i,
              event_type='save',
              directory='local',
              wait_for_prev_start_time=i * 10.0,
              wait_for_prev_duration_secs=default_wfp_duration_secs,
              checkpointer_blocking_start_time=i * 10.0 + 2,
              checkpointer_blocking_duration_secs=default_ckptr_blocking_duration_secs,
              get_old_steps_start_time=i * 10.0 + 3,
              get_old_steps_duration_secs=default_gos_duration_secs,
              checkpoint_manager_blocking_start_time=i * 10.0,
              checkpoint_manager_blocking_duration_secs=default_cm_blocking_duration_secs,
              reached_preemption=True,
              preemption_received_at=i * 10.0,
              synchronous=True,
          )
      )
      self.checkpoint_badput_calculator.entries.append(local_save_entry)

    expected_breakdown = (
        checkpoint_badput_calculator.SaveCheckpointManagerVerticalStepStats()
    )
    expected_breakdown.total_checkpoint_manager_blocking_time = (
        step_count * default_cm_blocking_duration_secs
    )
    expected_breakdown.average_checkpoint_manager_blocking_time = (
        default_cm_blocking_duration_secs
    )
    expected_breakdown.minimum_checkpoint_manager_blocking_time = (
        default_cm_blocking_duration_secs
    )
    expected_breakdown.maximum_checkpoint_manager_blocking_time = (
        default_cm_blocking_duration_secs
    )
    expected_breakdown.standard_deviation_checkpoint_manager_blocking_time = 0
    expected_breakdown.total_checkpointer_blocking_time = (
        step_count * default_ckptr_blocking_duration_secs
    )
    expected_breakdown.average_checkpointer_blocking_time = (
        default_ckptr_blocking_duration_secs
    )
    expected_breakdown.minimum_checkpointer_blocking_time = (
        default_ckptr_blocking_duration_secs
    )
    expected_breakdown.maximum_checkpointer_blocking_time = (
        default_ckptr_blocking_duration_secs
    )
    expected_breakdown.standard_deviation_checkpointer_blocking_time = 0
    expected_breakdown.total_wait_for_prev_time = (
        step_count * default_wfp_duration_secs
    )
    expected_breakdown.average_wait_for_prev_time = default_wfp_duration_secs
    expected_breakdown.minimum_wait_for_prev_time = default_wfp_duration_secs
    expected_breakdown.maximum_wait_for_prev_time = default_wfp_duration_secs
    expected_breakdown.standard_deviation_wait_for_prev_time = 0
    expected_breakdown.total_get_old_steps_time = (
        step_count * default_gos_duration_secs
    )
    expected_breakdown.average_get_old_steps_time = default_gos_duration_secs
    expected_breakdown.minimum_get_old_steps_time = default_gos_duration_secs
    expected_breakdown.maximum_get_old_steps_time = default_gos_duration_secs
    expected_breakdown.standard_deviation_get_old_steps_time = 0

    cm_breakdown = (
        self.checkpoint_badput_calculator.calculate_save_operation_checkpoint_manager_blocking_time(
            checkpoint_badput_calculator.OPERATION_TYPE_LOCAL
        )
    )
    for field in dataclasses.fields(cm_breakdown):
      value1 = getattr(cm_breakdown, field.name)
      value2 = getattr(expected_breakdown, field.name)
      if value1 != value2:
        raise ValueError(
            f"Mismatch in field '{field.name}':\n"
            f"  Actual: {value1}\n"
            f"  Expected: {value2}"
        )

  def test_checkpoint_badput_calculator_persistent_restore_operation(self):
    """Test for persistent restore operation."""
    step_count = 4
    default_cm_duration_secs = 4
    default_ckptr_duration_secs = 1
    for i in range(1, step_count+1):
      persitent_save_entry = dataclasses.asdict(
          MockRestoreStepStatistics(
              step=i,
              event_type='restore',
              directory='gs://bucket/path',
              checkpointer_start_time=i * 10.0,
              checkpointer_duration_secs=default_ckptr_duration_secs,
              checkpoint_manager_start_time=i * 10.0 + 2,
              checkpoint_manager_duration_secs=default_cm_duration_secs,
          )
      )
      self.checkpoint_badput_calculator.entries.append(persitent_save_entry)

    expected_breakdown = (
        checkpoint_badput_calculator.RestoreCheckpointManagerVerticalStepStats()
    )
    expected_breakdown.total_checkpoint_manager_time = (
        step_count * default_cm_duration_secs
    )
    expected_breakdown.average_checkpoint_manager_time = (
        default_cm_duration_secs
    )
    expected_breakdown.minimum_checkpoint_manager_time = (
        default_cm_duration_secs
    )
    expected_breakdown.maximum_checkpoint_manager_time = (
        default_cm_duration_secs
    )
    expected_breakdown.standard_deviation_checkpoint_manager_time = 0
    expected_breakdown.total_restore_time = (
        step_count * default_ckptr_duration_secs
    )
    expected_breakdown.average_restore_time = default_ckptr_duration_secs
    expected_breakdown.minimum_restore_time = default_ckptr_duration_secs
    expected_breakdown.maximum_restore_time = default_ckptr_duration_secs
    expected_breakdown.standard_deviation_restore_time = 0
    expected_breakdown.total_broadcast_time = 0
    expected_breakdown.average_broadcast_time = 0
    expected_breakdown.minimum_broadcast_time = 0
    expected_breakdown.maximum_broadcast_time = 0
    expected_breakdown.standard_deviation_broadcast_time = 0

    cm_breakdown = (
        self.checkpoint_badput_calculator.calculate_restore_operation_checkpoint_manager_blocking_time(
            checkpoint_badput_calculator.OPERATION_TYPE_PERSISTENT
        )
    )
    for field in dataclasses.fields(cm_breakdown):
      value1 = getattr(cm_breakdown, field.name)
      value2 = getattr(expected_breakdown, field.name)
      if value1 != value2:
        raise ValueError(
            f"Mismatch in field '{field.name}':\n"
            f"  Actual: {value1}\n"
            f"  Expected: {value2}"
        )

  def test_checkpoint_badput_calculator_local_restore_operation(self):
    """Test for local restore operation."""
    step_count = 4
    default_cm_duration_secs = 4
    default_ckptr_duration_secs = 2
    default_broadcast_duration_secs = 2
    for i in range(1, step_count+1):
      local_save_entry = dataclasses.asdict(
          MockEmergencyRestoreStepStatistics(
              step=i,
              event_type='emergency_restore',
              directory='local',
              checkpointer_start_time=i * 10.0,
              checkpointer_duration_secs=default_ckptr_duration_secs,
              checkpoint_manager_start_time=i * 10.0 + 2,
              checkpoint_manager_duration_secs=default_cm_duration_secs,
              broadcast_start_time=i * 10.0 + 3,
              broadcast_duration_secs=default_broadcast_duration_secs,
          )
      )
      self.checkpoint_badput_calculator.entries.append(local_save_entry)

    expected_breakdown = (
        checkpoint_badput_calculator.RestoreCheckpointManagerVerticalStepStats()
    )
    expected_breakdown.total_checkpoint_manager_time = (
        default_cm_duration_secs * step_count
    )
    expected_breakdown.average_checkpoint_manager_time = (
        default_cm_duration_secs
    )
    expected_breakdown.minimum_checkpoint_manager_time = (
        default_cm_duration_secs
    )
    expected_breakdown.maximum_checkpoint_manager_time = (
        default_cm_duration_secs
    )
    expected_breakdown.standard_deviation_checkpoint_manager_time = 0
    expected_breakdown.total_restore_time = (
        step_count * default_ckptr_duration_secs
    )
    expected_breakdown.average_restore_time = default_ckptr_duration_secs
    expected_breakdown.minimum_restore_time = default_ckptr_duration_secs
    expected_breakdown.maximum_restore_time = default_ckptr_duration_secs
    expected_breakdown.standard_deviation_restore_time = 0
    expected_breakdown.total_broadcast_time = (
        step_count * default_broadcast_duration_secs
    )
    expected_breakdown.average_broadcast_time = default_broadcast_duration_secs
    expected_breakdown.minimum_broadcast_time = default_broadcast_duration_secs
    expected_breakdown.maximum_broadcast_time = default_broadcast_duration_secs
    expected_breakdown.standard_deviation_broadcast_time = 0

    cm_breakdown = (
        self.checkpoint_badput_calculator.calculate_restore_operation_checkpoint_manager_blocking_time(
            checkpoint_badput_calculator.OPERATION_TYPE_LOCAL
        )
    )
    for field in dataclasses.fields(cm_breakdown):
      value1 = getattr(cm_breakdown, field.name)
      value2 = getattr(expected_breakdown, field.name)
      if value1 != value2:
        raise ValueError(
            f"Mismatch in field '{field.name}':\n"
            f"  Actual: {value1}\n"
            f"  Expected: {value2}"
        )
if __name__ == '__main__':
  absltest.main()
