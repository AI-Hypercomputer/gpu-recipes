"""Goodput monitoring API.

This file contains all the utilities to monitor and upload goodput data of a
user workload to Tensorboard asynchronously.
"""

import datetime
import logging
import math
import os
import threading
import time

from cloud_goodput.ml_goodput_measurement.src import gcp_metrics
from cloud_goodput.ml_goodput_measurement.src import goodput
from cloud_goodput.ml_goodput_measurement.src import goodput_utils
from tensorboardX import writer

BadputType = goodput_utils.BadputType
GCPOptions = goodput_utils.GCPOptions
GCPMetrics = gcp_metrics.GCPMetrics
GoodputCalculator = goodput.GoodputCalculator
ValueType = gcp_metrics.ValueType

ACTIVITY_EXCLUSION_LIST = goodput_utils.ACTIVITY_EXCLUSION_LIST
_TENSORBOARD_GCS_SUBDIR = 'goodput'
_TENSORBOARD_GOODPUT_LABEL = 'goodput'
_TENSORBOARD_BADPUT_LABEL = 'badput'
_TENSORBOARD_STEP_DEVIATION_LABEL = 'step_deviation'
_GOODPUT_DETAILS_KEY = 'goodput_time_dict'
_BADPUT_DETAILS_KEY = 'badput_time_dict'

logger = logging.getLogger(__name__)


class GoodputMonitor:
  """Queries and uploads goodput data to Tensorboard at a regular interval."""

  def __init__(
      self,
      job_name: str,
      logger_name: str,
      tensorboard_dir: str,
      upload_interval: int,
      monitoring_enabled: bool = False,
      pathway_enabled: bool = False,
      include_badput_breakdown=False,
      include_step_deviation=False,
      configured_ideal_step_time=None,
      step_deviation_interval_seconds=10,
      gcp_options: GCPOptions = GCPOptions(),
  ):
    """Initializes the GoodputMonitor.

    Args:
      job_name: The name of the job to monitor.
      logger_name: The name of the Google Cloud Logging logger to use.
      tensorboard_dir: The directory to write TensorBoard data to.
      upload_interval: The interval to upload data to TensorBoard and GCP
        Monitoring.
      monitoring_enabled: Whether to enable monitoring. If the application is
        interested in monitoring Goodput, it should set this value to True if
        monitoring from TPU worker 0 andthe application's configurations
        request Goodput monitoring.
      pathway_enabled: Whether the application is using Pathways.
      include_badput_breakdown: Whether to query and upload badput breakdown
        data to Tensorboard.
      include_step_deviation: Whether to query and upload step deviation data
        to Tensorboard.
      configured_ideal_step_time: The optional ideal step time configured by
        the user.
      step_deviation_interval_seconds: The interval to query step deviation
        data.
      gcp_options: The options for Google Cloud Monitoring.
    """
    if not monitoring_enabled:
      logger.info(
          'Monitoring is disabled. Returning without initializing'
          ' GoodputMonitor.'
      )
      return

    # Common configurations.
    self._job_name = job_name
    self._logger_name = logger_name
    self._tensorboard_dir = os.path.join(
        tensorboard_dir, _TENSORBOARD_GCS_SUBDIR
    )
    # Goodput configurations.
    self._upload_interval = upload_interval
    self._include_badput_breakdown = include_badput_breakdown

    # Step deviation configurations.
    self._include_step_deviation = include_step_deviation
    self._step_deviation_interval_seconds = step_deviation_interval_seconds
    self._configured_ideal_step_time = configured_ideal_step_time

    # Initialize the GoodputCalculator.
    self._goodput_calculator = GoodputCalculator(
        job_name=self._job_name,
        logger_name=self._logger_name,
        using_pathways=pathway_enabled,
    )
    self._writer = writer.SummaryWriter(self._tensorboard_dir)

    # Goodput uploader flags to signal the daemon thread if it exists when to
    # initate shutdown and wait for termination.
    self._goodput_uploader_thread_running = False
    self._goodput_upload_thread = None
    self._termination_event = threading.Event()
    self._termination_event.clear()

    # Step deviation threading flags.
    self._step_deviation_uploader_thread_running = False
    self._step_deviation_upload_thread = None
    self._step_deviation_termination_event = threading.Event()
    self._step_deviation_termination_event.clear()

    # Google Cloud Monitoring configurations.
    self._gcp_options = gcp_options
    self._metrics_sender = None

    # If step deviation is not included, disable GCP step deviation metrics.
    if not self._include_step_deviation:
      self._gcp_options.enable_gcp_step_deviation_metrics = False

    if (
        self._gcp_options.enable_gcp_goodput_metrics
        or self._gcp_options.enable_gcp_step_deviation_metrics
    ):
      if not self._gcp_options.project_id:
        self._gcp_options.project_id = goodput_utils.get_gcp_project_id()
      if not self._gcp_options.location:
        self._gcp_options.location = goodput_utils.get_node_zone()
      if not self._gcp_options.acc_type:
        self._gcp_options.acc_type = goodput_utils.get_accelerator_type()
      if self._gcp_options.project_id and self._gcp_options.location:
        self._metrics_sender = GCPMetrics(
            project_id=self._gcp_options.project_id
        )
      else:
        self._gcp_options.enable_gcp_goodput_metrics = False
        self._gcp_options.enable_gcp_step_deviation_metrics = False
        logger.warning(
            'Project ID or location is not set. GCP Monitoring will not be'
            ' enabled.'
        )
      # Goodput interval uploader flags.
      self._interval_uploader_thread_running = False
      self._interval_goodput_upload_thread = None
      self._interval_termination_event = threading.Event()
      self._interval_termination_event.clear()
      self._interval_window_size_seconds = 0

  def __del__(self):
    try:
      self.flush_and_stop_goodput_uploader()
      self.flush_and_stop_step_deviation_uploader()
      self.flush_and_stop_interval_goodput_uploader()

    except Exception:  # pylint: disable=broad-exception-caught
      pass

  def _log_tensorboard_scalars(
      self,
      label_prefix: str,
      data: dict[str, float | dict[str, float]],
      step: int,
  ):
    """Logs scalar values (flat or nested) to TensorBoard under a label prefix."""
    if self._writer is None:
      return

    for data_type, data_value in data.items():
      if isinstance(data_value, dict):
        for subtype, subval in data_value.items():
          full_label = f'{label_prefix}/{data_type}/{subtype}'.lower()
          self._writer.add_scalar(
              full_label, float(subval), step, display_name=subtype.lower()
          )
      else:
        full_label = f'{label_prefix}/{data_type.lower()}'
        self._writer.add_scalar(
            full_label, float(data_value), step, display_name=data_type.lower()
        )

    self._writer.flush()

  def _write_goodput_and_badput_data_to_tensorboard(
      self,
      job_goodput: float,
      badput_breakdown: dict[BadputType, float],
      last_step: int,
  ):
    """Writes goodput and badput breakdown to Tensorboard."""
    self._write_goodput_to_tensorboard(job_goodput, last_step)
    if self._include_badput_breakdown:
      self._write_badput_to_tensorboard(badput_breakdown, last_step)

  def _write_goodput_to_tensorboard(self, job_goodput: float, last_step: int):
    self._log_tensorboard_scalars(
        _TENSORBOARD_GOODPUT_LABEL,
        {_TENSORBOARD_GOODPUT_LABEL: job_goodput},
        last_step,
    )

  def _write_badput_to_tensorboard(
      self,
      job_badput_breakdown: dict[BadputType, float | dict[str, float]],
      last_step: int,
  ):
    """Writes badput breakdown to TensorBoard."""
    flattened_badput: dict[str, float | dict[str, float]] = {}

    for badput_type, badput_value in job_badput_breakdown.items():
      if isinstance(badput_value, dict):
        flattened_badput[badput_type.name.lower()] = {
            subtype.lower(): value for subtype, value in badput_value.items()
        }
      else:
        flattened_badput[badput_type.name.lower()] = badput_value

    self._log_tensorboard_scalars(
        _TENSORBOARD_BADPUT_LABEL,
        flattened_badput,
        last_step,
    )

  def _query_and_upload_goodput_to_tensorboard(self):
    """Queries and uploads goodput data to Tensorboard."""
    try:
      job_goodput, job_badput_breakdown, last_step = (
          self._goodput_calculator.get_job_goodput(
              include_badput_breakdown=self._include_badput_breakdown
          )
      )
      self._write_goodput_and_badput_data_to_tensorboard(
          job_goodput, job_badput_breakdown, last_step
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while querying and uploading goodput to Tensorboard. This'
          ' will not impact the workload. Error: %s',
          e,
      )

  def _flatten_badput_dict(
      self,
      badput_time_dict: dict[BadputType, float | dict[str, float]],
  ) -> list[tuple[str, float]]:
    """Flattens nested badput types into (label, value) pairs for export."""
    flat_badput = []
    for badput_type, val in badput_time_dict.items():
      if isinstance(val, dict):
        for subtype, subval in val.items():
          flat_badput.append((f'{badput_type.name}.{subtype.upper()}', subval))
      else:
        flat_badput.append((badput_type.name, val))
    return flat_badput

  def _send_goodput_metrics_to_gcp(self, goodput_details):
    """Sends goodput and badput metrics to GCP Monitoring."""
    try:
      gcp_goodput_metrics = []

      for goodput_type, time_value in goodput_details[
          _GOODPUT_DETAILS_KEY
      ].items():
        if goodput_type.name in ACTIVITY_EXCLUSION_LIST:
          continue
        gcp_goodput_metrics.append({
            'metric_type': 'compute.googleapis.com/workload/goodput_time',
            'value': time_value,
            'value_type': ValueType.DOUBLE,
            'metric_labels': {
                'goodput_source': goodput_type.name,
                'accelerator_type': self._gcp_options.acc_type,
            },
            'resource_type': 'compute.googleapis.com/Workload',
            'resource_labels': {
                'location': self._gcp_options.location,
                'workload_id': self._job_name,
                'replica_id': self._gcp_options.replica_id,
            },
        })
      for badput_label, time_value in self._flatten_badput_dict(
          goodput_details[_BADPUT_DETAILS_KEY]
      ):
        if badput_label in ACTIVITY_EXCLUSION_LIST:
          continue
        gcp_goodput_metrics.append({
            'metric_type': 'compute.googleapis.com/workload/badput_time',
            'value': time_value,
            'value_type': ValueType.DOUBLE,
            'metric_labels': {
                'badput_source': badput_label,
                'accelerator_type': self._gcp_options.acc_type,
            },
            'resource_type': 'compute.googleapis.com/Workload',
            'resource_labels': {
                'location': self._gcp_options.location,
                'workload_id': self._job_name,
                'replica_id': self._gcp_options.replica_id,
            },
        })
      if self._metrics_sender and gcp_goodput_metrics:
        self._metrics_sender.send_metrics(gcp_goodput_metrics)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while sending goodput metrics to GCP Monitoring. This'
          ' will not impact the workload. Error: %s',
          e,
      )

  def _query_and_upload_goodput(self):
    """Queries and uploads goodput data to Tensorboard."""
    while not self._termination_event.is_set():
      time.sleep(self._upload_interval)
      self._query_and_upload_goodput_to_tensorboard()
      if self._gcp_options.enable_gcp_goodput_metrics:
        self._send_goodput_metrics_to_gcp(
            self._goodput_calculator.get_job_goodput_details()
        )

  def _final_goodput_query_and_upload(self):
    """Performs final goodput query and uploads data to Tensorboard & GCM."""
    logger.info(
        'Final goodput query and upload for job: %s and logger: %s',
        self._job_name,
        self._logger_name,
    )
    try:
      job_goodput, job_badput_breakdown, last_step = (
          self._goodput_calculator.get_job_goodput(
              include_badput_breakdown=self._include_badput_breakdown
          )
      )
      self._write_goodput_and_badput_data_to_tensorboard(
          job_goodput, job_badput_breakdown, last_step
      )
      if self._gcp_options.enable_gcp_goodput_metrics:
        self._send_goodput_metrics_to_gcp(
            self._goodput_calculator.get_job_goodput_details()
        )
      logger.info(
          'Final goodput query and upload for job: %s and logger: %s completed'
          ' with total goodput: %.2f%%, last step: %d',
          self._job_name,
          self._logger_name,
          job_goodput,
          last_step,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while performing final goodput query and upload for job: %s'
          ' and logger: %s.  This will not impact the workload. Error: %s',
          self._job_name,
          self._logger_name,
          e,
      )

  def flush_and_stop_goodput_uploader(self):
    """Stops uploader and performs a final goodput upload."""
    if self._goodput_uploader_thread_running:
      self.stop_goodput_uploader()
    self._final_goodput_query_and_upload()

  def start_goodput_uploader(self):
    """Starts the goodput uploader thread."""
    if self._goodput_uploader_thread_running:
      raise RuntimeError('Goodput uploader thread is already running.')

    self._termination_event.clear()
    self._goodput_upload_thread = threading.Thread(
        target=self._query_and_upload_goodput, daemon=True
    )
    logger.info(
        'Starting goodput query and uploader thread in the background for job:'
        ' %s and logger: %s',
        self._job_name,
        self._logger_name,
    )
    self._goodput_upload_thread.start()
    self._goodput_uploader_thread_running = True

  def stop_goodput_uploader(self):
    """Stops the goodput uploader thread."""
    if not self._goodput_uploader_thread_running:
      raise RuntimeError('Goodput uploader thread is not running.')

    self._termination_event.set()
    if self._goodput_upload_thread is not None:
      logger.info('Waiting for goodput query and uploader thread to complete.')
      self._goodput_upload_thread.join()
      self._goodput_upload_thread = None
    logger.info(
        'Goodput query and uploader thread stopped. No more goodput data will'
        ' be uploaded to Tensorboard or GCP Monitoring.'
    )
    self._goodput_uploader_thread_running = False

  def _write_step_deviation_to_tensorboard(
      self, step_deviation: dict[int, float]
  ):
    if self._writer is not None:
      for step_count, step_deviation in step_deviation.items():
        self._writer.add_scalar(
            _TENSORBOARD_STEP_DEVIATION_LABEL,
            float(step_deviation),
            step_count,
        )
      self._writer.flush()

  def _send_step_deviation_metric_to_gcp(self, step_deviations):
    """Sends step deviation metric to GCP Monitoring."""
    try:
      if not step_deviations:
        logger.warning(
            'Step deviation is empty. This will not impact the workload.'
        )
        return
      avg_step_deviation = sum(step_deviations.values()) / len(step_deviations)

      if math.isnan(avg_step_deviation):
        logger.warning(
            'Step deviation is NaN. This will not impact the workload.'
        )
        return

      perf_metric = [{
          'metric_type': 'compute.googleapis.com/workload/performance',
          'value': avg_step_deviation,
          'value_type': ValueType.DOUBLE,
          'resource_type': 'compute.googleapis.com/Workload',
          'resource_labels': {
              'location': self._gcp_options.location,
              'workload_id': self._job_name,
              'replica_id': self._gcp_options.replica_id,
          },
      }]
      if self._metrics_sender:
        self._metrics_sender.send_metrics(perf_metric)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while sending step deviation to GCP Monitoring.'
          ' This will not impact the workload. Error: %s',
          e,
      )

  def _query_and_upload_step_deviation_to_tensorboard_and_gcp(self):
    """Queries and uploads step deviation data to Tensorboard and GCP Monitoring."""
    try:
      step_deviation = self._goodput_calculator.get_step_deviation(
          self._configured_ideal_step_time
      )
      self._write_step_deviation_to_tensorboard(step_deviation)
      if self._gcp_options.enable_gcp_step_deviation_metrics:
        self._send_step_deviation_metric_to_gcp(step_deviation)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while querying and uploading step deviation to Tensorboard.'
          ' This will not impact the workload. Error: %s',
          e,
      )

  def _query_and_upload_step_deviation(self):
    """Queries and uploads step deviation data to Tensorboard."""
    while not self._step_deviation_termination_event.is_set():
      time.sleep(self._step_deviation_interval_seconds)
      self._query_and_upload_step_deviation_to_tensorboard_and_gcp()

  def _final_step_deviation_query_and_upload(self):
    """Performs final step deviation query and uploads data to Tensorboard & GCM."""
    logger.info(
        'Final step deviation query and upload for job: %s and logger: %s',
        self._job_name,
        self._logger_name,
    )
    try:
      step_deviation = self._goodput_calculator.get_step_deviation(
          self._configured_ideal_step_time
      )
      self._write_step_deviation_to_tensorboard(step_deviation)
      if self._gcp_options.enable_gcp_step_deviation_metrics:
        self._send_step_deviation_metric_to_gcp(step_deviation)
      logger.info(
          'Final step deviation query and upload for job: %s and logger: %s'
          ' completed',
          self._job_name,
          self._logger_name,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while performing final step deviation query and upload for'
          ' job: %s and logger: %s. This will not impact the workload. Error:'
          ' %s',
          self._job_name,
          self._logger_name,
          e,
      )

  def flush_and_stop_step_deviation_uploader(self):
    """Stops uploader and performs a final step deviation upload."""
    if self._step_deviation_uploader_thread_running:
      self.stop_step_deviation_uploader()
    self._final_step_deviation_query_and_upload()

  def start_step_deviation_uploader(self):
    """Starts the step deviation uploader thread."""
    if not self._include_step_deviation:
      logger.info(
          'Step deviation monitoring is disabled. Returning without'
          ' initializing step deviation uploader thread.'
      )
      return

    if self._step_deviation_uploader_thread_running:
      raise RuntimeError('Step deviation uploader thread is already running.')

    self._step_deviation_termination_event.clear()
    self._step_deviation_upload_thread = threading.Thread(
        target=self._query_and_upload_step_deviation, daemon=True
    )
    logger.info(
        'Starting step deviation query and uploader thread in the background'
        ' for job: %s and logger: %s',
        self._job_name,
        self._logger_name,
    )
    self._step_deviation_upload_thread.start()
    self._step_deviation_uploader_thread_running = True

  def stop_step_deviation_uploader(self):
    """Stops the step deviation uploader thread."""
    if not self._step_deviation_uploader_thread_running:
      raise RuntimeError('Step deviation uploader thread is not running.')

    self._step_deviation_termination_event.set()
    if self._step_deviation_upload_thread is not None:
      logger.info(
          'Waiting for step deviation query and uploader thread to complete.'
      )
      self._step_deviation_upload_thread.join()
    logger.info(
        'Step deviation query and uploader thread stopped. No more step'
        ' deviation data will be uploaded to Tensorboard or GCP Monitoring.'
    )
    self._step_deviation_uploader_thread_running = False

  def _query_and_upload_interval_goodput(self):
    """Queries and uploads goodput interval data to Tensorboard."""
    while not self._interval_termination_event.is_set():
      time.sleep(self._upload_interval)
      if self._gcp_options.enable_gcp_goodput_metrics:
        window_end = datetime.datetime.now(datetime.timezone.utc)
        window_start = window_end - datetime.timedelta(
            seconds=self._interval_window_size_seconds
        )
        # Add timezone since deltatime removes it.
        window_start = window_start.replace(tzinfo=datetime.timezone.utc)
        self._send_goodput_metrics_to_gcp(
            self._goodput_calculator.get_job_goodput_interval_details(
                window_start, window_end
            )
        )

  def _final_interval_goodput_query_and_upload(self):
    """Performs final interval goodput query and uploads data to GCM."""
    logger.info(
        'Final interval goodput query and upload for job: %s and logger: %s',
        self._job_name,
        self._logger_name,
    )
    try:
      window_end = datetime.datetime.now(datetime.timezone.utc)
      window_start = window_end - datetime.timedelta(
          seconds=self._interval_window_size_seconds
      )
      # Add timezone since deltatime removes it.
      window_start = window_start.replace(tzinfo=datetime.timezone.utc)
      self._send_goodput_metrics_to_gcp(
          self._goodput_calculator.get_job_goodput_interval_details(
              window_start, window_end
          )
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while performing final interval goodput query and upload for'
          ' job: %s and logger: %s. This will not impact the workload. Error:'
          ' %s',
          self._job_name,
          self._logger_name,
          e,
      )

  def flush_and_stop_interval_goodput_uploader(self):
    """Stops uploader and performs a final interval goodput upload."""
    if self._interval_uploader_thread_running:
      self.stop_goodput_interval_uploader()
    self._final_interval_goodput_query_and_upload()

  def start_goodput_interval_uploader(self, window_size_seconds: float):
    """Starts the goodput uploader thread for a user-specified interval window."""
    if self._interval_uploader_thread_running:
      raise RuntimeError('Goodput interval uploader thread is already running.')

    self._interval_termination_event.clear()
    self._interval_window_size_seconds = window_size_seconds
    self._interval_goodput_upload_thread = threading.Thread(
        target=self._query_and_upload_interval_goodput,
        daemon=True,
    )
    logger.info(
        'Starting goodput interval query and uploader thread in the background'
        ' for job: %s and logger: %s',
        self._job_name,
        self._logger_name,
    )
    self._interval_goodput_upload_thread.start()
    self._interval_uploader_thread_running = True

  def stop_goodput_interval_uploader(self):
    """Stops the goodput uploader thread."""
    if not self._interval_uploader_thread_running:
      raise RuntimeError('Goodput intervaluploader thread is not running.')

    self._interval_termination_event.set()
    if self._interval_goodput_upload_thread is not None:
      logger.info(
          'Waiting for goodput interval query and uploader thread to complete.'
      )
      self._interval_goodput_upload_thread.join()
      self._interval_goodput_upload_thread = None
    logger.info(
        'Goodput interval query and uploader thread stopped. No more goodput'
        ' intervaldata will be uploaded to GCP Monitoring.'
    )
    self._interval_uploader_thread_running = False
