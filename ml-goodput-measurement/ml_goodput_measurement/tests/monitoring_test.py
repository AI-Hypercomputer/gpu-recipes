"""Tests to validate the monitoring module.

This module tests the GoodputMonitor class and its functionality, specifically
the uploading of step deviation, goodput and badput data to Tensorboard.
"""

from unittest import mock

from absl.testing import absltest
from cloud_goodput.ml_goodput_measurement.src import gcp_metrics
from cloud_goodput.ml_goodput_measurement.src import goodput_utils
from cloud_goodput.ml_goodput_measurement.src import monitoring

from google.cloud import monitoring_v3

BadputType = goodput_utils.BadputType
GCPOptions = goodput_utils.GCPOptions
GoodputMonitor = monitoring.GoodputMonitor
GoodputType = goodput_utils.GoodputType
MagicMock = mock.MagicMock
ValueType = gcp_metrics.ValueType

patch = mock.patch
_TEST_UPLOAD_INTERVAL = 1


class GoodputMonitorTests(absltest.TestCase):
  """Tests for the GoodputMonitor class."""

  def setUp(self):
    super().setUp()
    self.job_name = 'test-run'
    self.logger_name = 'test-logger'
    self.tensorboard_dir = 'test-dir'

  def _create_timeseries(
      self, metric_type: str, labels: dict, value: float
  ) -> monitoring_v3.TimeSeries:
    ts = monitoring_v3.TimeSeries()
    ts.metric.type = metric_type
    ts.metric.labels.update(labels)
    ts.resource.type = 'compute.googleapis.com/Workload'
    ts.resource.labels.update({
        'location': 'test-location',
        'workload_id': 'test-run',
        'replica_id': 'test-replica-id',
    })
    ts.points.append(
        monitoring_v3.Point(
            value=monitoring_v3.TypedValue(double_value=value),
        )
    )
    return ts

  def _compare_calls_ignore_time_series(
      self, expected_call, actual_call
  ) -> bool:
    if (
        expected_call.args != actual_call.args
        or expected_call.kwargs.keys() != actual_call.kwargs.keys()
    ):
      return False

    for key, expected_value in expected_call.kwargs.items():
      actual_value = actual_call.kwargs[key]
      if key == 'time_series':
        continue
      if expected_value != actual_value:
        return False

    return True

  def _setup_mock_goodput_monitor(
      self, mock_logging_client, mock_summary_writer, mock_metric_service_client
  ) -> GoodputMonitor:
    mock_client = MagicMock()
    mock_metric_service_client.return_value = mock_client
    mock_logging_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()

    gcp_options = GCPOptions(
        enable_gcp_goodput_metrics=True,
        project_id='test-project',
        location='test-location',
        acc_type='test-acc-type',
        replica_id='test-replica-id',
    )

    return GoodputMonitor(
        job_name='test-run',
        logger_name='test-logger',
        tensorboard_dir='/tmp',
        upload_interval=1,
        monitoring_enabled=True,
        gcp_options=gcp_options,
    )

  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_goodput_monitor_init(self, mock_logger_client, mock_summary_writer):
    mock_summary_writer.return_value = MagicMock()
    mock_logger_client.return_value = MagicMock()
    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
    )
    # Objects should be initialized correctly.
    self.assertIsNotNone(goodput_monitor)
    self.assertIs(goodput_monitor._writer, mock_summary_writer.return_value)
    self.assertIsNotNone(goodput_monitor._goodput_calculator)

    # Thread events should be initialized correctly.
    self.assertIsNotNone(goodput_monitor._step_deviation_termination_event)
    self.assertFalse(goodput_monitor._step_deviation_termination_event.is_set())
    self.assertFalse(goodput_monitor._step_deviation_uploader_thread_running)
    self.assertIsNotNone(goodput_monitor._termination_event)
    self.assertFalse(goodput_monitor._termination_event.is_set())
    self.assertFalse(goodput_monitor._goodput_uploader_thread_running)

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._write_goodput_to_tensorboard'
  )
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  async def test_goodput_monitor_start_goodput_uploader_success(
      self, mock_logger_client, mock_summary_writer, mock_goodput_to_tensorboard
  ):
    mock_summary_writer.return_value = MagicMock()
    mock_goodput_to_tensorboard.return_value = MagicMock()
    mock_logger_client.return_value = MagicMock()
    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
    )
    goodput_monitor.start_goodput_uploader()
    self.assertTrue(goodput_monitor._uploader_thread_running)
    self.assertIsNotNone(goodput_monitor._goodput_upload_thread)
    self.assertFalse(goodput_monitor._termination_event.is_set())
    mock_goodput_to_tensorboard.assert_called_once()
    mock_summary_writer.return_value.add_scalar.assert_called_once()
    goodput_monitor.stop_goodput_uploader()
    self.assertFalse(goodput_monitor._uploader_thread_running)
    self.assertIsNone(goodput_monitor._goodput_upload_thread)
    self.assertTrue(goodput_monitor._termination_event.is_set())

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._write_goodput_to_tensorboard'
  )
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  async def test_goodput_monitor_start_goodput_uploader_failure(
      self, mock_logger_client, mock_summary_writer, mock_goodput_to_tensorboard
  ):
    mock_logger_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()
    mock_goodput_to_tensorboard.side_effect = ValueError('Test Error')
    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
    )
    goodput_monitor.start_goodput_uploader()
    self.assertTrue(goodput_monitor._uploader_thread_running)
    self.assertIsNotNone(goodput_monitor._goodput_upload_thread)
    self.assertFalse(goodput_monitor._termination_event.is_set())
    mock_goodput_to_tensorboard.assert_called_once()
    with self.assertRaisesRegex(ValueError, 'Test Error'):
      goodput_monitor._query_and_upload_goodput()
    mock_summary_writer.return_value.add_scalar.assert_not_called()
    goodput_monitor.stop_goodput_uploader()
    self.assertFalse(goodput_monitor._uploader_thread_running)
    self.assertIsNone(goodput_monitor._goodput_upload_thread)
    self.assertTrue(goodput_monitor._termination_event.is_set())

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._write_badput_to_tensorboard'
  )
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  async def test_goodput_monitor_start_badput_uploader_success(
      self, mock_logger_client, mock_summary_writer, mock_badput_to_tensorboard
  ):
    mock_summary_writer.return_value = MagicMock()
    mock_badput_to_tensorboard.return_value = MagicMock()
    mock_logger_client.return_value = MagicMock()
    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        include_badput_breakdown=True,
    )

    goodput_monitor.start_goodput_uploader()
    self.assertTrue(goodput_monitor._uploader_thread_running)
    self.assertIsNotNone(goodput_monitor._goodput_upload_thread)
    self.assertFalse(goodput_monitor._termination_event.is_set())
    self.assertTrue(goodput_monitor._include_badput_breakdown)

    mock_badput_to_tensorboard.assert_called_once()
    mock_summary_writer.return_value.add_scalar.assert_called_once()

    goodput_monitor.stop_goodput_uploader()
    self.assertFalse(goodput_monitor._uploader_thread_running)
    self.assertIsNone(goodput_monitor._goodput_upload_thread)
    self.assertTrue(goodput_monitor._termination_event.is_set())

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._write_step_deviation_to_tensorboard'
  )
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  async def test_goodput_monitor_start_step_deviation_uploader_success(
      self,
      mock_logger_client,
      mock_summary_writer,
      mock_step_deviation_to_tensorboard,
  ):
    mock_logger_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()
    mock_step_deviation_to_tensorboard.return_value = MagicMock()
    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        include_step_deviation=True,
    )
    goodput_monitor.start_step_deviation_uploader()
    self.assertTrue(goodput_monitor._step_deviation_uploader_thread_running)
    self.assertIsNotNone(goodput_monitor._step_deviation_upload_thread)
    self.assertFalse(goodput_monitor._step_deviation_termination_event.is_set())
    mock_step_deviation_to_tensorboard.assert_called_once()
    mock_summary_writer.return_value.add_scalar.assert_called_once()
    goodput_monitor.stop_step_deviation_uploader()
    self.assertFalse(goodput_monitor._step_deviation_uploader_thread_running)
    self.assertIsNone(goodput_monitor._step_deviation_upload_thread)
    self.assertTrue(goodput_monitor._step_deviation_termination_event.is_set())

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._write_step_deviation_to_tensorboard'
  )
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  async def test_goodput_monitor_start_step_deviation_uploader_failure(
      self,
      mock_logger_client,
      mock_summary_writer,
      mock_query_and_upload_step_deviation,
  ):
    mock_logger_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()
    mock_query_and_upload_step_deviation.side_effect = ValueError('Test Error')
    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        include_step_deviation=True,
    )
    goodput_monitor.start_step_deviation_uploader()
    self.assertTrue(goodput_monitor._step_deviation_uploader_thread_running)
    self.assertIsNotNone(goodput_monitor._step_deviation_upload_thread)
    self.assertFalse(goodput_monitor._step_deviation_termination_event.is_set())
    mock_query_and_upload_step_deviation.assert_called_once()
    with self.assertRaisesRegex(ValueError, 'Test Error'):
      goodput_monitor._query_and_upload_step_deviation()
    mock_summary_writer.return_value.add_scalar.assert_not_called()
    goodput_monitor.stop_step_deviation_uploader()
    self.assertFalse(goodput_monitor._step_deviation_uploader_thread_running)
    self.assertIsNone(goodput_monitor._step_deviation_upload_thread)
    self.assertTrue(goodput_monitor._step_deviation_termination_event.is_set())

  @patch('google.cloud.monitoring_v3.MetricServiceClient')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_send_goodput_metrics_to_gcp_success(
      self,
      mock_logging_client,
      mock_summary_writer,
      mock_metric_service_client,
  ):
    mock_client = MagicMock()
    mock_metric_service_client.return_value = mock_client
    mock_logging_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()

    gcp_options = GCPOptions(
        enable_gcp_goodput_metrics=True,
        project_id='test-project',
        location='test-location',
        acc_type='test-acc-type',
        replica_id='test-replica-id',
    )

    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        gcp_options=gcp_options,
    )

    # Mock the get_job_goodput_details to return test data
    goodput_monitor._goodput_calculator.get_job_goodput_details = MagicMock(
        return_value={
            'goodput_time_dict': {
                GoodputType.TOTAL: 10.0,
            },
            'badput_time_dict': {
                BadputType.TPU_INITIALIZATION: 2.0,
                BadputType.DATA_LOADING_SYNC: 1.0,
            },
        }
    )

    goodput_monitor._send_goodput_metrics_to_gcp(
        goodput_monitor._goodput_calculator.get_job_goodput_details()
    )

    expected_calls = [
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/goodput_time',
                    {
                        'goodput_source': 'TOTAL',
                        'accelerator_type': 'test-acc-type',
                    },
                    10.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'TPU_INITIALIZATION',
                        'accelerator_type': 'test-acc-type',
                    },
                    2.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'DATA_LOADING_SYNC',
                        'accelerator_type': 'test-acc-type',
                    },
                    1.0,
                )
            ],
        ),
    ]

    actual_calls = mock_client.create_time_series.call_args_list

    # Verify each call individually
    for expected_call in expected_calls:
      self.assertTrue(
          any(
              self._compare_calls_ignore_time_series(expected_call, actual)
              for actual in actual_calls
          ),
          f'Expected call not found: {expected_call}',
      )

  @patch('google.cloud.monitoring_v3.MetricServiceClient')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_send_goodput_metrics_to_gcp_exception(
      self,
      mock_logging_client,
      mock_summary_writer,
      mock_metric_service_client,
  ):
    mock_client = MagicMock()
    mock_client.create_time_series.side_effect = Exception('Test Exception')
    mock_metric_service_client.return_value = mock_client
    mock_logging_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()

    gcp_options = GCPOptions(
        enable_gcp_goodput_metrics=True,
        project_id='test-project',
        location='test-location',
        acc_type='test-acc-type',
        replica_id='test-replica-id',
    )

    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        gcp_options=gcp_options,
    )

    # Mock the get_job_goodput_details to return test data
    goodput_monitor._goodput_calculator.get_job_goodput_details = MagicMock(
        return_value={
            'goodput_time_dict': {
                GoodputType.TOTAL: 10.0,
            },
            'badput_time_dict': {
                BadputType.DATA_LOADING_SYNC: 2.0,
            },
        }
    )

    goodput_monitor._send_goodput_metrics_to_gcp(
        goodput_monitor._goodput_calculator.get_job_goodput_details()
    )

    # Verify that create_time_series was called, even if it raised an exception
    mock_client.create_time_series.assert_called_once()

  @patch('google.cloud.monitoring_v3.MetricServiceClient')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_send_goodput_metrics_to_gcp_exclusion(
      self,
      mock_logging_client,
      mock_summary_writer,
      mock_metric_service_client
  ):
    mock_client = MagicMock()
    mock_metric_service_client.return_value = mock_client
    mock_logging_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()

    gcp_options = GCPOptions(
        enable_gcp_goodput_metrics=True,
        project_id='test-project',
        location='test-location',
        acc_type='test-acc-type',
        replica_id='test-replica-id',
    )

    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        gcp_options=gcp_options,
    )

    # Mock the get_job_goodput_details to return test data, including an
    # excluded type
    goodput_monitor._goodput_calculator.get_job_goodput_details = MagicMock(
        return_value={
            'goodput_time_dict': {
                GoodputType.TOTAL: 10.0,
            },
            'badput_time_dict': {
                BadputType.TPU_INITIALIZATION: 2.0,
                BadputType.DATA_LOADING_SYNC: 1.0,
                BadputType.DATA_LOADING_ASYNC: (
                    3.0
                ),  # DATA_LOADING_ASYNC is in ACTIVITY_EXCLUSION_LIST
            },
        }
    )

    goodput_monitor._send_goodput_metrics_to_gcp(
        goodput_monitor._goodput_calculator.get_job_goodput_details()
    )

    # Verify that create_time_series was called with the correct data,
    # excluding DATA_LOADING_ASYNC
    expected_calls = [
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/goodput_time',
                    {
                        'goodput_source': 'TOTAL',
                        'accelerator_type': 'test-acc-type',
                    },
                    10.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'TPU_INITIALIZATION',
                        'accelerator_type': 'test-acc-type',
                    },
                    2.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'DATA_LOADING_SYNC',
                        'accelerator_type': 'test-acc-type',
                    },
                    1.0,
                )
            ],
        ),
    ]

    actual_calls = mock_client.create_time_series.call_args_list

    # Verify each call individually
    for expected_call in expected_calls:
      self.assertTrue(
          any(
              self._compare_calls_ignore_time_series(expected_call, actual)
              for actual in actual_calls
          ),
          f'Expected call not found: {expected_call}',
      )
    # Verify unexpected calls are not made
    for actual_call in actual_calls:
      for ts in actual_call.kwargs.get('time_series', []):
        if (
            'badput_source' in ts.metric.labels
            and ts.metric.labels['badput_source'] == 'DATA_LOADING_ASYNC'
        ):
          self.fail(f'Unexpected call found: {ts}')

  @patch('google.cloud.monitoring_v3.MetricServiceClient')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_send_interval_goodput_metrics_to_gcp(
      self,
      mock_logging_client,
      mock_summary_writer,
      mock_metric_service_client,
  ):
    mock_client = MagicMock()
    mock_metric_service_client.return_value = mock_client
    mock_logging_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()

    gcp_options = GCPOptions(
        enable_gcp_goodput_metrics=True,
        project_id='test-project',
        location='test-location',
        acc_type='test-acc-type',
        replica_id='test-replica-id',
    )

    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        gcp_options=gcp_options,
    )

    # Mock the get_job_goodput_details to return test data
    goodput_monitor._goodput_calculator.get_job_goodput_interval_details = (
        MagicMock(
            return_value={
                'goodput_time_dict': {
                    GoodputType.TOTAL: 10.0,
                },
                'badput_time_dict': {
                    BadputType.TPU_INITIALIZATION: 2.0,
                    BadputType.DATA_LOADING_SYNC: 1.0,
                },
            }
        )
    )

    goodput_monitor._send_goodput_metrics_to_gcp(
        goodput_monitor._goodput_calculator.get_job_goodput_interval_details()
    )

    expected_calls = [
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/goodput_time',
                    {
                        'goodput_source': 'TOTAL',
                        'accelerator_type': 'test-acc-type',
                    },
                    10.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'TPU_INITIALIZATION',
                        'accelerator_type': 'test-acc-type',
                    },
                    2.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'DATA_LOADING_SYNC',
                        'accelerator_type': 'test-acc-type',
                    },
                    1.0,
                )
            ],
        ),
    ]

    actual_calls = mock_client.create_time_series.call_args_list

    # Verify each call individually
    for expected_call in expected_calls:
      self.assertTrue(
          any(
              self._compare_calls_ignore_time_series(expected_call, actual)
              for actual in actual_calls
          ),
          f'Expected call not found: {expected_call}',
      )

  @patch('google.cloud.monitoring_v3.MetricServiceClient')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_send_goodput_metrics_custom_sync_events(
      self, mock_logging_client, mock_summary_writer, mock_metric_service_client
  ):
    mock_client = MagicMock()
    mock_metric_service_client.return_value = mock_client
    mock_logging_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()

    gcp_options = GCPOptions(
        enable_gcp_goodput_metrics=True,
        project_id='test-project',
        location='test-location',
        acc_type='test-acc-type',
        replica_id='test-replica-id',
    )

    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        gcp_options=gcp_options,
    )

    # Mock the get_job_goodput_details to return test data, including an
    # excluded type
    goodput_monitor._goodput_calculator.get_job_goodput_details = MagicMock(
        return_value={
            'goodput_time_dict': {
                GoodputType.TOTAL: 10.0,
            },
            'badput_time_dict': {
                BadputType.TPU_INITIALIZATION: 2.0,
                BadputType.DATA_LOADING_SYNC: 1.0,
                BadputType.CUSTOM_BADPUT_EVENTS: {
                    'EVAL_STEP': 3.0,
                    'SDC_COMPILATION': 4.0,
                },
            },
        }
    )

    goodput_monitor._send_goodput_metrics_to_gcp(
        goodput_monitor._goodput_calculator.get_job_goodput_details()
    )

    expected_calls = [
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/goodput_time',
                    {
                        'goodput_source': 'TOTAL',
                        'accelerator_type': 'test-acc-type',
                    },
                    10.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'TPU_INITIALIZATION',
                        'accelerator_type': 'test-acc-type',
                    },
                    2.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'DATA_LOADING_SYNC',
                        'accelerator_type': 'test-acc-type',
                    },
                    1.0,
                )
            ],
        ),
    ]

    actual_calls = mock_client.create_time_series.call_args_list

    # Verify each call individually
    for expected_call in expected_calls:
      self.assertTrue(
          any(
              self._compare_calls_ignore_time_series(expected_call, actual_call)
              for actual_call in actual_calls
          ),
          f'Expected call not found: {expected_call}',
      )

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._final_interval_goodput_query_and_upload'
  )
  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._final_step_deviation_query_and_upload'
  )
  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._final_goodput_query_and_upload'
  )
  async def test_goodput_monitor_final_query_and_upload(
      self,
      mock_final_goodput_query_and_upload,
      mock_final_step_deviation_query_and_upload,
      mock_final_interval_goodput_query_and_upload,
  ):
    mock_final_goodput_query_and_upload.return_value = MagicMock()
    mock_final_step_deviation_query_and_upload.return_value = MagicMock()
    mock_final_interval_goodput_query_and_upload.return_value = MagicMock()
    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
    )
    goodput_monitor.__del__()
    mock_final_goodput_query_and_upload.assert_called_once()
    mock_final_step_deviation_query_and_upload.assert_called_once()
    mock_final_interval_goodput_query_and_upload.assert_called_once()


if __name__ == '__main__':
  absltest.main()
