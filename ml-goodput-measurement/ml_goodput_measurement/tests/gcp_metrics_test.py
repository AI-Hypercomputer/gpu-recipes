"""Tests for GCP metrics."""

from unittest import mock

from absl.testing import absltest
from cloud_goodput.ml_goodput_measurement.src import gcp_metrics
from google.api_core import exceptions
from google.cloud import monitoring_v3


ValueType = gcp_metrics.ValueType
GCPMetrics = gcp_metrics.GCPMetrics
patch = mock.patch
GoogleAPIError = exceptions.GoogleAPIError


class GCPMetricsTest(absltest.TestCase):

  @patch("google.cloud.monitoring_v3.MetricServiceClient")
  def setUp(self, mock_client):
    super().setUp()
    self.mock_client = mock_client.return_value
    self.project_id = "test-project"
    self.metrics_sender = GCPMetrics(self.project_id)

  def test_create_time_series(self):
    metric_type = "compute.googleapis.com/workload/goodput_time"
    value = 123.45
    value_type = ValueType.DOUBLE
    metric_labels = {
        "goodput_source": "TOTAL",
        "accelerator_type": "tpu-v5p",
    }
    resource_type = "compute.googleapis.com/Workload"
    resource_labels = {
        "location": "us-central1",
        "workload_id": "test-workload",
        "replica_id": "0",
    }
    seconds = 1677347200
    nanos = 123456789

    time_series = self.metrics_sender.create_time_series(
        metric_type,
        value,
        value_type,
        metric_labels,
        resource_type,
        resource_labels,
        seconds,
        nanos,
    )

    # Assertions to check if the TimeSeries object is created correctly
    self.assertIsInstance(time_series, monitoring_v3.TimeSeries)
    self.assertEqual(time_series.metric.type, metric_type)
    self.assertEqual(time_series.resource.type, resource_type)
    self.assertEqual(time_series.resource.labels, resource_labels)
    self.assertEqual(time_series.metric.labels, metric_labels)

    # Correctly check the value based on value_type
    if value_type == ValueType.BOOL:
      self.assertEqual(time_series.points[0].value.bool_value, value)
    elif value_type == ValueType.INT:
      self.assertEqual(time_series.points[0].value.int64_value, value)
    elif value_type == ValueType.DOUBLE:
      self.assertEqual(time_series.points[0].value.double_value, value)
    elif value_type == ValueType.STRING:
      self.assertEqual(time_series.points[0].value.string_value, value)
    elif value_type == ValueType.DISTRIBUTION:
      self.assertEqual(
          time_series.points[0].value.distribution_value, value
      )

  @patch("time.time")
  def test_send_metrics(self, mock_time):
    # Set a fixed return value for the mocked time.time()
    mock_time.return_value = 1677347200.5

    metrics_to_send = [
        {
            "metric_type": "compute.googleapis.com/workload/goodput_time",
            "value": 42.0,
            "value_type": ValueType.DOUBLE,
            "resource_type": "test_resource",
            "resource_labels": {"loc": "us"},
        },
        {
            "metric_type": "compute.googleapis.com/workload/badput_time",
            "value": 10,
            "value_type": ValueType.INT,
            "metric_labels": {"source": "test2"},
            "resource_type": "test_resource",
            "resource_labels": {"loc": "eu"},
        },
    ]

    self.metrics_sender.send_metrics(metrics_to_send)

    # Verify that create_time_series was called with the correct arguments
    expected_name = f"projects/{self.project_id}"
    expected_calls = []
    for metric in metrics_to_send:
      metric_labels = metric.get("metric_labels", {})
      series = self.metrics_sender.create_time_series(
          metric["metric_type"],
          metric["value"],
          metric["value_type"],
          metric_labels,
          metric["resource_type"],
          metric["resource_labels"],
          1677347200,  # seconds
          500000000,  # nanos
      )
      expected_calls.append(series)

    self.mock_client.create_time_series.assert_called_once()
    _, kwargs = self.mock_client.create_time_series.call_args
    self.assertEqual(kwargs["name"], expected_name)
    # Check time series
    actual_series = kwargs["time_series"]
    self.assertEqual(len(actual_series), len(expected_calls))
    for actual, expected in zip(actual_series, expected_calls):
      self.assertEqual(actual.metric.type, expected.metric.type)
      self.assertEqual(actual.resource.type, expected.resource.type)
      self.assertEqual(actual.resource.labels, expected.resource.labels)
      self.assertEqual(actual.metric.labels, expected.metric.labels)

  @patch("cloud_goodput.ml_goodput_measurement.src.gcp_metrics.logger.error")
  def test_send_metrics_failure(self, mock_logging_error):

    self.mock_client.create_time_series.side_effect = GoogleAPIError(
        "Test Error"
    )

    metrics_to_send = [
        {
            "metric_type": "compute.googleapis.com/workload/goodput_time",
            "value": 42.0,
            "value_type": ValueType.DOUBLE,
            "resource_type": "test_resource",
            "resource_labels": {"loc": "us"},
        }
    ]

    self.metrics_sender.send_metrics(metrics_to_send)
    mock_logging_error.assert_called_once()

if __name__ == "__main__":
  absltest.main()
