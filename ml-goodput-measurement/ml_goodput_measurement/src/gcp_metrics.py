"""A generic class to send multiple metrics to GCP Cloud Monitoring in a batch with dynamic resources."""
import enum
import logging
import time
from typing import Any, Dict

from google.api_core import exceptions
from google.cloud import monitoring_v3

GoogleAPIError = exceptions.GoogleAPIError
Enum = enum.Enum

logger = logging.getLogger(__name__)


class ValueType(Enum):
  """Enum for metric value types."""

  BOOL = "bool_value"
  INT = "int64_value"
  DOUBLE = "double_value"
  STRING = "string_value"
  DISTRIBUTION = "distribution_value"  # Add other types as needed


class GCPMetrics:
  """A generic class to send multiple metrics to GCP Cloud Monitoring in a batch with dynamic resources."""

  def __init__(self, project_id: str):
    """Initializes the GCPMetrics."""
    self.project_id = project_id
    self.client = monitoring_v3.MetricServiceClient()
    self.project_name = f"projects/{project_id}"

  def create_time_series(
      self,
      metric_type: str,
      value,
      value_type: ValueType,
      metric_labels: Dict[str, str],
      resource_type: str,
      resource_labels: Dict[str, str],
      seconds: int,
      nanos: int,
  ) -> monitoring_v3.TimeSeries:
    """Creates a TimeSeries object for a single metric with dynamic resources."""
    series = monitoring_v3.TimeSeries()
    series.metric.type = metric_type
    series.resource.type = resource_type
    series.resource.labels.update(resource_labels)
    if metric_labels:
      series.metric.labels.update(metric_labels)

    point = monitoring_v3.Point(
        interval=monitoring_v3.TimeInterval(
            end_time={"seconds": seconds, "nanos": nanos}
        ),
        value=monitoring_v3.TypedValue(**{value_type.value: value}),
    )
    series.points.append(point)

    return series

  def send_metrics(self, metrics: list[Dict[str, Any]]):
    """Sends multiple metrics to GCP Monitoring in a batch with dynamic resources.

    Args:
        metrics: A list of dictionaries, where each dictionary represents
                 a metric. Each dictionary should have the following keys:
                 - 'metric_type': str
                 - 'value': The metric value.
                 - 'value_type': ValueType (e.g., ValueType.INT,
                   ValueType.DOUBLE)
                 - 'metric_labels': dict (optional)
                 - 'resource_type': str
                 - 'resource_labels': dict
    """
    try:
      now = time.time()
      seconds = int(now)
      nanos = int((now - seconds) * 10**9)

      time_series_list = []
      for metric in metrics:
        try:
          metric_labels = metric.get("metric_labels", {})
          series = self.create_time_series(
              metric["metric_type"],
              metric["value"],
              metric["value_type"],
              metric_labels,
              metric["resource_type"],
              metric["resource_labels"],
              seconds,
              nanos,
          )
          time_series_list.append(series)
        except Exception as e:  # pylint: disable=broad-exception-caught
          logger.error("Failed to create time series: %s", e)
      self.client.create_time_series(
          name=self.project_name, time_series=time_series_list
      )
      logger.info("Sent %d metrics to GCP Monitoring.", len(metrics))

    except GoogleAPIError as e:
      logger.error("Failed to send metrics: %s", e)
