# Checkpoint statistics calculator

This Python utility calculates checkpoint write time statistics from NVIDIA NeMo log files.

## Usage

```
python calculate_checkpoint_metrics.py --gcs_logs_path <path_to_logs>

```
### Required arguments

- `--gcs_logs_path`: The path to NeMo logs in a GCS bucket. E.g. `gs://logs_bucket/experiment_name/experiment_version`

### Dependencies

The utility uses the `google-cloud-storage` Python package. You can install the package to your Python environment using the following command.

```
pip install google-cloud-storage`
```