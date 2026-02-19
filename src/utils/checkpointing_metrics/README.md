# Checkpoint statistics calculator

This Python utility calculates checkpoint write time statistics from NVIDIA NeMo 2 log files.

## Usage

```
python calculate_checkpoint_metrics.py --gcs_logs_path <path_to_logs>

```
### Required arguments

- `--gcs_logs_path`: The path to NeMo logs in a GCS bucket. E.g. `gs://logs_bucket/experiment_name/experiment_version`

### Sample output

```
$ python calculate_checkpoint_metrics.py \
    --gcs_logs_path gs://tess-benchmark-outputs/muzi-8b-dl-ckpt-20260217-175559
Analyzing file: muzi-8b-dl-ckpt-20260217-175559/run_0/nemo_log_globalrank-2_localrank-2.txt, Global rank: 2, Local rank: 2
Analyzing file: muzi-8b-dl-ckpt-20260217-175559/nemo_log_globalrank-1_localrank-1.txt, Global rank: 1, Local rank: 1
Analyzing file: muzi-8b-dl-ckpt-20260217-175559/nemo_log_globalrank-0_localrank-0.txt, Global rank: 0, Local rank: 0
Analyzing file: muzi-8b-dl-ckpt-20260217-175559/nemo_log_globalrank-3_localrank-3.txt, Global rank: 3, Local rank: 3
min checkpoint write duration: 4.9700000286102295s
max checkpoint write duration: 34.86500000953674s
average checkpoint write duration: 17.880999982357025s
checkpoint write time standard deviation: 12.443055009810607
```

### Dependencies

The utility uses the `google-cloud-storage` Python package. You can install the package to your Python environment using the following command.

```
pip install google-cloud-storage
```

## Testing

```
python3 -m unittest calculate_checkpoint_metrics_test -v
```