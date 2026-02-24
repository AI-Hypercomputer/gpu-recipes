# Checkpoint statistics calculator

This Python utility calculates checkpoint write time statistics from NVIDIA log files. It supports **multiple log formats** through a plugin-based parser architecture.

## Supported Log Formats

| Format | Flag Value | Description |
|---|---|---|
| NeMo 2 | `nemo2` (default) | NeMo 2.x checkpoint logs with `Global Checkpoint Save` messages |
| NeMo 1 | `nemo1` | NeMo 1.x checkpoint logs with `Checkpoint save for step` messages |

## Usage

```
python calculate_checkpoint_metrics.py --gcs_logs_path <path_to_logs> [--log_format auto|nemo1|nemo2]
```

### Required arguments

- `--gcs_logs_path`: The path to NeMo logs in a GCS bucket. E.g. `gs://logs_bucket/experiment_name/experiment_version`

### Optional arguments

- `--log_format`: The log format to parse. Choices: `auto`, `nemo1`, `nemo2`. Default: `auto` (auto-detects from log content).

### Examples

**NeMo 2 logs** (default):
```
python calculate_checkpoint_metrics.py \
    --gcs_logs_path gs://tess-benchmark-outputs/muzi-8b-dl-ckpt-20260217-175559
```

**NeMo 1 logs**:
```
python calculate_checkpoint_metrics.py \
    --gcs_logs_path gs://tess-benchmark-outputs/nemo1-experiment \
    --log_format nemo1
```

### Sample output

```
> python calculate_checkpoint_metrics.py \
    --gcs_logs_path gs://tess-benchmark-outputs/muzi-8b-dl-ckpt-20260217-175559
Analyzing file: muzi-8b-dl-ckpt-20260217-175559/nemo_log_globalrank-1_localrank-1.txt, Global rank: 1, Local rank: 1
Auto-detected log format: nemo2
Analyzing file: muzi-8b-dl-ckpt-20260217-175559/run_0/nemo_log_globalrank-2_localrank-2.txt, Global rank: 2, Local rank: 2
Auto-detected log format: nemo2
Analyzing file: muzi-8b-dl-ckpt-20260217-175559/nemo_log_globalrank-0_localrank-0.txt, Global rank: 0, Local rank: 0
Auto-detected log format: nemo2
Analyzing file: muzi-8b-dl-ckpt-20260217-175559/nemo_log_globalrank-3_localrank-3.txt, Global rank: 3, Local rank: 3
Auto-detected log format: nemo2
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

## Adding a New Log Format

To add support for a new framework:

1. Create `<framework>_parser.py` in the `checkpointing_metrics/` directory
2. Subclass `LogParser` from `log_parser.py`
3. Implement the abstract methods (regex patterns, time extraction, step normalization)
4. Decorate the class with `@register_parser`
5. Add an import in `calculate_checkpoint_metrics.py` to trigger registration
6. Add tests — no changes needed to core logic

## Testing

```
python3 -m unittest calculate_checkpoint_metrics_test -v
```