<!--
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->
# ML Goodput Measurement

## Overview

 ML Goodput Measurement is a library intended to be used with Cloud accelerators
 to log necessary information and query a job's Goodput and Badput Breakdown. It
 can be pip installed to import its modules, and retrieve information about a
 training job's overall productive Goodput and sources of Badput. The package
 exposes API interfaces to log useful information from the user application and
 query Goodput for the job run, gain insight into the productivity of ML
 workloads and utilization of compute resources.

 The package also exposes Goodput Monitoring APIs which allow asynchronous query
 and export of the job's Goodput, Badput and Step Time Deviation to Tensorboard
 with configurable upload interval.

## Components


 The ML Goodput Measurement library consists of the following main components: 

  - `GoodputRecorder`

  - `GoodputCalculator`
  - `GoodputMonitor`
  - `GoodputCache`


 The `GoodputRecorder`
 exposes APIs to the client to export key timestamps while a training job makes
 progress, namely APIs that allow logging of productive step time and total job
 run time. The library will serialize and store this data in Google Cloud
 Logging.

 The `GoodputCalculator` exposes APIs to compute Goodput based on the
 recorded data. Cloud Logging handles its internal operations asynchronously.
 The recommended way to compute Goodput is to run an analysis program separate
 from the training application, either on a CPU instance or on the users'
 development machine.

 Under the hood, the `GoodputCalculator` uses a `GoodputCache` which is an
 internal component that locally caches pre-computations and useful logs such
 that repeated computations can be made inexpensive.

 The `GoodputMonitor` exposes APIs to query and upload goodput and step time
 deviation data to Tensorboard asynchronously. It does this by instantiating a
 `GoodputCaluclator` under the hood.

## Installation

 To install the ML Goodput Measurement package, run the following command on the
 VM or machine you want to query or monitor your workload from:

 ```bash
 pip install ml-goodput-measurement
 ```

## Usage

The usage of this package requires the setup of a Google Cloud project with
billing enabled to properly use Google Cloud Logging. If you don't have a Google
Cloud project, or if you don't have billing enabled for your Google Cloud
project, then do the following:

1. In the Google Cloud console, on the project selector page,
 [select or create a Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects).

2. Make sure that billing is enabled for your Google Cloud project. Instructions can be found [here](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled#console)

3. [Enable](https://console.cloud.google.com/flows/enableapi?apiid=logging.googleapis.com&_ga=2.27841276.1571868865.1726250448-123998259.1726107009) the Cloud Logging API.

  To run your training on Cloud accelerator, set up the environment by following
  instructions [here](https://cloud.google.com/tpu/docs/setup-gcp-account).

  To learn more about Google Cloud Logging, visit this [page](https://cloud.google.com/logging/docs).

### Access Scopes

 You will need both read and write access scopes for cloud logging on both the
 GPU or TPU and CPU node pools. Full cloud logging access is granted by the
 following access scope during node pool creation:

  - `https://www.googleapis.com/auth/cloud-platform`

   XPK adds this access scope to the GPU, TPU and CPU node pools, so XPK is the
   recommended method to create clusters and node-pools in you intend to run
   your workloads on GKE.

   Instructions on how to create clusters using XPK can be
   found [here](https://github.com/AI-Hypercomputer/xpk/blob/main/README.md#cluster-create)
   and how to create workloads using XPK can be found
   [here](https://github.com/AI-Hypercomputer/xpk/blob/main/README.md#workload-create).

   > **_NOTE:_** Access Scopes are immutable and workloads can only be migrated
  to new node pools with required access scopes. Access scopes on already created
  clusters cannot be updated.

### Import

 To use this package, import the `goodput` module:

 ```python
 from ml_goodput_measurement import goodput
 from ml_goodput_measurement import monitoring
 ```

### Define the name of the Google Cloud Logging logger.

 Create a run-specific logger name where Cloud Logging entries can be written to
 and read from.

 > **IMPORTANT:** Please use a unique `run_name` for each individual experiment
 or workload that you intend to monitor separately. If you unintentionally re-use
 `run_name` or `goodput_logger_name` in the same storage bucket of a GCP project,
 your cumulative Goodput metrics may be inaccurately taking previous runs into
 account.

 For example:

 ```python
 goodput_logger_name = f'goodput_{config.run_name}' # Here run_name is unique.
 ```

### Create a `GoodputRecorder` object

 Next, create a recorder object with the following parameters:

 1. `job_name`: The full run name of the job.
 2. `logger_name`: The name of the Cloud Logging logger object (created in the previous step).
 3. `logging_enabled`: Whether or not this process has Cloud Logging enabled.

  > **_NOTE:_** For a multi-worker setup, please ensure that only one worker
   writes the logs to avoid the duplication. In JAX, for example, the check
   could be `if jax.process_index() == 0`

  > **_NOTE:_** `logging_enabled` defaults to `False` and Goodput computations
  cannot be completed if no logs are ever written.

 For example:

 ```python
 goodput_recorder = goodput.GoodputRecorder(
  job_name=config.run_name,
  logger_name=goodput_logger_name,
  logging_enabled=(jax.process_index() == 0)
  )
 ```

  > **_NOTE:_** JAX initialization should be complete before this call.

### Record Data with `GoodputRecorder`

#### Record Job Start and End Time

 Use the recorder object to record the job's overall start and end time.

 For example:

 ```python
 def main(argv: Sequence[str]) -> None:
 # Initialize configs…
 goodput_recorder.record_job_start_time(datetime.datetime.now())
 # Device Initialization and device scanning…
 # Set up other things for the main training loop…
 # Main training loop
 train_loop(config)
 goodput_recorder.record_job_end_time(datetime.datetime.now())
 ```

#### Record Step Time

 Use the recorder object to record a step's start time using
 `record_step_start_time(step_count)`:

For example:

 ```python
 def train_loop(config, state=None):
 # Set up mesh, model, state, checkpoint manager…

 # Initialize functional train arguments and model parameters…

 # Define the compilation

 for step in np.arange(start_step, config.steps):
   goodput_recorder.record_step_start_time(step)
   # Training step…

 return state
 ```

#### Record Device Initialization, Training Preparation and Data Loading Time

  - Use the recorder object to record Device Initialization time using 
      `record_tpu_init_start_time` and `record_tpu_init_end_time`.
  - Use the recorder object to record Training Preparation time using
      `record_training_preparation_start_time` and
      `record_training_preparation_end_time`.
  - Use the recorder object to record Data Loading time using
      `record_data_loading_start_time` and `record_data_loading_end_time`.

  For example:

  ```python
  def train_loop(config, state=None):
  goodput_recorder.record_tpu_init_start_time()
  # Set up mesh, model, state, checkpoint manager…
  goodput_recorder.record_tpu_init_end_time()
  goodput_recorder.record_training_preparation_start_time()
  # Set up training set, initialize functional train args and model parameters…
  # Define the compilation
  # Set up any metrics collectors
  goodput_recorder.record_training_preparation_end_time()

  for step in np.arange(start_step, config.steps):
    goodput_recorder.record_data_loading_start_time()
    example_batch = load_next_batch(data_iterator, example_batch, config)
    goodput_recorder.record_data_loading_end_time()
    goodput_recorder.record_step_start_time(step)
    # Training step…

  return state
  ```

#### Record Custom Badput Events (e.g., Evaluation, SDC Checks)

- Use the recorder object to record the **start** of a custom badput event using
  `record_custom_badput_event_start_time(custom_badput_event_type='your_event_name')`.
- Use the recorder object to record the **end** of a custom badput event using
  `record_custom_badput_event_end_time(custom_badput_event_type='your_event_name')`.

Use these APIs when you want to account for time spent on operations that
block the training loop and use accelerator resources, do not contribute to
productive training and occur while training is in progress — such as step
evaluations, SDC checks, or re-compilations.

For example:

```python
def train_loop(config, state=None):
  goodput_recorder.record_training_preparation_start_time()
  # Initialize training config, setup model, load checkpoint...
  goodput_recorder.record_training_preparation_end_time()

  for step in range(config.steps):
    goodput_recorder.record_data_loading_start_time()
    batch = load_batch(train_data)
    goodput_recorder.record_data_loading_end_time()

    goodput_recorder.record_step_start_time(step)
    # Run training step...
    run_train_step(step, state)

    if step % config.eval_interval == 0:
      # Record a custom badput event for evaluation
      goodput_recorder.record_custom_badput_event_start_time(
          custom_badput_event_type="eval_step")
      run_step_evaluation(model, val_data)
      goodput_recorder.record_custom_badput_event_end_time(
          custom_badput_event_type="eval_step")

    if step % config.sdc_check_interval == 0:
      # Record a custom badput event for SDC check
      goodput_recorder.record_custom_badput_event_start_time(
          custom_badput_event_type="sdc_check")
      run_sdc_check(state)
      goodput_recorder.record_custom_badput_event_end_time(
          custom_badput_event_type="sdc_check")

  return state
```

> **_NOTE:_** The `custom_badput_event_type` string should be descriptive and
consistent (e.g., "eval_step", "sdc_check"), to ensure accurate aggregation and
reporting in badput breakdowns.

### Retrieve Goodput with `GoodputCalculator`

In order to retrieve the Goodput of a job run, all you need to do is instantiate
a `GoodputCalculator` object with the job's run name and the Cloud Logging
logger name used to record data for that job run. Then call the
`get_job_goodput` API to get the computed Goodput for the job run.

It is recommended to make the `get_job_goodput` calls for a job run from an
instance that runs elsewhere from your training machine.

#### Create a `GoodputCalculator` object

Create the calculator object:

```python
goodput_logger_name = f'goodput_{config.run_name}' # You can choose your own logger name.
goodput_calculator = goodput.GoodputCalculator(job_name=config.run_name, logger_name=goodput_logger_name)
```

If you want to enable Pathways, turn on the `using_pathways` flag:

```python
goodput_logger_name = f'goodput_{config.run_name}' # You can choose your own logger name.
goodput_calculator = goodput.GoodputCalculator(job_name=config.run_name, logger_name=goodput_logger_name, using_pathways=True)
```

#### Retrieve Goodput

Finally, call the `get_job_goodput` API to retrieve Goodput for the entire job
run. This API takes an optional parameter `include_badput_breakdown`. which
defaults to `False`.

The returned result is a tuple of the job’s Goodput at query-time, a dictionary
mapping various sources of Badput and their corresponding percentages and the
last recorded step. If `include_badput_breakdown` is not set, an empty
dictionary for Badput is returned.

If you are only interested in Goodput:

```python
total_goodput, _, _ = goodput_calculator.get_job_goodput()
print(f"Total job goodput: {total_goodput:.2f}%")
```

#### Retrieve Badput Breakdown

Badput breakdown is dictionary representation of various sources of Badput
mapped to its corresponding value. Badput is the percentage of time spent by the
job doing work that is not training to the total lifetime of the job. This
includes time spent doing device initialization, training preparation,
program startup, checkpoint loading, compilation or re-compilation, data loading,
checkpoint saving, custom badput events, wasted progress and time lost due
to disruptions.

Following Badput Breakdown buckets are supported by the library at this time:

```python
# Supported Badput Types
class BadputType(enum.Enum):
  """The type of Badput."""

  TPU_INITIALIZATION = 1
  TRAINING_PREP = 2
  PROGRAM_STARTUP = 3
  DATA_LOADING_SYNC = 4
  DATA_LOADING_ASYNC = 5 # This does not affect Goodput
  UNPRODUCTIVE_CHECKPOINT_SAVE_TIME = 6
  UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME = 7
  WASTED_PROGRESS_FROM_DISRUPTION = 8
  CUSTOM_BADPUT_EVENTS = 9
  OTHER = 10
```

#### Badput Breakdown Details

 - Accelerator Initialization Time (TPU_INITIALIZATION)

  This is the time spent on device discovery, slice initialization,
  device driver re-initialization and reset, security setup, initialization of
  pre-mapped buffers and more.

 - Training Preparation Time (TRAINING_PREP)

  This is the time spent on the creation of checkpoint managers, checkpoint
  loading, running mesh and model optimizers and more.

 - Program Startup Time (PROGRAM_STARTUP)

  This is the time spent on framework specific function transformations
  (such as JAX tracing), compilation tasks, runtime initialization etc.

 - Data Loading Time (DATA_LOADING_SYNC)

  This is the time spent on loading each batch of data for the training at a
  step to continue. This should be a small contribution to Badput if parallel
  data loading is used.

  Async data loading is accumulated overlapping with training steps and is
  non-blocking, therefore is not unproductive time. The time spent on overlapped
  data loading is stored in BadputType.DATA_LOADING_ASYNC, but does **not**
  affect overall Goodput of the workload.

 - Checkpointing Time (UNPRODUCTIVE_CHECKPOINT_SAVE_TIME, UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME)

  This is the time spent on saving a checkpoint and restoring a checkpoint.

  Depending on the type of checkpointing technology used by the program, there
  could be unproductive time while saving a checkpoint. When checkpointing is
  synchronous, the save operation will block training progress until it is complete.

  During asynchronous checkpointing, the model parameters or weights have to be
  transferred from the device memory to the host memory which is a blocking
  operation on the device. After the transfer, the device can proceed with model
  training while the CPU saves the checkpoint to storage in the background. The
  first blocking operation contributes to unproductive checkpoint time.

  If auto checkpointing is used, the checkpoint save operation is initiated upon
  detection of a planned disruption signal. The save operation in type of
  checkpointing is synchronous resulting in time lost to Badput.

 - Wasted Progress due to Disruption (WASTED_PROGRESS_FROM_DISRUPTION)

 Based on checkpointing frequency, a disruption may result in time lost in the
 form of wasted progress, i.e. time that was spent on productive training but
 lost after restart as well as time lost for the infrastructure to restart the
 workload.

  When there is a disruption, Badput is expected to accumulate in
  each of the following buckets after restart:

  - Accelerator Initialization
  - Training Preparation
  - Program Startup
  - Wasted Progress due to Disruption

 - Custom Badput Events (CUSTOM_BADPUT_EVENTS)

 Your application can optionally use record and monitor badput from custom
 synchronous (blocking training) and overlapping (between training steps)
 events. These events are are generally used for useful non-training activity on
 the accelerator while training is in progress such as performing SDC checks
 or evaluations.

If you are interested in retrieving Badput Breakdown along with Goodput:

```python
goodput, badput_breakdown, last_step = goodput_calculator.get_job_goodput(include_badput_breakdown=True)
print(f"Last step recorded: {last_step}")
print(f"Goodput: {goodput:.2f}%")
print(f"Badput due to TPU initialization: {badput_breakdown[goodput.BadputType.TPU_INITIALIZATION]:.2f}%")
print(f"Badput due to training preparation: {badput_breakdown[goodput.BadputType.TRAINING_PREP]:.2f}%")
print(f"Badput due to program startup: {badput_breakdown[goodput.BadputType.PROGRAM_STARTUP]:.2f}%")
print(f"Badput due to data loading: {badput_breakdown[goodput.BadputType.DATA_LOADING_SYNC]:.2f}%")
print(f"Badput due to disruption and wasted progress: {badput_breakdown[goodput.BadputType.WASTED_PROGRESS_FROM_DISRUPTION]:.2f}%")
print(f"Badput due to checkpoint save: {badput_breakdown[goodput.BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME]:.2f}%")
print(f"Badput due to checkpoint restore: {badput_breakdown[goodput.BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME]:.2f}%")
print(f"Badput due to step evaluation: {badput_breakdown[goodput.BadputType.CUSTOM_BADPUT_EVENTS].get('EVAL_STEP', 0.0):.2f}%")
print(f"Badput due to SDC checks: {badput_breakdown[goodput.BadputType.CUSTOM_BADPUT_EVENTS].get('SDC_CHECK', 0.0):.2f}%")
print(f"Badput from unknown source: {badput_breakdown[goodput.BadputType.OTHER]:.2f}%")
```

#### Interval Query Goodput and Badput

If you are interested in retrieving Goodput and Badput of the workload within a
specific window of time, the `GoodputCalculator` exposes the
`get_job_goodput_interval` API which computes metrics between the start and end
of this window.

This API also returns the last step recorded for the job. the total job time in
this window and the number of disruptions within the interval window.

> **_IMPORTANT:_** **Use this API if** you know the exact window of time within
 the workload's total run time that you are interested in.

> **_IMPORTANT:_** **Do NOT use this API if** your workload has been manually
 disrupted.

> **_IMPORTANT:_** **Do NOT use this API if** you have accidentally re-used a
 previous `run_name`.

```python
# Example usage
start_time_str = "2024-12-16 1:05:00"
start_time_utc = convert_pst_to_utc(start_time_str)
end_time_str = "2024-12-17 2:00:00"
end_time_utc = convert_pst_to_utc(end_time_str)
current_goodput, badput_breakdown, last_step, total_time, disruptions = goodput_calculator.get_job_goodput_interval(start_time_utc, end_time_utc)
```

### Monitor Goodput with `GoodputMonitor`

In order to monitor the Goodput of a job run on Tensorboard, all you need to do
is instantiate a `GoodputMonitor` object with the job's run name, cloud logger
name and Goodput monitoring configurations (as described below). Then call the
`start_goodput_uploader` API to asynchronously query and upload measured Goodput
to the specified Tensorboard directory.

#### Create a `GoodputMonitor` object

Create a `GoodputMonitor` object with the following parameters:

 1. `job_name`: The full run name of the job.
 2. `logger_name`: The name of the Cloud Logging logger object (created in the previous step).
 3. `tensorboard_dir`: The directory to write TensorBoard data to.
 4. `upload_interval`: The time interval at which to query and upload data to TensorBoard.
 5. `monitoring_enabled`: Whether or not monitoring is enabled.
        If the application is interested in monitoring Goodput, it should set
        this value to True. Only one worker should enable monitoring.
 6. `include_badput_breakdown`: Whether to query and upload badput breakdown
        data to Tensorboard.

> **_NOTE:_** Please ensure that only **one** worker enables monitoring of Goodput.
   In JAX, for example, the check could be `if jax.process_index() == 0`

For example:

```python
goodput_logger_name = f'goodput_{config.run_name}' # You can choose your own logger name.
goodput_monitoring_enabled = config.monitor_goodput and jax.process_index() == 0 # Check for configs whether or not the enable monitoring.

goodput_monitor = monitoring.GoodputMonitor(
      job_name=config.run_name,
      logger_name=logger_name,
      tensorboard_dir=config.tensorboard_dir,
      upload_interval=config.goodput_upload_interval_seconds,
      monitoring_enabled=True,
      include_badput_breakdown=True,
    )
```

If you want to enable Pathways, turn on the `pathway_enabled` flag:

```python
goodput_logger_name = f'goodput_{config.run_name}' # You can choose your own logger name.
goodput_monitoring_enabled = config.monitor_goodput and jax.process_index() == 0 # Check for configs whether or not the enable monitoring.

goodput_monitor = monitoring.GoodputMonitor(
      job_name=config.run_name,
      logger_name=logger_name,
      tensorboard_dir=config.tensorboard_dir,
      upload_interval=config.goodput_upload_interval_seconds,
      monitoring_enabled=True,
      include_badput_breakdown=True,
      pathway_enabled=True
    )
```

If you want to monitor Step Time Deviation, configure the `GoodputMonitor`
as follows:

```python
goodput_logger_name = f'goodput_{config.run_name}' # You can choose your own logger name.
goodput_monitoring_enabled = config.monitor_goodput and jax.process_index() == 0 # Check for configs whether or not the enable monitoring.

goodput_monitor = monitoring.GoodputMonitor(
      job_name=config.run_name,
      logger_name=logger_name,
      tensorboard_dir=config.tensorboard_dir,
      upload_interval=config.goodput_upload_interval_seconds,
      monitoring_enabled=True,
      include_badput_breakdown=True,
      include_step_deviation=True,
      configured_ideal_step_time=None # Optional, the library will compute ideal step time if it is not provided
    )
```

#### Start asynchronous "query and upload" of Goodput

Call the `start_goodput_uploader` API to spin off a thread which continuously
queries and uploads Goodput.

Note: This will upload Goodput and Badput data to Google Cloud Monitoring
by default.

```python
goodput_monitor.start_goodput_uploader()
```

#### Start asynchronous "query and upload" of Step Time Deviation

Call the `start_step_deviation_uploader` API to spin off a thread which
continuously queries and uploads step time deviation.

Note: This will upload Step Time Deviation data to Google Cloud Monitoring
by default.

```python
goodput_monitor.start_step_deviation_uploader()
```

#### Visualize on Tensorboard

1. Make sure you have `tensorboard-plugin-profile`, `tensorflow` and `tensorboard` packages installed
2. Follow instructions [here](https://cloud.google.com/tpu/docs/profile-tpu-vm#start_profiling_the_model_training) to start the Tensorboard server

#### Access Goodput, Badput and Step Deviation on Google Cloud Monitoring

By default, performance data ([goodput](https://cloud.google.com/monitoring/api/metrics_gcp#:~:text=workload/goodput_time), [badput](https://cloud.google.com/monitoring/api/metrics_gcp#:~:text=workload/badput_time), and [step deviation](https://cloud.google.com/monitoring/api/metrics_gcp#:~:text=workload/performance)) is automatically sent to Google Cloud Monitoring, enabling visualization on dashboards.

This feature leverages Google VM metadata (project ID, location, accelerator type)
and supports replica IDs for uniquely identifying workloads in multi-replica
deployments.

```python

gcp_options = goodput_utils.GCPOptions(
      project_id=None, # If None, the library will automatically identify from GCE internal metadata
      location=None, # If None, the library will automatically identify from GCE internal metadata
      replica_id='0', # Default is '0'
      acc_type=None, # If None, the library will automatically identify from GCE internal metadata
      enable_gcp_goodput_metrics=True,
      enable_gcp_step_deviation_metrics=True,
    )

goodput_monitor = monitoring.GoodputMonitor(
      job_name=config.run_name,
      logger_name=logger_name,
      tensorboard_dir=config.tensorboard_dir,
      upload_interval=config.goodput_upload_interval_seconds,
      monitoring_enabled=True,
      include_badput_breakdown=True,
      include_step_deviation=True,
      configured_ideal_step_time=None, # Optional, the library will compute ideal step time if it is not provided
      gcp_options=gcp_options
    )
```

If you do not wish to send metrics to Google Cloud Monitoring then please set
the flag `enable_gcp_goodput_metrics` to `False` for disabling goodput metrics
and `enable_gcp_step_deviation_metrics` to `False` for disabling step deviation
metrics while creating the GCPOptions object.

Setting `monitoring_enabled` to `False` will disable both tensorboard and GCM
monitoring.

```python

gcp_options = goodput_utils.GCPOptions(
      project_id=None, # If None, the library will automatically identify from GCE internal metadata
      location=None, # If None, the library will automatically identify from GCE internal metadata
      replica_id='0', # Default is '0'
      acc_type=None, # If None, the library will automatically identify from GCE internal metadata
      enable_gcp_goodput_metrics=False,
      enable_gcp_step_deviation_metrics=False,
    )


goodput_monitor = monitoring.GoodputMonitor(
      job_name=config.run_name,
      logger_name=logger_name,
      tensorboard_dir=config.tensorboard_dir,
      upload_interval=config.goodput_upload_interval_seconds,
      monitoring_enabled=True,
      include_badput_breakdown=True,
      include_step_deviation=True,
      configured_ideal_step_time=None,
      gcp_options=gcp_options,
    )
```

If you want to monitor Goodput and Badput metrics computed in a specific window
of time, you can use the `start_goodput_interval_uploader` monitoring API.

#### Create the `GoodputMonitor` with `enable_gcp_goodput_metrics` set to `True` in `GCPOptions`

```python

gcp_options = goodput_utils.GCPOptions(
      project_id=None, # If None, the library will automatically identify from GCE internal metadata
      location=None, # If None, the library will automatically identify from GCE internal metadata
      replica_id='0', # Default is '0'
      acc_type=None, # If None, the library will automatically identify from GCE internal metadata
      enable_gcp_goodput_metrics=True,
    )

goodput_monitor = monitoring.GoodputMonitor(
      job_name=config.run_name,
      logger_name=logger_name,
      tensorboard_dir=config.tensorboard_dir,
      upload_interval=config.goodput_upload_interval_seconds,
      monitoring_enabled=True,
      include_badput_breakdown=True,
      gcp_options=gcp_options
    )
```

#### Start asynchronous "query and upload" of Interval Goodput to GCM.

Call the `start_goodput_interval_uploader` API and specify `window_size_seconds`
to compute Goodput and Badput metrics only in the sliding time window.
The interval starts `window_size_seconds` prior to time of query, ends at time
of query, and moves ahead by `upload_interval` seconds.

This call is asynchronous and will only upload Goodput and Badput data to
Google Cloud Monitoring, and not to Tensorboard.

```python
# Set the window size to be 12h
goodput_monitor.start_goodput_interval_uploader(window_size_seconds = 43200)
```

Note: Google Cloud Monitoring will allow you to view all the metrics reported
during the entire workload. GCM will also allow you to filter by any time window
(irrespective of `window_size_seconds`). Each data point that is reported by
this API will correspond to computation only within the sliding window of size
`window_size_seconds`.