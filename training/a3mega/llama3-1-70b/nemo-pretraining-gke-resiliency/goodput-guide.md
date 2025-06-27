# Maximizing ML Training Efficiency: A General Guide to Improving GoodPut

Effective utilization of resources in large-scale machine learning (ML) training is crucial for both cost efficiency and rapid model development. A key metric for measuring this efficiency is **ML GoodPut**. As discussed in the Google Cloud blog post, "[Train AI for less: Improve ML Goodput with elastic training and optimized checkpointing](https://cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput)," GoodPut represents the actual productive training time, excluding time lost to various inefficiencies. Even a small percentage improvement in GoodPut can lead to significant cost savings and faster time-to-market for your models. For instance, consider a large training job utilizing 1024 GPUs. If this job runs for 30 days, the total available GPU hours are 1024 GPUs * 30 days * 24 hours/day = 737,280 GPU hours. If the GoodPut is only 50%, this means 368,640 GPU hours are wasted due to inefficiencies. Improving GoodPut by just 10% (from 50% to 60%) would reclaim 73,728 GPU hours, potentially saving hundreds of thousands of dollars and accelerating research by weeks.

## Table of Contents
- [TLDR: Recommended Lego Blocks for Your Deployment](#tldr-recommended-lego-blocks-for-your-deployment)
- [Minimizing Downtime: Optimized Checkpointing](#minimizing-downtime-optimized-checkpointing)
  - [1. Asynchronous Checkpointing](#1-asynchronous-checkpointing)
  - [2. Multi-Tier Checkpointing Strategy (Leveraging GCS with FUSE)](#2-multi-tier-checkpointing-strategy-leveraging-gcs-with-fuse)
  - [3. Distributed Checkpointing](#3-distributed-checkpointing)
  - [4. Configurable Checkpoint Frequency](#4-configurable-checkpoint-frequency)
- [Addressing Interruptions: Elastic Training](#addressing-interruptions-elastic-training)
  - [1. Failure Sensing and Mitigation: The Supervisor System](#1-failure-sensing-and-mitigation-the-supervisor-system)
  - [2. Remediation Strategies](#2-remediation-strategies)
- [Measuring Success: Goodput Analysis](#measuring-success-goodput-analysis)
- [Tying It All Together: A Holistic Approach](#tying-it-all-together-a-holistic-approach)

Achieving high GoodPut can be challenging due to several factors common in large distributed training environments. The table below outlines the main sources of BadPut and their potential impact:

| Source of BadPut                             | Description/Impact                                                                          | Potential GoodPut Loss (Example %) |
| :------------------------------------------- | :------------------------------------------------------------------------------------------ | :--------------------------------- |
| **Hardware Failures and System Errors**      | Causes crashes, lost progress, time to detect/reprovision/restart.                            | 5-15%                              |
| **Preemptions and Evictions**                | Similar to hardware failures, results in lost work and restart overhead.                    | 5-10%                              |
| **Slow Checkpoint Save and Load Times**      | GPUs idle during synchronous saves; slow loads extend downtime.                             | 3-10%                              |
| **Suboptimal Checkpoint Frequency**          | Too infrequent leads to large work loss; too frequent causes high overhead.                 | 2-8%                               |
| **Stragglers and Performance Bottlenecks**   | Slower nodes delay the entire job, underutilizing resources.                                | 3-7%                               |
| **Lack of Rapid Failure Detection and Diagnosis** | Longer detection/diagnosis time increases downtime.                                         | 2-5%                               |

This guide provides a general overview of techniques and tools to address these common challenges and maximize ML GoodPut. While the principles discussed are broadly applicable, we will use the [Llama 3.1 70B pretraining recipe](https://github.com/AI-Hypercomputer/gpu-recipes/tree/main/training/a3mega/llama3-1-70b/nemo-pretraining-gke-resiliency) as a concrete case study to illustrate how these components can be implemented and customized for large-scale training workloads on Google Cloud. The goal is to showcase a "DIY" style product, where users can understand and selectively adopt these "Lego blocks" to build resilient and efficient training pipelines.

## TLDR: Recommended Lego Blocks for Your Deployment
For customers looking to improve GoodPut on their own ML training workloads, here’s a concise guide to the key strategies discussed in this document, presented as 'Lego blocks' you can implement:

1.  **Optimize Checkpointing - Start with Asynchronous Checkpointing:**
    *   **Why:** Minimize GPU idle time by offloading checkpoint saves to CPU/background processes. This directly boosts GoodPut. This can directly recover a significant portion of GoodPut, potentially 3-10%, by minimizing GPU idle time during saves.
    *   **How:** Enable asynchronous checkpointing features in your training framework (e.g. `--enable-async-ckpt` in our recipe). Ensure you have sufficient CPU and memory resources on host machines for this. See [Asynchronous Checkpointing](#1-asynchronous-checkpointing) for details.

2.  **Leverage Proper Optimized Storage (For Example: Cloud Storage with FUSE) for Checkpoints:**
    *   **Why:** Provides durable, accessible, and scalable storage for your checkpoints, crucial for recovery across different nodes or after failures.
    *   **How:** Use a proper service, like Google Cloud Storage (GCS) with the Cloud Storage FUSE CSI driver to mount GCS buckets as local filesystems. Configure your training job to save checkpoints to this mounted path. More details can be found in [Multi-Tier Checkpointing Strategy (Leveraging GCS with FUSE)](#2-multi-tier-checkpointing-strategy-leveraging-gcs-with-fuse).

3.  **Consider Distributed Checkpointing (For Very Large Models/Setups):**
    *   **Why:** You can get a speed up by parallelize the checkpoint save/load process itself.
    *   **How:** Utilize distributed checkpointing features within your framework (e.g., `--enable-dist-ckpt` in recipe). This typically involves each worker saving its shard of the model and optimizer state. Refer to [Distributed Checkpointing](#3-distributed-checkpointing) for more information.

4.  **Tune Checkpoint Frequency:**
    *   **Why:** Balance the risk of lost work against the overhead of checkpointing.
    *   **How:** Configure how often checkpoints are saved (e.g., based on training steps or time). Monitor your failure rates and checkpoint durations to find an optimal balance. See [Configurable Checkpoint Frequency](#4-configurable-checkpoint-frequency) for guidance.

5.  **Implement a Job Resiliency System (Elastic Training):**
    *   **Why:** This is foundational for resilience, addressing BadPut from hardware failures and preemptions, which can account for 5-15% of lost GoodPut. It automates detection and recovery.
    *   **How:** Adapt or implement a supervisor system like the one detailed in the 'Elastic Training' section. Focus on failure sensing, policy-based remediation (like node hot-swapping), and ensuring your training job can be controlled externally (start/stop/checkpoint). The Google Cloud Resiliency library components (Sensor, Controller, Actuator, Host Monitors) provide a strong template. Detailed implementation strategies are discussed in [Addressing Interruptions: Elastic Training](#addressing-interruptions-elastic-training).

Begin by optimizing your checkpointing process (options 1-4 above), choosing the techniques most relevant to your workload's scale and characteristics, as this often provides the most immediate GoodPut gains. Then, implement a robust supervisor system to build upon this with comprehensive resilience against interruptions. Finally, continuously monitor your GoodPut to measure improvements.

## Minimizing Downtime: Optimized Checkpointing

Checkpointing is vital for fault tolerance, allowing training to resume from a saved state. However, the checkpointing process itself can consume valuable time and, if not optimized, reduce GoodPut. The Llama 3.1 70B recipe, as an example, incorporates several strategies for optimized checkpointing, aligning with principles from the [Google Cloud blog post](https://cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput).

Choosing the right checkpointing strategy, or combination of strategies, is crucial for both minimizing training disruption and ensuring robust recovery. The methods described below—asynchronous, distributed, and multi-tier storage—can be seen as complementary building blocks. Your choice will depend on factors like model size, training scale, and infrastructure characteristics.

Consider the following when making your decision:

*   **Asynchronous Checkpointing:** This is generally recommended for most training jobs. By offloading the checkpoint save operation to background processes (typically on the CPU), it allows the GPUs to continue training with minimal interruption. This directly improves GoodPut by reducing idle GPU time. It's effective for both single-node and multi-node training.

*   **Distributed Checkpointing:** When training very large models across a significant number of nodes and GPUs, the process of gathering and saving the model state can still be time-consuming, even if asynchronous. Distributed checkpointing parallelizes the save (and load) process itself, where each worker or a subset of workers handles its portion of the model state concurrently. This is often used in conjunction with asynchronous checkpointing to further reduce the critical path of saving checkpoints.

*   **Integration with the Supervisor System:** The Supervisor system (detailed in the "Elastic Training" section) acts as the overall training controller and relies on a robust and efficient checkpointing mechanism to enable automated recovery from hardware failures or preemptions. When the Supervisor restarts a job or a pod, it depends on the training application's ability to quickly load the latest checkpoint. Therefore, selecting fast and reliable checkpointing methods (like asynchronous and distributed, saved to resilient storage like GCS) is key to minimizing downtime when the Supervisor needs to intervene. The goal is a synergistic relationship: checkpointing provides the recovery points, and the Supervisor automates the recovery process.

These strategies can often be combined. For instance, a large distributed training job would ideally use both distributed checkpointing (to quickly gather state from all workers) and asynchronous checkpointing (to offload the writing to persistent storage without stalling GPUs), all while being monitored by the Supervisor for fault tolerance.

### 1. Asynchronous Checkpointing

To prevent training pauses during checkpoint saves, this recipe (Llama 3.1 70B resiliency recipe that can be found in this repository) leverages asynchronous checkpointing. This means the training process (e.g., GPU computation) can continue while checkpoints are being written to storage in the background. This is typically achieved by first copying the checkpoint data from GPU memory to host CPU memory, which is a fast operation, and then the host CPU handles the slower write to persistent storage.

*   This capability is enabled in the recipe via flags in the main `workload.flags` section of [values.yaml](values.yaml):
    *   `--enable-async-ckpt`: Enables the basic asynchronous checkpointing feature.
    *   `--enable-optimized-async-ckpt`: Enables further optimizations for the asynchronous checkpointing mechanism, potentially improving the efficiency of offloading data from GPU HBM to host memory and managing the subsequent save.
    *   `--ckpt-threads-per-rank=2`: (Example from [values.yaml](values.yaml)) Configures the number of threads per rank dedicated to checkpointing operations, which can help parallelize and speed up the process. Users can tune the `--ckpt-threads-per-rank` value; increasing it may improve checkpointing speed if the process is I/O bound and sufficient CPU resources are available, but excessive threads could also lead to contention. Optimal values should be determined through experimentation.

### 2. Multi-Tier Checkpointing Strategy (Leveraging GCS with FUSE) (Preview)

GCP offers Managed Lustre as a preferred first step for a high-performance file system tier in a multi-tiered checkpointing strategy (Preview). [Our blog post](https://cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput?e=48754805) further describes an ideal multi-tiered approach (local node storage, peer node storage, cloud storage) for balancing speed and resilience. As an alternative, if customers have specific requirements or existing GCS utilization patterns that make it a better fit, the LLaMA3-1-70B recipe prominently features Google Cloud Storage (GCS) as a robust and scalable tier for durable checkpoint storage, accessed via the [Cloud Storage FUSE CSI driver](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver).

*   **GCS for Checkpoints:**
    *   The [values-gcs.yaml](values-gcs.yaml) file defines the GCS bucket to be used (e.g., `gcs-checkpoints`). Users should ensure this GCS bucket is provisioned in the same region as their GKE cluster, has appropriate write/read permissions for the training job's service account, and has Hierarchical Namespace enabled for potentially better performance, as detailed in the main recipe [README.md](README.md).
    *   The main [README.md](README.md) of the recipe details setting up the GCS bucket (Hierarchical Namespace recommended) and configuring access via a Kubernetes Persistent Volume (PV) and Persistent Volume Claim (PVC).
    *   The `infrastructure.enable_gcsfuse: true` setting in [values.yaml](values.yaml) ensures that GCS FUSE is utilized for the job.
    *   The underlying Helm chart for GCS FUSE setup can be found in [src/helm-charts/storage/gcs-fuse/](../../../../src/helm-charts/storage/gcs-fuse/).
*   **How GCS FUSE Helps:** GCS FUSE allows Kubernetes Pods to mount a GCS bucket as a local filesystem. This simplifies access for training frameworks, as they can read/write checkpoints to what appears to be a local path, while the data is actually persisted to GCS. This is crucial for both saving checkpoints and for restoring them during job recovery.
    *   **Note:** GCS FUSE is not a POSIX compliant file system; certain file system features like file locking and atomic rename may not work as expected. It is recommended to test GCS FUSE with your workload before using it in production.
*   While this recipe focuses on GCS as the primary persistent checkpointing backend, advanced configurations might allow for staging checkpoints on local SSDs before asynchronous upload to GCS, achieving a multi-tier behavior.

### 3. Distributed Checkpointing

For large models trained across many GPUs, saving and loading checkpoints can be a bottleneck if handled by a single process or node. Distributed checkpointing, often a feature of the training framework (like PyTorch, which NeMo builds upon), addresses this by parallelizing the save/load operations across multiple workers/nodes. Each rank or a subset of ranks saves its portion of the model state concurrently.

*   The `--enable-dist-ckpt` flag in [values.yaml](values.yaml) activates this feature.
*   For more details on PyTorch's distributed checkpointing capabilities, refer to the [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html) (specific links may vary by PyTorch version, search for "distributed checkpointing" or "state_dict").

### 4. Configurable Checkpoint Frequency

The optimal frequency for saving checkpoints is a balance: too infrequent, and you risk losing significant work; too frequent, and the overhead (even if async) can become substantial.

*   The `--checkpoint-interval=25` (by default, measured in training steps) in the `workload.flags` section of [values.yaml](values.yaml) allows users to tune this. This value is specified in terms of training steps. The optimal interval is a trade-off: smaller intervals reduce the amount of lost computation in case of a failure but increase the aggregate time spent on checkpointing. Larger intervals minimize checkpointing overhead but risk more lost work. Users should tune this based on their specific job's typical step duration and observed failure rates.
*   Other related flags like `--topk-ckpt=-1` (from [values.yaml](values.yaml), meaning keep all checkpoints in this case) also play a role in the checkpointing strategy. A value of `-1` (as shown in the example) means all checkpoints are kept, which can consume considerable storage over long runs. Users should set this to a positive integer to keep only the latest 'k' checkpoints, balancing recovery needs with storage costs.

## Addressing Interruptions: Elastic Training

Elastic training is a core strategy for improving ML GoodPut by making training workloads resilient to interruptions. Instead of a job failing entirely when an issue occurs, elastic training allows the job to adapt to the changing environment. This could involve recovering from a transient error, transparently moving to different hardware, or adjusting the job size to continue training on available resources.

The Llama 3.1 70B recipe, as a case study, implements these elastic training principles through the **Google Cloud Resiliency library**. This library is designed to work with GKE and leverages the [NVIDIA Resiliency Extension (NVRx)](https://github.com/NVIDIA/nvidia-resiliency-ext) for certain low-level hardware interactions and failure signaling.

Key components and concepts include:

### 1. Failure Sensing and Mitigation: The Supervisor System

A sophisticated supervisor system is deployed to monitor the health of the training cluster and the job itself. This system is crucial for quickly identifying issues and orchestrating a response. It consists of:

*   **Supervisor Components:** These typically run on a dedicated CPU node pool.
    *   **Sensor:** Actively monitors the training job and cluster components for failure signals, performance degradation, or straggler behavior (in the future). It might use heartbeat mechanisms (polling worker nodes) and receive signals from other sources like the Host Monitors. The [`heartbeat_polling_period_s`](values-supervisor.yaml) and [`heartbeat_timeout_s`](values-supervisor.yaml) in [values-supervisor.yaml](values-supervisor.yaml) are critical for this.
    *   **Controller:** The central "brain" that receives event data from the Sensor. It consults a user-defined policy (or its internal logic) to decide on the appropriate remediation action.
    *   **Actuator:** Executes the remediation actions chosen by the Controller, such as initiating a job restart, requesting a node replacement, or triggering a scaling operation.
    *   The configuration for these components, including their Docker images and startup commands, can be found in [values-supervisor.yaml](values-supervisor.yaml).
    *   The Kubernetes service accounts and roles required for the Supervisor to interact with GKE resources are defined in [ksa-setup.yaml](ksa-setup.yaml).
    *   The underlying Helm chart that deploys these supervisor components is located in [src/helm-charts/resiliency/supervisor-chart/](../../../../src/helm-charts/resiliency/supervisor-chart/).

This entire Supervisor system (Sensor, Controller, Actuator, and Host Monitor DaemonSets) is designed as a modular 'Lego block'. While showcased here with NeMo, its components and principles can be adapted for other training frameworks by customizing the interaction points, primarily through the Actuator's remediation scripts and the policies defined in [values-supervisor.yaml](values-supervisor.yaml).

#### Using the Supervisor with Your Custom Model
This Supervisor system can be integrated with your custom training frameworks or models beyond the Llama 3.1 70B NeMo example. Here's a general guide:

*   **Deployment:** The Supervisor system (Supervisor Controller and Host Monitor DaemonSets) is deployed via its dedicated Helm chart, found at [src/helm-charts/resiliency/supervisor-chart/](../../../../src/helm-charts/resiliency/supervisor-chart/).
*   **Configuration:** Crucially, you'll need to customize the [values-supervisor.yaml](values-supervisor.yaml) file. This includes:
    *   Defining your GKE cluster setup (node pools, etc.).
    *   Setting appropriate monitoring parameters like heartbeat intervals, timeouts, and failure detection thresholds ([`heartbeat_polling_period_s`](values-supervisor.yaml), [`heartbeat_timeout_s`](values-supervisor.yaml), [`pod_termination_threshold_s`](values-supervisor.yaml), [`jobset_downtime_threshold_s`](values-supervisor.yaml)) to match your job's behavior.
    *   Specifying the remediation policies and scripts the Actuator should use for events like job restarts, node replacements, or scaling.
*   **Actuator Integration:** The core of the integration lies in how the Supervisor's Actuator component interacts with your custom training application. Your application must be controllable via external commands or signals that the Actuator can trigger. This might involve:
    *   The Actuator executing custom scripts that interact with your job (e.g., to stop, start, or send signals).
    *   Your training framework exposing APIs that the Actuator can call.
    *   Using signals (e.g., SIGUSR1, SIGTERM) that your application traps to initiate actions like saving a checkpoint and exiting, or re-evaluating cluster membership.
*   **Checkpointing and Resumption:** Your custom application must implement robust checkpointing and the ability to resume training from these checkpoints. This is essential because Supervisor-initiated actions (like restarting a job after a failure or preemption) will rely on your application's capability to continue from the last known good state.

By carefully configuring these aspects, you can leverage the Google Cloud Resiliency library's Supervisor system to bring enhanced fault tolerance and elastic training capabilities to a wide range of ML workloads.

*   **Host Monitors:** These are deployed as a Kubernetes DaemonSet, ensuring one runs on each GPU worker node (e.g., A3 Mega nodes).
    *   They provide granular, node-level health information and can detect local hardware issues (like GPU errors) more directly.
    *   They communicate with the central Supervisor, feeding it critical data for decision-making. Configuration details are also present in [values-supervisor.yaml](values-supervisor.yaml) (see [`host_daemon` section](values-supervisor.yaml)).

The interaction between these components allows the system to automatically sense disruptions (e.g., using parameters like [`pod_termination_threshold_s`](values-supervisor.yaml) and [`jobset_downtime_threshold_s`](values-supervisor.yaml) from [values-supervisor.yaml](values-supervisor.yaml)) and initiate mitigation procedures. The system also supports fault injection ([`enable_fault_injection`](values-supervisor.yaml) in [values-supervisor.yaml](values-supervisor.yaml)) for testing resiliency.

### 2. Remediation Strategies

The Google Cloud Resiliency library, leveraging the NVIDIA Resiliency Extension, is designed to support various remediation strategies. The exact policy and automation level can be customized:

*   **In-Job Restarts / GPU Reset:** For certain correctable errors (e.g., transient GPU issues identified by lower-level hardware monitoring), the NVIDIA library might enable an in-job restart to restore functionality. Following such recovery attempts, the Supervisor system orchestrates the restart of the affected training job components (e.g., Kubernetes pods) to ensure they rejoin the training process and resume from the last valid checkpoint. The primary goal from the Supervisor's perspective is to bring the job back to a healthy, training state.
*   **Node Hot Swap:** This is a core capability of the Supervisor system. When the Sensor (using health-check parameters like [`heartbeat_polling_period_s`](values-supervisor.yaml) and [`heartbeat_timeout_s`](values-supervisor.yaml) from `values-supervisor.yaml`) and the Host Monitors detect an unrecoverable node failure, the Controller evaluates the situation based on its configured policies. If a node replacement is deemed necessary, the Actuator component interacts with GKE to de-allocate the failed node and provision a new one from the available resource pool. The training job, often managed by a higher-level controller like JobSet, subsequently resumes on the reconstituted set of nodes, loading from the latest checkpoint.
*   **Scaling Down (and Up):** The target size of the training job is defined by parameters such as [`num_dp_replicas`](values-supervisor.yaml) and [`num_nodes_per_dp`](values-supervisor.yaml) in [values-supervisor.yaml](values-supervisor.yaml). If nodes fail and replacement resources are not immediately available, the Supervisor's Controller can decide to scale down the job to continue training on the remaining healthy nodes. In such scenarios, the Actuator would modify the job specification (e.g., by updating the JobSet resource if Kubernetes JobSet is being used, or by interacting with the specific training framework's scaling mechanisms). The system is designed to scale back up to its target size if new resources become available or previously failed nodes are restored. The Supervisor components facilitating these actions are deployed via a Helm chart, available at [src/helm-charts/resiliency/supervisor-chart/](../../../../src/helm-charts/resiliency/supervisor-chart/).

#### Customizing Remediation Logic
While [values-supervisor.yaml](values-supervisor.yaml) defines the monitoring parameters (like heartbeats and timeouts) and high-level remediation policies (e.g., whether to attempt a node swap or scale down), the precise commands and mechanisms for interacting with the *specific training application* during remediation are typically implemented within the controller definition by overriding the `event_policy()` method. For instance, the exact command to gracefully stop a NeMo pod, instruct MaxText to save an emergency checkpoint, or re-launch a specific training script with an updated list of participating nodes resides in this layer. Users can customize these Actuator scripts or provide their own implementations to integrate the Supervisor system seamlessly with their chosen training framework's operational needs, thus making the resiliency solution highly adaptable.

## Measuring Success: Goodput Analysis

Improving GoodPut is an ongoing process, and being able to measure it is critical to understanding the impact of the strategies you implement. The `gpu-recipes` repository provides a utility to help with this analysis.

*   **Resiliency Metrics Tool:**
    *   Located in the [src/utils/resiliency_metrics/](../../../../src/utils/resiliency_metrics/) directory (relative to the root of the `gpu-recipes` repository), the [calculator.py](../../../../src/utils/resiliency_metrics/calculator.py) script is designed to analyze training job logs and calculate various metrics, including the overall GoodPut percentage.
    *   The main [README.md](README.md#goodput-analysis-for-the-job) for the Llama 3.1 70B recipe includes detailed instructions on how to set up and run this tool (see the [Goodput Analysis for the job](README.md#goodput-analysis-for-the-job) section). Generally, using the tool involves these key steps:
        *   Navigating to the [src/utils/resiliency_metrics/](../../../../src/utils/resiliency_metrics/) directory.
        *   Creating a Python virtual environment and installing required packages from [requirements.txt](../../../../src/utils/resiliency_metrics/requirements.txt).
        *   Executing the `python3 calculator.py` script with necessary arguments, such as `--job-name <YOUR_JOB_NAME>` (which can be found using `kubectl get jobsets`), and parameters for log lookback periods (e.g., `--gcloud-logging-lookback-days 1`) and reference step times.

Using this tool, or similar log analysis techniques, allows you to quantify the benefits of elastic training and optimized checkpointing, identify remaining bottlenecks, and further tune your setup for maximum efficiency.


## Installation and Callback

In order to fully integrate our recipe and make use of the provided recipe
customization flags, please add callbacks to your training code by
replacing existing Lightning checkpoint callbacks with the ones provided in our library.

Our library provides five types of callbacks, namely `autocheckpoint`, `comm_overlap`, `logging`, `model_checkpoint`, and `profile`, for users to freely make use of. 

Regarding more details on how to make use of our auto checkpointing and model checkpointing callbacks, please refer to other sections of this guide and [the ReadMe file in our open-source Google Cloud Resiliency Library](https://github.com/AI-Hypercomputer/resiliency/blob/main/README.md).

Logging provides functionality of logging of memory, steps, TPS of steps and/or Tensorboard.

You can refer to all the callback functions and features in [the callback scripts of our open-source Google Cloud Resiliency Library](https://github.com/AI-Hypercomputer/resiliency/tree/main/resiliency/callbacks)

Below is an example of a `model_checkpoint` callback. Please ensure to install our library before attempting to use resiliency callbacks.

```
from resiliency.plugins._ckpt_utils import get_is_checkpoint_file_handler, find_latest_checkpoint_path
from resiliency.callbacks import model_checkpoint
callbacks=[]
callbacks.append(
	model_checkpoint.ModelCheckpoint(
		dirpath=f"{log_dir}/checkpoint",
		save_last=False,
		monitor="step",
		save_top_k=1,
		mode="max",
		save_weights_only=False,
		every_n_train_steps=5,
		save_on_train_epoch_end=True,
		save_optim_on_train_ends=True,
		always_save_context=False,
		filename="{step}",
		enable_version_counter=False,
		use_in_cluster_local_ckpts=None,
		enable_high_scale_ckpt=False,
		preprocess_files=True,
	}
)
```

## Summary Tables of Configurable Flags Provided by Google Cloud Resiliency Library and Recipe

| workload.flags on workload yaml file | Description | Type | Default |  |
|---|---|---|---|---|
| **_Checkpointing_** |  |  |  |  |
| --enable-async-ckpt  | Enable asynchronous checkpointing to offload the actual checkpointing threads to CPU. | bool | False |  |
| --enable-dist-ckpt | Enable distributed checkpointing for saving and loading checkpoints across multiple ranks in parallel. | bool | False |  |
| --enable-optimized-async-ckpt | Enable our optimized async checkpointing solution that offloads both the checkpointing preparation threads and the actual checkpointing threads to CPU. | bool | False |  |
| --enable-in-cluster-local-ckpt | Enable in-cluster in-node local checkpointing for the local layer in multi-tier checkpointing (MTC). | bool | False |  |
| --enable-high-scale-ckpt | Enable high scale checkpointing for the local layer in multi-tier checkpointing (MTC). `--local-ckpt-dir` must be set. | bool | False |  |
| --enable-ckpt-load-replication | Enable checkpoint load replication. `--local-ckpt-dir` must be set and `--num-optimizer-replicas` must be >= `2` | bool | False |  |
| --profile-ckpt-interval | Number of steps in between checkpoint profiling. For example, `10` means every 10th step.  | int | None |  |
| --ckpt-threads-per-rank | Number of threads used for writing checkpoint files per rank. | int | 2 |  |
| **_Training parameters_** |  |  |  |  |
| --job-name | Name of the training job. | str | "test_job" |  |
| --model | Optional field for defining model size. Possible options are `"36M"`, `"2B"`, `"8B"`, `"70B"`, `"405B"`. | str | "36M" |  |
| --num-nodes | Number of GPU nodes used in the training. | int | 1 |  |
| --num-gpus | Number of GPU chips used in the training. | int | 8 |  |
| --max-steps | Number of training steps. | int | 1_000_000 |  |
| --global-bs | Source of truth for global batch size (GBS). | int | None |  |
| --topk-ckpt | Number of top checkpoints to keep. | int | 10 | |
| --tokenizer-path | Path of the tokenizer file.  | str | "tokenizer.model" |  |
| --val-check-interval | Number of steps of the validation check interval. | int | 40 |  |
| --limit-val-batches | Number of batches to be used for validation. | int | 10 |  |
| --enable-comm-overlap | Enable communication overlap for improving MFU. | bool | False |  |
| --enable-gc | Enable garbage collection. | bool | False |  |
| --enable-fault-tolerance | Enable NVRx fault tolerance. | bool | False |  |
| --num-optimizer-replicas | Number of optimizer replicas. | int | 1 |  |
| --sim-fault-desc | Description of a fault to be simulated. Format: `"<fault_type>,<base_delay>"` Example: `"random,120"`| str | "" |  |
|  **_Logging_** |  |  |  |  |
| --log-dir | Directory for log output. | str | "/log/" |  |
| --log-to-remote-storage | Enable logging to remote storage log directory; otherwise, it logs to `"/tmp/"` folder | bool | False |  |
|  |  |  |  |  |
| --log-level | Log level. Logs above this level will be outputted. Options are `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`. | str | "INFO" |  |
| **_Some other key variables on workload yaml file_** | Description | Type | Default |  |
| workload.checkpointing.local_ckpt_dir | Directory for the checkpointing local storage such as SSD or ramdisk. If you do not want to save checkpoints on the local layer, omit this flag and omit `workload.checkpointing.local_ckpt_interval`. | str |  |  |
| workload.checkpointing.local_ckpt_interval | Number of steps in between saving checkpoints in the local in-node layer such as SSD or ramdisk. For example, `10` means every 10th step. Must be used with `workload.checkpointing.local_ckpt_dir`. | int |  |  |
| workload.checkpointing.persistent_ckpt_dir | Directory for the checkpointing persistent storage such as GCS. If you do not want to save checkpoints on the persistent storage layer, omit this flag and omit `workload.checkpointing.persistent_ckpt_interval`.  | str |  |  |
| workload.checkpointing.persistent_ckpt_interval | Number of steps in between saving checkpoints in the persistent storage layer such as GCS. For example, `10` means every 10th step. Must be used with `workload.checkpointing.persistent_ckpt_dir`. | int |  |  |
| workload.enable_tensorboard | Enable Tensorboard logging. | bool |  |  |
| infrastructure.use_supervisor | Use our supervisor from our Google Cloud Resiliency Library. | bool |  |  |
| infrastructure.enable_gcsfuse | Enable GCSFuse as storage solution. GCSFuse mounts and accesses GCS buckets as flattened local file systems for faster read and write. Official docs | bool |  |  |
| infrastructure.gcsfuse_bucket | Path to GCSFuse bucket. | str |  |  |
| infrastructure.pvc | Name of persistent volume claim (PVC). We provided an example pvc.yaml for reference. | str |  |  |
| infrastructure.host_daemon_port | Host daemon port number. `60060` or `61000` is used in our recipe examples. | int |  |  |
| infrastructure.max_workload_restarts | Maximum number of workload restarts before Jobset is considered failed. Set to 0 to hot-swap faulty nodes without first attempting to restart the workload. | int |  |  |
| infrastructure.max_in_job_restarts | Maximum number of NVRx in-job restarts. Set to 0 to disable it. After the maximum number of NVRx in-job restarts is reached, workload will restart instead. | int |  |  |