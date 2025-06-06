# Pretrain Mixtral-8x7B workloads on A3 Ultra GKE Node pools

This recipe outlines the steps for running a Mixtral 8x7B pretraining workload on
[A3 Ultra GKE Node pools](https://cloud.google.com/kubernetes-engine) by using the
[NVIDIA NeMo framework](https://github.com/NVIDIA/nemo).

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- Pretraining job configuration and deployment - A Helm chart is used to configure and deploy
  the [Kubernetes Jobset](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
  resource which manages the execution  of the
  [NeMo pretraining workload](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_gpt_pretraining.py).

## Test environment

This recipe has been optimized for and tested with the following configuration:

- GKE cluster
    - [A regional standard cluster](https://cloud.google.com/kubernetes-engine/docs/concepts/configuration-overview) version: 1.31.7-gke.1265000 or later.
    - A GPU node pool with 32 or 64 [a3-ultragpu-8g](https://cloud.google.com/compute/docs/gpus#h200-gpus) provisioned using the DENSE deployment type.
    - [Workload Identity Federation for GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/workload-identity) enabled.
    - [Cloud Storage FUSE CSI driver for GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/cloud-storage-fuse-csi-driver) enabled.
    - [DCGM metrics](https://cloud.google.com/kubernetes-engine/docs/how-to/dcgm-metrics) enabled.
    - [Kueue](https://kueue.sigs.k8s.io/docs/reference/kueue.v1beta1/) and [JobSet](https://jobset.sigs.k8s.io/docs/overview/) APIs installed.
    - Kueue configured to support [Topology Aware Scheduling](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/).
- A regional Google Cloud Storage (GCS) bucket to store logs generated by the recipe runs.

To prepare the required environment, see
[GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-ultra.md).

## Training dataset

This recipe uses a mock pretraining dataset provided by the NeMo framework

## Docker container image

This recipe uses the following [Deep Learning Software Layer](https://cloud.google.com/ai-hypercomputer/docs/software-stack#cluster_images) container image:

`us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo24.07-gib1.0.3-A3U`.

This image is based on NVIDIA NeMo 24.07 and contains the NCCL gIB plugin v1.0.3, bundling all NCCL binaries validated for use with A3 Ultra GPUs.

## Run the recipe

From your client workstation, complete the following steps:

### Configure environment settings

Set the environment variables to match your environment:

 ```bash
 export PROJECT_ID=<PROJECT_ID>
 export CLUSTER_REGION=<CLUSTER_REGION>
 export CLUSTER_NAME=<CLUSTER_NAME>
 export GCS_BUCKET=<GCS_BUCKET>
 export KUEUE_NAME=<KUEUE_NAME>
 ```

 Replace the following values:

 - `<PROJECT_ID>`: your Google Cloud project ID.
 - `<CLUSTER_REGION>`: the region where your cluster is located.
 - `<CLUSTER_NAME>`: the name of your GKE cluster.
 - `<GCS_BUCKET>`: the name of your Cloud Storage bucket. Don't include the `gs://` prefix.
 - `<KUEUE_NAME>`: the name of the Kueue local queue.  The default queue created by the cluster toolkit is `a3-ultra`. Make sure to verify the name of the local queue in your cluster.

Set the default project:

 ```bash
 gcloud config set project $PROJECT_ID
 ```

### Get the recipe

Clone the `gpu-recipes` repository and set a reference to the recipe folder.

```
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/training/a3ultra/mixtral-8x7b/nemo-pretraining-gke
cd $RECIPE_ROOT
```

### Get cluster credentials

```
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```


### Configure and submit a pretraining job

#### Using 32 nodes (256 GPUs)

The default job setting is 30 training steps and bf16 precision. To execute the job with the
default settings, run the following command:

```bash
helm  install -f $RECIPE_ROOT/values.yaml \
    --set-file workload_launcher=$REPO_ROOT/src/launchers/nemo-10-launcher.sh \
    --set-file workload_config=$REPO_ROOT/src/frameworks/a3ultra/nemo-configs/mixtral-8x7b-256gpus-a3u-bf16.yaml \
    --set queue=${KUEUE_NAME} \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    $USER-mixtral-8x7b-nemo \
    $REPO_ROOT/src/helm-charts/a3ultra/jobset
```

#### Using 64 nodes (512 GPUs)

The default job setting is 30 training steps and bf16 precision. To execute the job with the
default settings, run the following command:

```bash
helm  install -f $RECIPE_ROOT/values.yaml \
    --set-file workload_launcher=$REPO_ROOT/src/launchers/nemo-10-launcher.sh \
    --set-file workload_config=$REPO_ROOT/src/frameworks/a3ultra/nemo-configs/mixtral-8x7b-256gpus-a3u-bf16.yaml \
    --set queue=${KUEUE_NAME} \
    --set workload.gpus=512 \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    $USER-mixtral-8x7b-nemo-512 \
    $REPO_ROOT/src/helm-charts/a3ultra/jobset
```

#### Configure job settings

You can overwrite any of the default
[NeMo configurations](../../../../src/frameworks/a3ultra/nemo-configs/mixtral-8x7b-256gpus-a3u-bf16.yaml)
for this job. To do this, we can set the new arguments using `--set workload.arguments`.

**Examples**

To set the number of training steps to 100, run the following command:

```bash
helm install -f $RECIPE_ROOT/values.yaml \
    --set-file workload_launcher=$REPO_ROOT/src/launchers/nemo-10-launcher.sh \
    --set-file workload_config=$REPO_ROOT/src/frameworks/a3ultra/nemo-configs/mixtral-8x7b-256gpus-a3u-bf16.yaml \
    --set queue=${KUEUE_NAME} \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set workload.arguments[0]="trainer.max_steps=100" \
    $USER-mixtral-8x7b-nemo \
    $REPO_ROOT/src/helm-charts/a3ultra/jobset
```

### Monitor the job

To check the status of pods in your job, run the following command:

```
kubectl get pods | grep JOB_NAME_PREFIX
```

Replace the following:
- JOB_NAME_PREFIX - your job name prefix. For example $USER-mixtral-8x7b-nemo.

To get the logs for one of the pods, run the following command:

```
kubectl logs POD_NAME
```

Information about the training job's progress, including crucial details such as loss,
step count, and step time, is generated by the rank 0 process.
This process runs on the pod whose name begins with `JOB_NAME_PREFIX-workload-0-0`.
For example: `user-mixtral-8x7b-nemo-workload-0-0-s9zrv`.

### Analyze results

When completed, the job creates several artifacts, including logs and traces, and places them
in the configured Google Cloud Storage bucket as follows:

```
gs://${GCS_BUCKET}/nemo-experiments/JOB_ID
├── hparams.yaml
├── lightning_logs.txt
├── nemo_error_logs.txt
├── nemo_log_globalrank-[RANK]_localrank-[LOCAL].txt
├── dllogger
│   ├── rank-0
│   │   ├── dllogger.json
...
```

- `hparams.yaml`: the NeMo configuration used by the pretraining script. This includes
   the combined [configuration file](../../../../src/frameworks/a3ultra/nemo-configs/mixtral-8x7b-256gpus-a3u-bf16.yaml)
   and the command line overrides.
- `lightning_logs.txt`: the log files generated by PyTorch Lightning, which is used by NeMo.
- `nemo_error_logs.txt`: the warning and error logs generated by NeMo.
- `nemo_log_globalrank-[RANK]_localrank-[LOCAL].txt`: the NeMo logs for each rank.
- `dllogger/: The log captured by [NVIDIA DLLogger](https://github.com/NVIDIA/dllogger)`:
   DLLogger is configured to store logs on the rank 0 node. The log is in JSON format
   and includes loss, step_time, and other key metrics for each training step.

The JOB_ID has the following format:

$USER-mixtral-8x7b-nemo-[YYYY]-[MM]-[DD]-[hh]-[mm]-[ss], where the suffix of the ID is a day and time when the job was started.

Here is an example of an entry in the DLLogger log:

```json
DLLL {
  "timestamp": "1733239212.681539",
  "datetime": "2024-12-03 15:20:12.681539",
  "elapsedtime": "171.829225",
  "type": "LOG",
  "step": 26,
  "data": {
    "reduced_train_loss": 6.339119911193848,
    "lr": 0.0000040880504457163624,
    "global_step": 26,
    "consumed_samples": 27648,
    "train_backward_timing in s": 0.000040674211049918085,
    "train_step_timing in s": 2.7396187782287598,
    "epoch": 0
  }
}
```

The DLLogger log can be used to calculate the Model FLOPS Utilization (MFU) metric,
as described in the next section.

### Calculate training performance metrics (MFU, TFLOPS, Average Step Time)

This section explains how to calculate key training performance metrics,
such as Model FLOPS Utilization (MFU), using the `dllogger.json` file generated during training.

We provide a tool called [training_metrics](../../../../src/utils/training_metrics/) to help
you easily compute these metrics. This tool can calculate the following metrics:

- *MFU*: Model FLOPS Utilization
- *Average training step time*: the average time taken for each training step
- *TFLOPS per GPU*: the number of Tera Floating Point Operations per second achieved by each GPU

To calculate training performance metrics using the `training_metrics` tool, complete the
following steps:

1. Download the `dllogger.json` file. The `dllogger.json` file is generated during the
   training session.

    To download the file, run the following command. Replace `<JOB_ID>` with the ID of your
    training session.

    ```bash
    gcloud storage cp gs://${GCS_BUCKET}/nemo-experiments/<JOB_ID>/dllogger/rank-0/dllogger.json \
    $RECIPE_ROOT/dllogger.json
    ```

2. Run the
   [`process_training_results.py`](../../../../src/utils/training_metrics/process_training_results.py)
   script

   ```bash
   cd $REPO_ROOT/src/utils/training_metrics
   python3 process_training_results.py --file $RECIPE_ROOT/dllogger.json \
   --batch_size 1024 \
   --num_accelerators 256 \
   --precision bf16 \
   --model_type mixtral-7b \
   --accelerator_type h200
   ```

**Note:** The `batch_size`, `num_accelerators`, `precision`, `model_type` and `accelerator_type` are the
specific values for this recipe running the default configuration with 32 nodes. Average step time
is computed by default using the steps 10 to 30.

For more detailed information and advanced usage instructions of this tool,
see the [full documentation](../../../../src/utils/training_metrics/README.md)

### Troubleshooting

This section provides guidance on troubleshooting issues with the training job.

To check the status of the job's pods, use the following command:

```bash
kubectl get pods | grep JOB_NAME_PREFIX
```

Replace `JOB_NAME_PREFIX` with the prefix of your job name. For example, `$USER-mixtral-8x7b-nemo`. This command will list all pods associated with the specified job, along with their current status.


To get the logs from a specific pod, use the following command:

```bash
kubectl logs POD_NAME
```

Replace `POD_NAME` with the name of the pod you want to inspect.

In this recipe, the training job is orchestrated by the [Kubernetes JobSet](https://jobset.sigs.k8s.io/docs/overview/). If the JobSet encounters a fatal failure, it removes all pods, making it impossible to inspect their logs directly. To analyze logs from a failed job, retrieve them from Cloud Logging using the following filter:

```
resource.type="k8s_container"
resource.labels.project_id="PROJECT_ID"
resource.labels.location="CLUSTER_REGION"
resource.labels.cluster_name="CLUSTER_NAME"
resource.labels.namespace_name="default"
resource.labels.pod_name=~"^JOB_NAME_PREFIX.*"
severity>=DEFAULT
```

Replace the following:
- `PROJECT_ID`: your Google Cloud project ID.
- `CLUSTER_REGION`: the region where your cluster is located.
- `CLUSTER_NAME`: the name of your GKE cluster.
- `JOB_NAME_PREFIX`: the prefix of your job name (e.g., `$USER-mixtral-8x7b-nemo`).

This filter will retrieve logs from all containers within pods that match the job with the specified name prefix.

### Uninstall the Helm release

You can delete the job and other resources created by the Helm chart.
To uninstall Helm, run the following command from your client:

```bash
helm uninstall $USER-mixtral-8x7b-nemo
helm uninstall $USER-mixtral-8x7b-nemo-512
```

### Running the recipe on a cluster that does not use the default configuration.

If you created your cluster using the [GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-ultra.md), it is configured with default settings that include the names for networks and subnetworks used for communication between:

- The host to  external services.
- GPU-to GPU communication.

For clusters with this default configuration, the Helm chart can automatically generate the [required networking annotations in a Pod's metadata](https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom#configure-pod-manifests-rdma). Therefore, you can use the streamlined command to install the chart, as described in the the [Configure and submit a pretraining job](#configure-and-submit-a-pretraining-job) section.

To configure the correct networking annotations for a cluster that uses non-default names for GKE Network resources, provide the names of the GKE Network resources in you cluster  when installing the chart. Use the following example command. Be sure to replace the example values with the actual names of your cluster's GKE Network resources:

```bash
helm  install -f $RECIPE_ROOT/values.yaml \
    --set-file workload_launcher=$REPO_ROOT/src/launchers/nemo-10-launcher.sh \
    --set-file workload_config=$REPO_ROOT/src/frameworks/a3ultra/nemo-configs/mixtral-8x7b-256gpus-a3u-bf16.yaml \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set queue=${KUEUE_NAME} \
    --set network.subnetworks[0]=default \
    --set network.subnetworks[1]=gvnic-1 \
    --set network.subnetworks[2]=rdma-0 \
    --set network.subnetworks[3]=rdma-1 \
    --set network.subnetworks[4]=rdma-2 \
    --set network.subnetworks[5]=rdma-3 \
    --set network.subnetworks[6]=rdma-4 \
    --set network.subnetworks[7]=rdma-5 \
    --set network.subnetworks[8]=rdma-6 \
    --set network.subnetworks[9]=rdma-7 \
    $USER-mixtral-8x7b-nemo \
    $REPO_ROOT/src/helm-charts/a3ultra/jobset
```



