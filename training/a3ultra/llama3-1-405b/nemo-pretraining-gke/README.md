# Pretrain Llama-3.1-405B workloads on A3 Ultra GKE Node pools

This recipe outlines the steps for running a Llama-3.1-405B pretraining workload on
[A3 Ultra GKE Node pools](https://cloud.google.com/kubernetes-engine) by using the
[NVIDIA NeMo framework](https://github.com/NVIDIA/nemo).

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- Job configuration and deployment - Helm chart is used to configure and deploy the
  [Kubernetes Index Job](https://kubernetes.io/blog/2021/04/19/introducing-indexed-jobs).
  This job encapsulates the
  [NVIDIA NeMo Megatron GPT pretraining workload](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_gpt_pretraining.py).
  The chart generates the job's manifest, adhering to best practices for using RDMA Over Ethernet (RoCE)
  with Google Kubernetes Engine (GKE).

## Test environment

This recipe has been optimized for and tested with the following configuration:

- A cluster with 72 [a3-ultragpu-8g](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) machines
- Machine placement in the cluster is configured using a [compact placement policy](https://cloud.google.com/kubernetes-engine/docs/how-to/compact-placement)
- [NVIDIA NeMo NGC container image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags): 24.07
- FP8/BF16 precision training
- Uses a mock pretraining dataset provided by the NeMo framework. By default, the job
  is configured to execute 30 training steps. If you want to change the number of training steps,
  see [Configure and submit a pretraining job](#configure-and-submit-a-pretraining-job).

## Prerequisites

Before running this recipe, ensure your environment is configured as follows:

- A GKE cluster with the following setup:
    - An A3 Ultra node pool (72 nodes, 576 GPUs)
    - Topology-aware scheduling enabled
- An Artifact Registry repository to store the Docker image.
- A Google Cloud Storage (GCS) bucket to store results.
  *Important: This bucket must be in the same region as the GKE cluster*.
- A client workstation with the following pre-installed:
   - Google Cloud SDK
   - Helm
   - kubectl

To prepare the required environment, see
[GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-ultra.md).

## Run the recipe

It is recommended to use Cloud Shell as your client to complete the steps.
Cloud Shell comes pre-installed with the necessary utilities, including
`kubectl`, `the Google Cloud SDK`, and `Helm`.

### Launch Cloud Shell

In the Google Cloud console, start a [Cloud Shell Instance](https://console.cloud.google.com/?cloudshell=true).

### Configure environment settings

From your client, complete the following steps:

1. Set the environment variables to match your environment:

  ```bash
  export PROJECT_ID=<PROJECT_ID>
  export REGION=<REGION>
  export CLUSTER_REGION=<CLUSTER_REGION>
  export CLUSTER_NAME=<CLUSTER_NAME>
  export GCS_BUCKET=<GCS_BUCKET>
  export KUEUE_NAME=<KUEUE_NAME>
  ```

  Replace the following values:

  - `<PROJECT_ID>`: your Google Cloud project ID
  - `<REGION>`: the region where you want to run Cloud Build
  - `<CLUSTER_REGION>`: the region where your cluster is located
  - `<CLUSTER_NAME>`: the name of your GKE cluster
  - `<GCS_BUCKET>`: the name of your Cloud Storage bucket. Do not include the `gs://` prefix
  - `<KUEUE_NAME>`: the name of the Kueue queue configured for TAS. The default queue created by the cluster toolkit is `a3-ultra`. Please check it that is the name of your local queue by running `kubectl get queues` and modify accordingly.

1. Set the default project:

  ```bash
  gcloud config set project $PROJECT_ID
  ```

### Get the recipe

From your client, clone the `gpu-recipes` repository and set a reference to the recipe folder.

```
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/training/a3ultra/llama3-1-405b/nemo-pretraining-gke
```

### Get cluster credentials

From your client, get the credentials for your cluster.

```
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

### Docker container

This recipe uses the following docker image: `us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo24.12-gib1.0.3-A3U`.

This image is based on NVIDIA NeMo 24.12 and contains the NCCL gIB plugin v1.0.3, bundling all NCCL binaries validated for use with A3 Ultra GPUs.

### Configure and submit a pretraining job

The default job setting is 30 training steps and fp8 precision. To execute the job with the
default settings, run the following command from your client:

```bash
cd $RECIPE_ROOT
helm  install -f values.yaml \
    --set-file nemo_config=$REPO_ROOT/src/frameworks/a3ultra/nemo-configs/llama3-1-405b-576gpus-a3ultra-fp8.yaml \
    --set workload.image=us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo24.12-gib1.0.3-A3U \
    --set queue=${KUEUE_NAME} \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    $USER-llama-3-1-405b-nemo-fp8 \
    $REPO_ROOT/src/helm-charts/a3ultra/nemo-training
```

- for BF16 precision:

```bash
cd $RECIPE_ROOT
helm  install -f values.yaml \
    --set-file nemo_config=$REPO_ROOT/src/frameworks/a3ultra/nemo-configs/llama3-1-405b-576gpus-a3ultra-bf16.yaml \
    --set workload.image=us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo24.12-gib1.0.3-A3U \
    --set queue=${KUEUE_NAME} \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    $USER-llama-3-1-405b-nemo-bf16 \
    $REPO_ROOT/src/helm-charts/a3ultra/nemo-training
```

#### Configure job settings

You can overwrite any of the default
[NeMo configurations fp8](../../../../src/frameworks/a3ultra/nemo-configs/llama3-1-405b-576gpus-a3ultra-fp8.yaml)
[NeMo configurations bf16](../../../../src/frameworks/a3ultra/nemo-configs/llama3-1-405b-576gpus-a3ultra-bf16.yaml)

for this job. To do this, we can set the new arguments using `--set workload.arguments`.

**Examples**

- To set the number of training steps to 100, run the following command from your client:

  ```bash
  cd $RECIPE_ROOT
  helm install -f values.yaml \
      --set-file nemo_config=$REPO_ROOT/src/frameworks/a3ultra/nemo-configs/llama3-1-405b-576gpus-a3ultra-fp8.yaml \
      --set workload.image=us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo24.12-gib1.0.3-A3U \
      --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
      --set queue=${KUEUE_NAME} \
      --set workload.arguments="{trainer.max_steps=100}" \
      $USER-llama-3-1-405b-nemo-fp8 \
      $REPO_ROOT/src/helm-charts/a3ultra/nemo-training
  ```


### Monitor the job

To check the status of pods in the indexed job, run the following command from your client:

```
kubectl get pods | grep $USER-llama-3-1-405b-nemo-fp8
```

To get the logs for one of the pods, run the following command from your client:

```
kubectl logs "<pod_name>"
```

### Analyze results

When completed, the job creates several artifacts, including logs and traces, and places them
in the configured Google Cloud Storage bucket as follows:

```
gs://${GCS_BUCKET}/nemo-experiments/megatron_gpt/<JOB_ID>
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
   the combined [configuration file](../../../../src/frameworks/a3ultra/nemo-configs/llama3-1-405b-576gpus-a3ultra-fp8.yaml)
   and the command line overrides
- `lightning_logs.txt`: the log files generated by PyTorch Lightning, which is used by NeMo
- `nemo_error_logs.txt`: the warning and error logs generated by NeMo
- `nemo_log_globalrank-[RANK]_localrank-[LOCAL].txt`: the NeMo logs for each rank
- `dllogger/: The log captured by [NVIDIA DLLogger](https://github.com/NVIDIA/dllogger)`:
   DLLogger is configured to store logs on the rank 0 node. The log is in JSON format
   and includes loss, step_time, and other key metrics for each training step

Here is an example of an entry in the DLLogger log:

```json
DLLL {
  "timestamp": "1739210331.717398",
  "datetime": "2025-02-10 17:58:51.717398",
  "elapsedtime": "1750.205571",
  "type": "LOG",
  "step": 16,
  "data":
  {
    "reduced_train_loss": 10.91420841217041,
    "lr": 1.2000000424450263e-06,
    "global_step": 16.0,
    "consumed_samples": 34272.0,
    "train_backward_timing in s": 4.649162292480469e-05,
    "train_step_timing in s": 92.28738164901733,
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
following steps command from your client:

1. Download the `dllogger.json` file. The `dllogger.json` file is generated during the
   training session.

    To download the file, run the following command. Replace `<JOB_ID>` with the ID of your
    training session.

    ```bash
    gcloud storage cp gs://${GCS_BUCKET}/nemo-experiments/megatron_gpt/<JOB_ID>/dllogger/rank-0/dllogger.json \
        $RECIPE_ROOT/dllogger.json
    ```

2. Run the
   [`process_training_results.py`](../../../../src/utils/training_metrics/process_training_results.py)
   script

   ```bash
   cd $REPO_ROOT/src/utils/training_metrics
   python3 process_training_results.py --file $RECIPE_ROOT/dllogger.json \
   --batch_size 2016 \
   --num_accelerators 576 \
   --precision fp8 \
   --model_type llama3.1-405b \
   --accelerator_type h200
   ```

**Note:** The `batch_size`, `num_accelerators`, `precision`, `model_type` and `accelerator_type` are the
specific values for this recipe running the default configuration. Average step time
is computed by default using the steps 10 to 30.

For more detailed information and advanced usage instructions of this tool,
see the [full documentation](../../../../src/utils/training_metrics/README.md)

### Uninstall the Helm release

You can delete the job and other resources created by the Helm chart.
To uninstall Helm, run the following command from your client:

```bash
helm uninstall $USER-llama-3-1-405b-nemo-fp8
helm uninstall $USER-llama-3-1-405b-nemo-bf16

```

### Running the recipe on a cluster that does not use the default configuration.

If you created your cluster using the [GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-ultra.md), it is configured with default settings that include the names for networks and subnetworks used for communication between:

- The host to  external services.
- GPU-to GPU communication.

For clusters with this default configuration, the Helm chart can automatically generate the [required networking annotations in a Pod's metadata](https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom#configure-pod-manifests-rdma). Therefore, you can use the streamlined command to install the chart, as described in the the [Configure and submit a pretraining job](#configure-and-submit-a-pretraining-job) section.

To configure the correct networking annotations for a cluster that uses non-default names for GKE Network resources, you must provide the names of the GKE Network resources in you cluster  when installing the chart. Use the following example command, remembering to replace the example values with the actual names of your cluster's GKE Network resources:

```bash
cd $RECIPE_ROOT
helm  install -f values.yaml \
    --set-file nemo_config=$REPO_ROOT/src/frameworks/a3ultra/nemo-configs/llama3-1-405b-576gpus-a3ultra-fp8.yaml \
    --set workload.image=us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo24.12-gib1.0.3-A3U \
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
    $USER-llama-3-1-405b-nemo-fp8 \
    $REPO_ROOT/src/helm-charts/a3ultra/nemo-training
```