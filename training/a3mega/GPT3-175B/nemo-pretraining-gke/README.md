# Pretraining GPT3-175B

This recipe outlines the steps for running a GPT-3 175B pretraining workload on the [A3 Mega machine series](https://cloud.google.com/kubernetes-engine) using the [NVIDIA NeMo framework](https://github.com/NVIDIA/nemo). [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine) is used for orchestration.

This recipe uses a Helm chart to configure and deploy a [Kubernetes Index Job](https://kubernetes.io/blog/2021/04/19/introducing-indexed-jobs). This job encapsulates the [NVIDIA NeMo Megatron GPT pretraining workload](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_gpt_pretraining.py). The chart generates the job's manifest, adhering to best practices for using GPUDirect-TCPXO with Google Kubernetes Engine (GKE). This includes setting optimal values for NVIDIA NCCL and the TCPXO NCCL plugin.

This recipe has been optimized for and tested with the following configuration:
- A cluster with 32 [a3-megagpu-8g](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms) machines configured using a [compact placement policy](https://cloud.google.com/kubernetes-engine/docs/how-to/compact-placement)
- [GPUDirect-TCPXO](https://cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx#required-features-capabilities) component versions:
    - NCCL Plugin: v1.0.3
    - RxDM sidecar: v1.0.9
- [NVIDIA NeMo NGC container image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags): 24.07
- FP8 precision training
- This recipe uses a mock pretraining dataset provided by the NeMo framework


By default, the job is configured to execute 50 training steps and can optionally capture an NVIDIA Nsight System profile. Refer to the following [instructions](#configure-and-submit-a-pretraining-job) if you want to change the number of training steps or enable Nsight System profiling.


## Environment setup

Before running this recipe, ensure you have the following environment configured:

- A GKE cluster with:
    - An A3 Mega node pool (32 nodes, 256 GPUs)
    - Topology-aware scheduling enabled
- An Artifact Registry repository to store the Docker image.
- A Google Cloud Storage (GCS) bucket to store results. *Important: This bucket must be in the same region as the GKE cluster*.
- A client workstation with the following pre-installed:
   - Google Cloud SDK
   - Helm
   - kubectl

Refer to the [GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-mega.md) for detailed instructions on how to prepare the required environment.

## Running the recipe

It is recommended to use Cloud Shell to walk through the recipe steps. Cloud Shell comes pre-installed with the necessary utilities, including `kubectl`, `the Google Cloud SDK`, and `Helm`.

### Launch Cloud Shell
In the Google Cloud console, start a [Cloud Shell Instance](https://console.cloud.google.com/?cloudshell=true).

### Configure environment settings

Set the environment variables to match your environment:

```bash
export PROJECT_ID=<PROJECT_ID>
export REGION=<REGION>
export CLUSTER_REGION=<CLUSTER_REGION>
export CLUSTER_NAME=<CLUSTER_NAME>
export GCS_BUCKET=<GCS_BUCKET>
export ARTIFACT_REGISTRY=<ARTIFACT_REGISTRY>
```
Replace the following values:
- `<PROJECT_ID>`: Your Google Cloud project ID
- `<REGION>`: The region where you want to run the Cloud Build
- `<CLUSTER_REGION>`: The region where your cluster is located
- `<CLUSTER_NAME>`: The name of your GKE cluster
- `<GCS_BUCKET>`: The name of your Cloud Storage bucket. Do not include the `gs://` prefix
- `<ARTIFACT_REGISTRY>`: The full name of your Artifact Registry in the following format: *LOCATION*-docker.pkg.dev/*PROJECT_ID*/*REPOSITORY*

Set the project:

```bash
gcloud config set project $PROJECT_ID
```

### Get the recipe

Clone the `gpu-recipes` repository and set a reference to the recipe folder.

```
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/training/a3mega/GPT3-175B/nemo-pretraining-gke
```

### Get cluster credentials:

```
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

### Build and push a docker container image to Artifact Registry

Use Cloud Build to build and push the container image.

```bash
cd $REPO_ROOT/src/docker/nemo-24.07
gcloud builds submit --region=${REGION} \
    --config cloudbuild.yml \
    --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY \
    --timeout "2h" \
    --machine-type=e2-highcpu-32 \
    --quiet \
    --async
```

This command outputs the build ID. You can monitor the build progress by streaming the logs using the following command:

```bash
BUILD_ID=<BUILD_ID>

gcloud beta builds log $BUILD_ID --region=$REGION
```

Replace <BUILD_ID> with your build ID.

### Configure and submit a pretraining job


 To execute the job with the default settings (50 training steps, no profiling) run the following command:


```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file nemo_config=$REPO_ROOT/src/frameworks/nemo-configs/gpt3-175b-256gpus-fp8.yaml \
    --set workload.image=${ARTIFACT_REGISTRY}/nemo_workload:24.07 \
    --set workload.gcsBucketForDataCataPath=${GCS_BUCKET} \
    $USER-gpt3-175b-nemo \
    $REPO_ROOT/src/helm-charts/nemo-training
```

You can overwrite any of the default [NeMo configurations](../../../../src/frameworks/nemo-configs/gpt3-175b-256gpus-fp8.yaml) for this job. In order to achieve this, we can set the new arguments using `--set workload.arguments`.

For example, to set the number of training steps to 100 use the following command:

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file nemo_config=$REPO_ROOT/src/frameworks/nemo-configs/gpt3-175b-256gpus-fp8.yaml \
    --set workload.image=${ARTIFACT_REGISTRY}/nemo_workload:24.07 \
    --set workload.gcsBucketForDataCataPath=${GCS_BUCKET} \
    --set workload.arguments="{trainer.max_steps=100}" \
    $USER-gpt3-175b-nemo \
    $REPO_ROOT/src/helm-charts/nemo-training
```

To enable Nsight Systems profiling:

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file nemo_config=$REPO_ROOT/src/frameworks/nemo-configs/gpt3-175b-256gpus-fp8.yaml \
    --set workload.image=${ARTIFACT_REGISTRY}/nemo_workload:24.07 \
    --set workload.gcsBucketForDataCataPath=${GCS_BUCKET} \
    --set workload.arguments="{model.nsys_profile.enabled=true,model.nsys_profile.start_step=41,model.nsys_profile.end_step=43}" \
    $USER-gpt3-175b-nemo \
    $REPO_ROOT/src/helm-charts/nemo-training
```

### Monitor the job

To check the status of pods in the indexed job:

```
kubectl get pods | grep $USER-gpt3-175b-nemo
```

To get the logs for one of the pods, execute the following command:

```
kubectl logs "<pod_name>"
```

### Analyze results

Upon completion, the job will create several artifacts, including logs and traces, in the configured GCS location:

```
gs://${GCP_BUCKET}/nemo-experiments/<JOB_ID>
├── hparams.yaml
├── lightning_logs.txt
├── nemo_error_logs.txt
├── noderank-0.nsys-rep
├── nemo_log_globalrank-[RANK]_localrank-[LOCAL].txt
├── dllogger
│   ├── rank-0
│   │   ├── dllogger.json
...
```

- hparams.yaml: The effective  NeMo configuration used by the pretraining script - (the combined [configuration file](../../../../src/frameworks/nemo-configs/gpt3-175b-256gpus-fp8.yaml) and the command line overrides)
- lightning_logs.txt: The log files generated by PyTorch Lightning, which is used by NeMo
- nemo_error_logs.txt: The warning and error logs generated by NeMo
- noderank-0.nsys-rep: If Nsight profiling was enabled, this file contains the profiling trace.
- nemo_log_globalrank-[RANK]_localrank-[LOCAL].txt: The NeMo logs for each rank.
- dllogger/: The log captured by [NVIDIA DLLogger](https://github.com/NVIDIA/dllogger). DLLogger is configured to store logs on the rank 0 node. The log is in JSON format and includes loss, step_time, and other key metrics for each training step..

Here is an example of an entry in the DLLogger log:

```json
DLLL {
  "timestamp": "1727463426.984522",
  "datetime": "2024-09-27 18:57:06.984522",
  "elapsedtime": "1645.119157",
  "type": "LOG",
  "step": 47,
  "data": {
    "reduced_train_loss": 7.890198707580566,
    "lr": 0.000036782606912311167,
    "global_step": 47,
    "consumed_samples": 98304,
    "train_backward_timing in s": 0.00004844665454584174,
    "grad_norm": 8.800999641418457,
    "train_step_timing in s": 25.115550994873047,
    "epoch": 0
  }
}
```

The DLLogger log can be used to calculate the Model FLOPS Utilization (MFU) metric, as described in the next section.

### Calculate Training Performance Metrics (MFU, TFLOPS, Average Step Time)

This section explains how to calculate key training performance metrics, such as Model FLOPS Utilization (MFU), using the `dllogger.json` file generated during training.

We provide a tool called [training_metrics](../../../../src/utils/training_metrics/) to help you easily compute these metrics. It can calculate:
- *MFU:* Model FLOPS Utilization
- *Average training step time:* The average time taken for each training step.
- *TFLOPS per GPU:* The number of Tera Floating Point Operations per second achieved by each GPU.

Using the `training_metrics` tool:

**1. Download dllogger.json file**

Download the `dllogger.json` file generated during the training session.
To download the file:
- Modify the command below, replacing <JOB_ID> with the ID of your training session.
- Run the command in your terminal or cloud shell.

```bash
gcloud storage cp gs://${GCS_BUCKET}/nemo-experiments/<JOB_ID>/dllogger/rank-0/dllogger.json \
    /path/to/your/local/dllogger.json
```

**2. Run the [`process_training_results.py`](../../../../src/utils/training_metrics/process_training_results.py) script:**
```bash
cd $REPO_ROOT/src/utils/training_metrics
python3 process_training_results.py --file /path/to/your/local/dllogger.json \
--batch_size 2048 \
--num_accelerators 256 \
--model_type gpt3-175b \
--accelerator_type h100
```
**Note:** The batch_size, num_accelerators, model_type and accelerator_type are the specific values for this recipe running the default configuration. Average step time is computed by default using the steps 10 to 30.

For more detailed information and advanced usage instructions of this tool, refer to the [full documentation](../../../../src/utils/training_metrics/README.md)

### Uninstall the Helm release

You can delete the job and other resources created by the Helm chart using the following command:

```bash
helm uninstall $USER-gpt3-175b-nemo
```
