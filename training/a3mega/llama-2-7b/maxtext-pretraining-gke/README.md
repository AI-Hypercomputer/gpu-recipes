# Pretrain Llama-2-7B workloads on A3 Mega GKE Node pools using MaxText


This recipe outlines the steps for running a Llama-2-7B pretraining workload on
[A3 Mega GKE Node pools](https://cloud.google.com/kubernetes-engine) by using the
[MaxText framework](https://github.com/AI-Hypercomputer/maxtext).

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- Job configuration and deployment - Helm chart is used to configure and deploy the
  [Kubernetes Index Job](https://kubernetes.io/blog/2021/04/19/introducing-indexed-jobs).
  This job encapsulates the
  [MaxText pretraining workload](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/train.py).
  The chart generates the job's manifest, adhering to best practices for using GPUDirect-TCPXO with Google Kubernetes Engine (GKE), which includes setting optimal values for NVIDIA NCCL and the TCPXO NCCL plugin.

## Test environment

This recipe has been optimized for and tested with the following configuration:

- A cluster with 32 or 16 [a3-megagpu-8g](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms) machines.
- Machine placement in the cluster is configured using a [compact placement policy](https://cloud.google.com/kubernetes-engine/docs/how-to/compact-placement)
- MaxText docker container
- BF16 precision training
- Uses a synthetic pretraining dataset provided by the MaxText framework. By default, the job
  is configured to execute 50 training steps. If you want to change the number of training steps,
  see [Configure and submit a pretraining job](#configure-and-submit-a-pretraining-job).

## Prerequisites

Before running this recipe, ensure your environment is configured as follows:

- A GKE cluster with the following setup:
    - An A3 Mega node pool (32 nodes, 256 GPUs / 16 nodes, 128 GPUs)
    - Topology-aware scheduling enabled
- An Artifact Registry repository to store the Docker image.
- A Google Cloud Storage (GCS) bucket to store results.
  *Important: This bucket must be in the same region as the GKE cluster*.
- A client workstation with the following pre-installed:
   - Google Cloud SDK
   - Helm
   - kubectl

To prepare the required environment, see
[GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-mega.md).

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
  export ARTIFACT_REGISTRY=<ARTIFACT_REGISTRY>
  ```

  Replace the following values:

  - `<PROJECT_ID>`: your Google Cloud project ID
  - `<REGION>`: the region where you want to run Cloud Build
  - `<CLUSTER_REGION>`: the region where your cluster is located
  - `<CLUSTER_NAME>`: the name of your GKE cluster
  - `<GCS_BUCKET>`: the name of your Cloud Storage bucket. Do not include the `gs://` prefix
  - `<ARTIFACT_REGISTRY>`: the full name of your Artifact
    Registry in the following format: *LOCATION*-docker.pkg.dev/*PROJECT_ID*/*REPOSITORY*

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
export RECIPE_ROOT=$REPO_ROOT/training/a3mega/llama-2-7b/maxtext-pretraining-gke
```

### Get cluster credentials

From your client, get the credentials for your cluster.

```
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

### Build and push a docker container image to Artifact Registry

To build the container, complete the following steps from your client:

1. Use Cloud Build to build and push the container image.

    ```bash
    cd $REPO_ROOT/src/docker/maxtext
    gcloud builds submit --region=${REGION} \
        --config cloudbuild.yml \
        --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY \
        --timeout "2h" \
        --machine-type=e2-highcpu-32 \
        --quiet \
        --async
    ```

  This command outputs the `build ID`.

1. You can monitor the build progress by streaming the logs for the `build ID`.
   To do this, run the following command.

   Replace `<BUILD_ID>` with your build ID.

   ```bash
   BUILD_ID=<BUILD_ID>

   gcloud beta builds log $BUILD_ID --region=$REGION
   ```


### Configure and submit a pretraining job

The default job setting is 30 training steps and bf16 precision over 32 nodes. To execute the job with the
default settings, run the following command from your client:

```bash
cd $RECIPE_ROOT
export SESSION_ID=`date "+%Y-%m-%d-%H-%M-%S"`
helm install -f values.yaml \
    --set-file maxtext_config=$REPO_ROOT/src/frameworks/a3mega/maxtext-configs/llama-2-7b-256gpus-a3mega-bf16.yaml \
    --set sessionid=${SESSION_ID} \
    --set workload.image=${ARTIFACT_REGISTRY}/maxtext-nightly \
    --set workload.run_name=$USER-llama-2-7b-maxtext \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set volumes.gcsMounts[0].name=${USER}-${GCS_BUCKET} \
    $USER-llama-2-7b-256gpus-maxtext \
    $REPO_ROOT/src/helm-charts/a3mega/maxtext-training
```


To run the recipe on 16 nodes, run the following command from your client:
```bash
cd $RECIPE_ROOT
export SESSION_ID=`date "+%Y-%m-%d-%H-%M-%S"`
helm install -f values.yaml \
    --set-file maxtext_config=$REPO_ROOT/src/frameworks/a3mega/maxtext-configs/llama-2-7b-128gpus-a3mega-bf16.yaml \
    --set sessionid=${SESSION_ID} \
    --set workload.image=${ARTIFACT_REGISTRY}/maxtext-nightly \
    --set workload.run_name=$USER-llama-2-7b-maxtext \
    --set workload.gpus=128 \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set volumes.gcsMounts[0].name=${USER}-${GCS_BUCKET} \
    $USER-llama-2-7b-128gpus-maxtext \
    $REPO_ROOT/src/helm-charts/a3mega/maxtext-training
```



#### Configure job settings

**Examples**

- To set the number of training steps to 100, run the following command from your client:

```bash
cd $RECIPE_ROOT
export SESSION_ID=`date "+%Y-%m-%d-%H-%M-%S"`
helm install -f values.yaml \
    --set-file maxtext_config=$REPO_ROOT/src/frameworks/a3mega/maxtext-configs/llama-2-7b-256gpus-a3mega-bf16.yaml \
    --set sessionid=${SESSION_ID} \
    --set workload.image=${ARTIFACT_REGISTRY}/maxtext-nightly \
    --set workload.run_name=$USER-llama-2-7b-maxtext \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set volumes.gcsMounts[0].name=${USER}-${GCS_BUCKET} \
    --set workload.steps=100 \
    $USER-llama-2-7b-256gpus-maxtext \
    $REPO_ROOT/src/helm-charts/a3mega/maxtext-training
```

### Monitor the job

To check the status of pods in the indexed job, run the following command from your client:

```
kubectl get pods | grep $USER-llama-2-7b-256gpus-maxtext
```

To get the logs for one of the pods, run the following command from your client:

```
kubectl logs "<pod_name>"
```

### Analyze results

When completed, the job creates tensorboard logs in the following location:

```
gs://${GCS_BUCKET}/maxtext/$JOB_ID/tensorboard/$JOB_ID/
├── events.out.tfevents....
...
```

To inspect the text logs generated by MaxText, retrieve them from any Pod in the job using the following command:
 `kubectl logs "<pod_name>"`


Here is an example of an entry in the logs:

```
completed step: 12, seconds: 1.500, TFLOP/s/device: 503.384, Tokens/s/device: 10922.973, total_weights: 4194304, loss: 0.000
```

The logs will show you the step time in seconds and the TFLOP/s/device.

### Calculate training performance metrics (MFU)

This section explains how to calculate the Model FLOPS Utilization (MFU), using the logs from the pods.
Using the example logs from the previous step, and considering the number of TFLOP/s/device of 503.384,
you can compute the MFU using the following formula:

```
           TFLOP/s/device       503.384
MFU =   ------------------- =  --------- = 0.514 = 51.4%
             MAX TFLOP H100       989

```

MAX TFLOP H100:

- BF16: 989


### Uninstall the Helm release

You can delete the job and other resources created by the Helm chart.
To uninstall Helm, run the following command from your client:

```bash
helm uninstall $USER-llama-2-7b-256gpus-maxtext
```

or for 16 nodes run:

```bash
helm uninstall $USER-llama-2-7b-128gpus-maxtext
```
