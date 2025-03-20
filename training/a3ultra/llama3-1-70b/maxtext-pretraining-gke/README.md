# Pretrain Llama-3.1-70B workloads on A3 Ultra GKE Node pools using MaxText

This recipe outlines the steps for running a Llama-3.1-70B pretraining workload on
[A3 Ultra GKE Node pools](https://cloud.google.com/kubernetes-engine) by using the
[MaxText framework](https://github.com/AI-Hypercomputer/maxtext).

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- Job configuration and deployment - Helm chart is used to configure and deploy the
  [Kubernetes Index Job](https://kubernetes.io/blog/2021/04/19/introducing-indexed-jobs).
  This job encapsulates the
  [MaxText pretraining workload](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/train.py).
  The chart generates the job's manifest, adhering to best practices for using RDMA Over Ethernet (RoCE) with Google Kubernetes Engine (GKE).

## Test environment

This recipe has been optimized for and tested with the following configuration:

- A cluster with 32 or 64 [a3-ultragpu-8g](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) machines.
- Machine placement in the cluster is configured using a [compact placement policy](https://cloud.google.com/kubernetes-engine/docs/how-to/compact-placement)
- MaxText docker container
- BF16 and FP8 precision training
- Uses a synthetic pretraining dataset provided by the MaxText framework. By default, the job
  is configured to execute 50 training steps. If you want to change the number of training steps,
  see [Configure and submit a pretraining job](#configure-and-submit-a-pretraining-job).

## Prerequisites

Before running this recipe, ensure your environment is configured as follows:

- A GKE cluster with the following setup:
    - An A3 Ultra node pool (32 nodes - 256 GPUs or 64 nodes - 512 GPUs)
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
  export ARTIFACT_REGISTRY=<ARTIFACT_REGISTRY>
  export KUEUE_NAME=<KUEUE_NAME>
  ```

  Replace the following values:

  - `<PROJECT_ID>`: your Google Cloud project ID
  - `<REGION>`: the region where you want to run Cloud Build
  - `<CLUSTER_REGION>`: the region where your cluster is located
  - `<CLUSTER_NAME>`: the name of your GKE cluster
  - `<GCS_BUCKET>`: the name of your Cloud Storage bucket. Do not include the `gs://` prefix
  - `<ARTIFACT_REGISTRY>`: the full name of your Artifact
    Registry in the following format: *LOCATION*-docker.pkg.dev/*PROJECT_ID*/*REPOSITORY*
  - `<KUEUE_NAME>`: the name of the Kueue queue configured for TAS. The default queue created by the cluster toolkit is `a3-ultra`. Please verify the name of your local queue by running `kubectl get queues` and modify it as needed.
1. Set the default project:

  ```bash
  gcloud config set project $PROJECT_ID
  ```

### Get the recipe

From your client, clone the `gpu-recipes` repository and set a reference to the recipe folder.

```
cd
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/training/a3ultra/llama3-1-70b/maxtext-pretraining-gke
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

#### Using 32 nodes (256 GPUs)

The default job setting is 50 training steps and bf16 precision. To execute the job with the
default settings, run the following command from your client:

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file maxtext_config=$REPO_ROOT/src/frameworks/a3ultra/maxtext-configs/llama3-1-70b-256gpus-a3u-bf16.yaml \
    --set workload.image=${ARTIFACT_REGISTRY}/maxtext-benchmark \
    --set workload.run_name=$USER-llama-3-1-70b-maxtext \
    --set workload.gpus=256 \
    --set queue=$KUEUE_NAME \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    $USER-llama-3-1-70b-maxtext \
    $REPO_ROOT/src/helm-charts/a3ultra/maxtext-training
```

To run the recipe on `fp8` precision, run the following command from your client:
```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file maxtext_config=$REPO_ROOT/src/frameworks/a3ultra/maxtext-configs/llama3-1-70b-256gpus-a3u-fp8.yaml \
    --set workload.image=${ARTIFACT_REGISTRY}/maxtext-benchmark \
    --set workload.run_name=$USER-llama-3-1-70b-maxtext-fp8 \
    --set workload.gpus=256 \
    --set queue=$KUEUE_NAME \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    $USER-llama-3-1-70b-maxtext-fp8 \
    $REPO_ROOT/src/helm-charts/a3ultra/maxtext-training
```

#### Using 64 nodes (512 GPUs)

The default job setting is 50 training steps and bf16 precision. To execute the job with the
default settings, run the following command from your client:

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file maxtext_config=$REPO_ROOT/src/frameworks/a3ultra/maxtext-configs/llama3-1-70b-512gpus-a3u-bf16.yaml \
    --set workload.image=${ARTIFACT_REGISTRY}/maxtext-benchmark \
    --set workload.run_name=$USER-llama-3-1-70b-maxtext-64nodes \
    --set workload.gpus=512 \
    --set queue=$KUEUE_NAME \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    $USER-llama-3-1-70b-maxtext-64nodes \
    $REPO_ROOT/src/helm-charts/a3ultra/maxtext-training
```

To run the recipe on `fp8` precision, run the following command from your client:
```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file maxtext_config=$REPO_ROOT/src/frameworks/a3ultra/maxtext-configs/llama3-1-70b-512gpus-a3u-fp8.yaml \
    --set workload.image=${ARTIFACT_REGISTRY}/maxtext-benchmark \
    --set workload.run_name=$USER-llama-3-1-70b-maxtext-fp8-64nodes \
    --set workload.gpus=512 \
    --set queue=$KUEUE_NAME \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    $USER-llama-3-1-70b-maxtext-fp8-64nodes \
    $REPO_ROOT/src/helm-charts/a3ultra/maxtext-training
```

#### 64 nodes (512 GPUs) global batch size 2048

The default job setting is 50 training steps and fp8 precision. To execute the job with the
default settings, run the following command from your client:

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file maxtext_config=$REPO_ROOT/src/frameworks/a3ultra/maxtext-configs/llama3-1-70b-512gpus-a3u-fp8-gbs2048.yaml \
    --set workload.image=${ARTIFACT_REGISTRY}/maxtext-benchmark \
    --set workload.run_name=$USER-llama-3-1-70b-maxtext-fp8-64nodes-2048 \
    --set workload.gpus=512 \
    --set queue=$KUEUE_NAME \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    $USER-llama-3-1-70b-maxtext-fp8-64nodes-2048 \
    $REPO_ROOT/src/helm-charts/a3ultra/maxtext-training
```

#### 128 nodes (1024 GPUs) global batch size 2048

The default job setting is 50 training steps and fp8 precision. To execute the job with the
default settings, run the following command from your client:

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file maxtext_config=$REPO_ROOT/src/frameworks/a3ultra/maxtext-configs/llama3-1-70b-1024gpus-a3u-fp8-gbs2048.yaml \
    --set workload.image=${ARTIFACT_REGISTRY}/maxtext-benchmark \
    --set workload.run_name=$USER-llama-3-1-70b-maxtext-fp8-128nodes-2048 \
    --set workload.gpus=1024 \
    --set queue=$KUEUE_NAME \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    $USER-llama-3-1-70b-maxtext-fp8-128nodes-2048 \
    $REPO_ROOT/src/helm-charts/a3ultra/maxtext-training
```

#### Configure job settings

**Examples**

- To set the number of training steps to 100, run the following command from your client:

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file maxtext_config=$REPO_ROOT/src/frameworks/a3ultra/maxtext-configs/llama3-1-70b-256gpus-a3u-bf16.yaml \
    --set workload.image=${ARTIFACT_REGISTRY}/maxtext-benchmark \
    --set workload.run_name=$USER-llama-3-1-70b-maxtext \
    --set queue=$KUEUE_NAME \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set workload.steps=100 \
    $USER-llama-3-1-70b-maxtext \
    $REPO_ROOT/src/helm-charts/a3ultra/maxtext-training
```

### Monitor the job

To check the status of pods in the indexed job, run the following command from your client:

```
kubectl get pods | grep $USER-llama-3-1-70b-maxtext
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


Here is an example of an entry in :

```
completed step: 12, seconds: 15.516, TFLOP/s/device: 508.371, Tokens/s/device: 1055.949, total_weights: 4194304, loss: 0.000
```

The logs will show you the step time in seconds and the TFLOP/s/device.

### Calculate training performance metrics (MFU)

This section explains how to calculate the Model FLOPS Utilization (MFU), using the logs from the pods.
Using the example logs from the previous step, and considering the number of TFLOP/s/device of 508.371,
you can compute the MFU using the following formula:

```
           TFLOP/s/device       508.371
MFU =   ------------------- =  --------- = 0.514 = 51.4%
             MAX TFLOP H200       989

```

MAX TFLOP H200:

- BF16: 989
- FP8: 1979


### Uninstall the Helm release

You can delete the job and other resources created by the Helm chart.
To uninstall Helm, run the following command from your client:

```bash
helm uninstall $USER-llama-3-1-70b-maxtext
helm uninstall $USER-llama-3-1-70b-maxtext-64nodes
```

or for fp8:

```bash
helm uninstall $USER-llama-3-1-70b-maxtext-fp8
helm uninstall $USER-llama-3-1-70b-maxtext-fp8-64nodes
```

### Running the recipe on a cluster that does not use the default configuration.

If you created your cluster using the [GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-ultra.md), it is configured with default settings that include the names for networks and subnetworks used for communication between:

- The host to  external services.
- GPU-to GPU communication.

For clusters with this default configuration, the Helm chart can automatically generate the [required networking annotations in a Pod's metadata](https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom#configure-pod-manifests-rdma). Therefore, you can use the streamlined command to install the chart, as described in the the [Configure and submit a pretraining job](#configure-and-submit-a-pretraining-job) section.

To configure the correct networking annotations for a cluster that uses non-default names for GKE Network resources, you must provide the names of the GKE Network resources in you cluster  when installing the chart. Use the following example command, remembering to replace the example values with the actual names of your cluster's GKE Network resources:

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file maxtext_config=$REPO_ROOT/src/frameworks/a3ultra/maxtext-configs/llama3-1-70b-256gpus-a3u-bf16.yaml \
    --set workload.image=${ARTIFACT_REGISTRY}/maxtext-benchmark \
    --set workload.run_name=$USER-llama-3-1-70b-maxtext \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set queue=$KUEUE_NAME \
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
    $USER-llama-3-1-70b-maxtext \
    $REPO_ROOT/src/helm-charts/a3ultra/maxtext-training
```