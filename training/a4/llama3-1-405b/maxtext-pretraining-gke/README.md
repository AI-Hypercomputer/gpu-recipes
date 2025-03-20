# Pretrain Llama-3.1-405B workloads on A4 GKE Node pools using MaxText

This recipe outlines the steps for running a Llama-3.1-405B pretraining workload
on [A4 GKE Node pools](https://cloud.google.com/kubernetes-engine) by using the
[MaxText framework](https://github.com/AI-Hypercomputer/maxtext).

## Orchestration and deployment tools

For this recipe, the following setup is used:

-   Orchestration -
    [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
-   Job configuration and deployment - Helm chart is used to configure and
    deploy the
    [Kubernetes Index Job](https://kubernetes.io/blog/2021/04/19/introducing-indexed-jobs).
    This job encapsulates the
    [MaxText pretraining workload](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/train.py).
    The chart generates the job's manifest, adhering to best practices for using
    RDMA Over Ethernet (RoCE) with Google Kubernetes Engine (GKE).

## Test environment

This recipe has been optimized for and tested with the following configuration:

-   A cluster with 32
    [a4-highgpu-8g](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)
    machines.
-   Machine placement in the cluster is configured using a
    [compact placement policy](https://cloud.google.com/kubernetes-engine/docs/how-to/compact-placement)
-   FP8 precision training
-   Uses a synthetic pretraining dataset provided by the MaxText framework. By
    default, the job is configured to execute 15 training steps. If you want to
    change the number of training steps, see
    [Configure and submit a pretraining job](#configure-and-submit-a-pretraining-job).

## Prerequisites

Before running this recipe, ensure your environment is configured as follows:

-   A GKE cluster with the following setup:
    -   An A4 node pool (32 nodes - 256 GPUs)
    -   Kueue Topology-aware scheduling enabled
-   A Google Cloud Storage (GCS) bucket to store results. *Important: This
    bucket must be in the same region as the GKE cluster*.
-   A client workstation with the following pre-installed:
    -   Google Cloud SDK
    -   Helm
    -   kubectl

To prepare the required environment, see
[GKE environment setup guide](../../../../docs/configuring-environment-gke-a4.md).

## Run the recipe

It is recommended to use Cloud Shell as your client to complete the steps. Cloud
Shell comes pre-installed with the necessary utilities, including `kubectl`,
`the Google Cloud SDK`, and `Helm`.

### Launch Cloud Shell

In the Google Cloud console, start a
[Cloud Shell Instance](https://console.cloud.google.com/?cloudshell=true).

### Configure environment settings

From your client, complete the following steps:

1.  Set the environment variables to match your environment:

    ```bash
    export PROJECT_ID=<PROJECT_ID>
    export REGION=<REGION>
    export CLUSTER_REGION=<CLUSTER_REGION>
    export CLUSTER_NAME=<CLUSTER_NAME>
    export GCS_BUCKET=<GCS_BUCKET>
    export KUEUE_NAME=<KUEUE_NAME>
    ```

    Replace the following values:

    -   `<PROJECT_ID>`: your Google Cloud project ID
    -   `<REGION>`: the region where you want to run Cloud Build
    -   `<CLUSTER_REGION>`: the region where your cluster is located
    -   `<CLUSTER_NAME>`: the name of your GKE cluster
    -   `<GCS_BUCKET>`: the name of your Cloud Storage bucket. Do not include
        the `gs://` prefix
    -   `<KUEUE_NAME>`: the name of the Kueue queue configured for TAS. The
        default queue created by the cluster toolkit is `a4-high`. Please verify
        the name of your local queue by running `kubectl get queues` and modify
        it as needed.

1.  Set the default project:

    ```bash
    gcloud config set project $PROJECT_ID
    ```

### Get the recipe

From your client, clone the `gpu-recipes` repository and set a reference to the
recipe folder.

```
cd
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/training/a4/llama3-1-405b/maxtext-pretraining-gke
```

### Get cluster credentials

From your client, get the credentials for your cluster.

```
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

### Configure and submit a pretraining job

#### Using 32 nodes (256 GPUs)

The default job setting is 15 training steps and fp8 precision. To execute the
job with the default settings, run the following command from your client:

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file maxtext_config=$REPO_ROOT/src/frameworks/a4/maxtext-configs/llama3-1-405b-256gpus-a4-fp8.yaml \
    --set workload.image=us-central1-docker.pkg.dev/deeplearning-images/reproducibility/jax-maxtext-gpu:jax0.5.1-cuda_dl25.02-rev1-maxtext-20150317  \
    --set workload.run_name=$USER-llama-3-1-405b-maxtext-fp8 \
    --set workload.gpus=256 \
    --set queue=$KUEUE_NAME \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    $USER-llama-3-1-405b-maxtext-fp8 \
    $REPO_ROOT/src/helm-charts/a4/maxtext-training
```

#### Configure job settings

**Examples**

-   To set the number of training steps to 100, run the following command from
    your client:

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file maxtext_config=$REPO_ROOT/src/frameworks/a4/maxtext-configs/llama3-1-405b-256gpus-a4-fp8.yaml \
    --set workload.image=us-central1-docker.pkg.dev/deeplearning-images/reproducibility/jax-maxtext-gpu:jax0.5.1-cuda_dl25.02-rev1-maxtext-20150317  \
    --set workload.run_name=$USER-llama-3-1-405b-maxtext-fp8 \
    --set workload.gpus=256 \
    --set queue=$KUEUE_NAME \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set workload.steps=100 \
    $USER-llama-3-1-405b-maxtext-fp8 \
    $REPO_ROOT/src/helm-charts/a4/maxtext-training
```

### Monitor the job

To check the status of pods in the indexed job, run the following command from
your client:

```
kubectl get pods | grep $USER-llama-3-1-405b-maxtext-fp8
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

To inspect the text logs generated by MaxText, retrieve them from any Pod in the
job using the following command: `kubectl logs "<pod_name>"`

Here is an example of an entry in :

```
completed step: 11, seconds: 28.294, TFLOP/s/device: 1520.276, Tokens/s/device: 579.059, total_weights: 4194304, loss: 13.652
```

The logs will show you the step time in seconds and the TFLOP/s/device.

### Calculate training performance metrics (eMFU)

This section explains how to calculate the effective Model FLOPS Utilization
(eMFU), using the logs from the pods. Using the example logs from the previous
step, and considering the number of TFLOP/s/device of 903.017, you can compute
the eMFU using the following formula:

```
           TFLOP/s/device        1520.276
eMFU =   ------------------- =  --------- = 0.6796 = 67.96%
             MAX TFLOP B200        2237

```

MAX TFLOP B200 BF16: 2237

### Uninstall the Helm release

You can delete the job and other resources created by the Helm chart. To
uninstall Helm, run the following command from your client:

```bash
helm uninstall $USER-llama-3-1-405b-maxtext-fp8
```
