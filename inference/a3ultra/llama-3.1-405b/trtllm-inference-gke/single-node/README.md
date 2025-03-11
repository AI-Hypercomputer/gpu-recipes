# Single node inference benchmark of Llama 3.1 405B with NVIDIA TensorRT-LLM (TRT-LLM) on A3 Ultra GKE Node Pool

This recipe outlines the steps to benchmark inference of a Llama 3.1 405B model using [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) on an [A3 Ultra GKE Node pool](https://cloud.google.com/kubernetes-engine) with a single node.
## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- Job configuration and deployment - Helm chart is used to configure and deploy the
  [Kubernetes Index Job](https://kubernetes.io/blog/2021/04/19/introducing-indexed-jobs).
  This job encapsulates inference of Llama 3.1 405B model using TensorRT-LLM.
  The chart generates the job's manifest, adhering to best practices for using RDMA Over Ethernet (RoCE)
  with Google Kubernetes Engine (GKE).

## Prerequisites

Before running this recipe, ensure your environment is configured as follows:

- A GKE cluster with the following setup:
    - An A3 Ultra node pool (1 node, 8 GPUs)
    - Topology-aware scheduling enabled
- An Artifact Registry repository to store the Docker image.
- A Google Cloud Storage (GCS) bucket to store results.
  *Important: This bucket must be in the same region as the GKE cluster*.
- A client workstation with the following pre-installed:
   - Google Cloud SDK
   - Helm
   - kubectl
- To access the [Llama 3.1 405B model](https://huggingface.co/meta-llama/Llama-3.1-405B) through Hugging Face, you'll need a Hugging Face token. Follow these steps to generate a new token if you don't have one already:
   - Create a [Hugging Face account](https://huggingface.co/), if you don't already have one.
   - Click Your **Profile > Settings > Access Tokens**.
   - Select **New Token**.
   - Specify a Name of your choice and a Role of at least `Read`.
   - Select **Generate a token**.
   - Copy the generated token to your clipboard.
- Get access to the Llama 3.1 405B model checkpoints from Hugging Face.
   - You can get access to the Llama 3.1 405B model checkpoints by signing up for a Hugging Face account and [joining the Llama 3.1 405B model family.](https://huggingface.co/meta-llama/Llama-3.1-405B)

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
  export TRT_LLM_IMAGE=trtllm
  export TRT_LLM_VERSION=0.16.0
  ```
  Replace the following values:

  - `<PROJECT_ID>`: your Google Cloud project ID
  - `<REGION>`: the region where you want to run Cloud Build
  - `<CLUSTER_REGION>`: the region where your cluster is located
  - `<CLUSTER_NAME>`: the name of your GKE cluster
  - `<GCS_BUCKET>`: the name of your Cloud Storage bucket. Do not include the `gs://` prefix
  - `<ARTIFACT_REGISTRY>`: the full name of your Artifact
    Registry in the following format: *LOCATION*-docker.pkg.dev/*PROJECT_ID*/*REPOSITORY*
  - `<TRT_LLM_IMAGE>`: the name of the TensorRT-LLM image
  - `<TRT_LLM_VERSION>`: the version of the TensorRT-LLM image

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
export RECIPE_ROOT=$REPO_ROOT/inference/a3ultra/llama-3.1-405b/trtllm-inference-gke/single-node
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
    cd $REPO_ROOT/src/docker/trtllm-0.16.0
    gcloud builds submit --region=${REGION} \
        --config cloudbuild.yml \
        --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY,_TRT_LLM_IMAGE=$TRT_LLM_IMAGE,_TRT_LLM_VERSION=$TRT_LLM_VERSION \
        --timeout "2h" \
        --machine-type=e2-highcpu-32 \
        --disk-size=1000 \
        --quiet \
        --async
    ```
  This command outputs the `build ID`.

2. You can monitor the build progress by streaming the logs for the `build ID`.
   To do this, run the following command.

   Replace `<BUILD_ID>` with your build ID.

   ```bash
   BUILD_ID=<BUILD_ID>

   gcloud beta builds log $BUILD_ID --region=$REGION
   ```


## Single A3 Ultra Node Benchmarking using FP8 Quantization
The recipe runs inference benchmark for Llama 3.1 405B model on a single A3 Ultra node converting the Hugging Face checkpoint to TensorRT-LLM optimized format with FP8 quantization.

### Run the benchmarking using `trtllm-bench`

[`trtllm-bench`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-benchmarking.md) is a command-line tool from NVIDIA that can be used to benchmark the performance of TensorRT-LLM engine.
For more information about `trtllm-bench`, see the [TensorRT-LLM documentation](https://github.com/NVIDIA/TensorRT-LLM).

To run the benchmarking, the recipe does the following steps:

1. Download the full Llama 3.1 405B model checkpoints from [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-405B). Please see the [prerequisites](#Prerequisites) section to get access to the model.
2. Convert the model checkpoints to TensorRT-LLM optimized format
3. Build TensorRT-LLM engines for the model with FP8 quantization
4. Run the throughput and latency benchmarking

The recipe uses the helm chart to run the above steps.

1. Create Kubernetes Secret with Hugging Face token to allow the job to download the model checkpoints.

    ```bash
    export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
    ```

    ```bash
    kubectl create secret generic hf-secret \
    --from-literal=hf_api_token=${HF_TOKEN} \
    --dry-run=client -o yaml | kubectl apply -f -
    ```

2. Install the helm chart to prepare the model.

    NOTE: This helm chart currently runs only a single experiment for 30k requests for 128 tokens of input/output lengths. To run other experiments, you can uncomment the various combinations provided in the [values.yaml](values.yaml) file.

    ```bash
    cd $RECIPE_ROOT
    helm install -f values.yaml \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set job.image.repository=${ARTIFACT_REGISTRY}/${TRT_LLM_IMAGE} \
    --set job.image.tag=${TRT_LLM_VERSION} \
    $USER-benchmark-llama-model \
    $REPO_ROOT/src/helm-charts/a3ultra/trtllm-inference/single-node
    ```

3. To view the logs for the job, you can run
    ```bash
    kubectl logs -f job/$USER-benchmark-llama-model
    ```

4. Verify the job has completed by running

    ```bash
    kubectl get job/$USER-benchmark-llama-model
    ```

    If the job has completed, you should see the following output:

    ```bash
    NAME                COMPLETIONS   DURATION   AGE
    $USER-benchmark-llama-model   1/1     ##s     #m##s
    ```

5. Once the job starts running, you will see logs similar to this:
    ```bash
    Running benchmark for meta-llama/Llama-3.1-405B with ISL=128, OSL=128, TP=8
    [TensorRT-LLM] TensorRT-LLM version: 0.16.0
    Parse safetensors files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 191/191 [00:01<00:00, 101.48it/s]
    [01/21/2025-02:18:19] [TRT-LLM] [I] Found dataset.
    [01/21/2025-02:18:20] [TRT-LLM] [I]
    ===========================================================
    = DATASET DETAILS
    ===========================================================
    Max Input Sequence Length:      128
    Max Output Sequence Length:     128
    Max Sequence Length:    256
    Target (Average) Input Sequence Length: 128
    Target (Average) Output Sequence Length:        128
    Number of Sequences:    30000
    ===========================================================


    [01/21/2025-02:18:20] [TRT-LLM] [I] Max batch size and max num tokens are not provided, use tuning heuristics or pre-defined setting from trtllm-bench.
    [01/21/2025-02:18:20] [TRT-LLM] [I] Estimated total available memory for KV cache: 717.36 GB
    [01/21/2025-02:18:20] [TRT-LLM] [I] Estimated total KV cache memory: 681.49 GB
    [01/21/2025-02:18:20] [TRT-LLM] [I] Estimated max number of requests in KV cache memory: 11076.91
    [01/21/2025-02:18:20] [TRT-LLM] [I] Set dtype to bfloat16.
    [01/21/2025-02:18:20] [TRT-LLM] [I] Set multiple_profiles to True.
    [01/21/2025-02:18:20] [TRT-LLM] [I] Set use_paged_context_fmha to True.
    [01/21/2025-02:18:20] [TRT-LLM] [I] Set use_fp8_context_fmha to True.
    [01/21/2025-02:18:20] [TRT-LLM] [I]
    ===========================================================
    = ENGINE BUILD INFO
    ===========================================================
    Model Name:             meta-llama/Llama-3.1-405B
    Model Path:             /ssd/meta-llama/Llama-3.1-405B
    Workspace Directory:    /ssd
    Engine Directory:       /ssd/meta-llama/Llama-3.1-405B/tp_8_pp_1

    ===========================================================
    = ENGINE CONFIGURATION DETAILS
    ===========================================================
    Max Sequence Length:            256
    Max Batch Size:                 4096
    Max Num Tokens:                 8192
    Quantization:                   FP8
    ===========================================================

    [TensorRT-LLM] TensorRT-LLM version: 0.16.0
    [TensorRT-LLM] TensorRT-LLM version: 0.16.0
    [TensorRT-LLM] TensorRT-LLM version: 0.16.0
    [TensorRT-LLM] TensorRT-LLM version: 0.16.0
    [TensorRT-LLM] TensorRT-LLM version: 0.16.0
    [TensorRT-LLM] TensorRT-LLM version: 0.16.0
    [TensorRT-LLM] TensorRT-LLM version: 0.16.0
    [TensorRT-LLM] TensorRT-LLM version: 0.16.0
    Loading Model: [1/2]    Loading HF model to memory
    Loading checkpoint shards: 100%|██████████| 191/191 [02:48<00:00,  1.14it/s]
    Inserted 2649 quantizers
    Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
    Disable lm_head quantization for TRT-LLM export due to deployment limitations.
    current rank: 0, tp rank: 0, pp rank: 0
    Time: 1723.944s
    Loading Model: [2/2]    Building TRT-LLM engine
    Time: 600.103s
    Loading model done.
    Total latency: 2324.046s

    [TensorRT-LLM] TensorRT-LLM version: 0.16.0
    [01/22/2025-08:40:31] [TRT-LLM] [I] Preparing to run throughput benchmark...
    [01/22/2025-08:40:33] [TRT-LLM] [I] Setting up benchmarker and infrastructure.
    [01/22/2025-08:40:33] [TRT-LLM] [I] Initializing Throughput Benchmark. [rate=-1 req/s]
    [01/22/2025-08:40:33] [TRT-LLM] [I] Ready to start benchmark.
    [01/22/2025-08:40:33] [TRT-LLM] [I] Initializing Executor.

    01/22/2025-08:41:37] [TRT-LLM] [I] WAITING ON EXECUTOR...
    [01/22/2025-08:41:37] [TRT-LLM] [I] Starting response daemon...
    [01/22/2025-08:41:37] [TRT-LLM] [I] Executor started.
    [01/22/2025-08:41:37] [TRT-LLM] [I] WAITING ON BACKEND TO BE READY...
    [01/22/2025-08:41:37] [TRT-LLM] [I] Request serving started.
    [01/22/2025-08:41:37] [TRT-LLM] [I] Starting statistics collection.
    [01/22/2025-08:41:37] [TRT-LLM] [I] Collecting live stats...
    [01/22/2025-08:41:37] [TRT-LLM] [I] Benchmark started.
    [01/22/2025-08:41:37] [TRT-LLM] [I] Request serving stopped.
    [01/22/2025-08:57:55] [TRT-LLM] [I] Collecting last stats...
    [01/22/2025-08:57:55] [TRT-LLM] [I] Ending statistics collection.
    [01/22/2025-08:57:55] [TRT-LLM] [I] Stop received.
    [01/22/2025-08:57:55] [TRT-LLM] [I] Stopping response parsing.
    [01/22/2025-08:57:55] [TRT-LLM] [I] Collecting last responses before shutdown.
    [01/22/2025-08:57:55] [TRT-LLM] [I] Completed request parsing.
    [01/22/2025-08:57:55] [TRT-LLM] [I] Parsing stopped.
    [01/22/2025-08:57:55] [TRT-LLM] [I] Request generator successfully joined.
    [01/22/2025-08:57:55] [TRT-LLM] [I] Statistics process successfully joined.
    [01/22/2025-08:57:55] [TRT-LLM] [I]

    ===========================================================
    = ENGINE DETAILS
    ===========================================================
    Model:			meta-llama/Llama-3.1-405B
    Engine Directory:	/ssd/meta-llama/Llama-3.1-405B/tp_8_pp_1
    TensorRT-LLM Version:	0.16.0
    Dtype:			bfloat16
    KV Cache Dtype:		FP8
    Quantization:		FP8
    Max Sequence Length:	256

    ===========================================================
    = WORLD + RUNTIME INFORMATION
    ===========================================================
    TP Size:		8
    PP Size:		1
    Max Runtime Batch Size:	4096
    Max Runtime Tokens:	8192
    Scheduling Policy:	Guaranteed No Evict
    KV Memory Percentage:	95.00%
    Issue Rate (req/sec):	9.5448E+12

    ===========================================================
    = PERFORMANCE OVERVIEW
    ===========================================================
    Number of requests:		30000
    Average Input Length (tokens):	128.0000
    Average Output Length (tokens):	128.0000
    Token Throughput (tokens/sec):	3926.0575
    Request Throughput (req/sec):	30.6723
    Total Latency (ms):		978080.4323

    ===========================================================

    [01/22/2025-08:57:55] [TRT-LLM] [I] Benchmark Shutdown called!
    [01/22/2025-08:57:55] [TRT-LLM] [I] Shutting down ExecutorServer.
    [TensorRT-LLM][INFO] Orchestrator sendReq thread exiting
    [TensorRT-LLM][INFO] Orchestrator recv thread exiting
    [01/22/2025-08:57:55] [TRT-LLM] [I] Executor shutdown.
    ```

6. Once the job has completed, you can see the results in the Cloud Storage bucket.

    ```bash
    gsutil ls gs://${GCS_BUCKET}/benchmark_logs/
    ```

### Cleanup

To clean up the resources created by this recipe, complete the following steps:

1. Uninstall the helm chart.

    ```bash
    helm uninstall $USER-benchmark-llama-model
    ```

2. Delete the Kubernetes Secret.

    ```bash
    kubectl delete secret hf-secret
    ```

### Running the recipe on a cluster that does not use the default configuration.

If you created your cluster using the [GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-ultra.md), it is configured with default settings that include the names for networks and subnetworks used for communication between:

- The host to  external services.
- GPU-to GPU communication.

For clusters with this default configuration, the Helm chart can automatically generate the [required networking annotations in a Pod's metadata](https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom#configure-pod-manifests-rdma). Therefore, you can use the streamlined command to install the chart, as described in the the [Single A3 Ultra Node Benchmarking using FP8 Quantization](#single-a3-ultra-node-benchmarking-using-fp8-quantization) section.

To configure the correct networking annotations for a cluster that uses non-default names for GKE Network resources, you must provide the names of the GKE Network resources in you cluster  when installing the chart. Use the following example command, remembering to replace the example values with the actual names of your cluster's GKE Network resources:

```bash
cd $RECIPE_ROOT
helm  install -f values.yaml \
    --set workload.image=${ARTIFACT_REGISTRY}/${TRT_LLM_IMAGE}:${TRT_LLM_VERSION} \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
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
    $USER-benchmark-llama-model \
    $REPO_ROOT/src/helm-charts/a3ultra/trtllm-inference/single-node
```