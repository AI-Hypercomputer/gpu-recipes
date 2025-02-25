# Multi node inference benchmark of DeepSeek R1 671B with SGLang on A3 Mega GKE Node Pool

This recipe outlines the steps to benchmark inference of a DeepSeek R1 671B model using [SGLang](https://github.com/sgl-project/sglang/tree/main) on an [A3 Mega GKE Node pool](https://cloud.google.com/kubernetes-engine) with multiple nodes.

In order to run this recipe, we use the [LeaderWorkerSet API](https://github.com/kubernetes-sigs/lws) in Kubernetes to spin up multiple nodes and handle distributed inference.

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- Job configuration and deployment - Helm chart is used to configure and deploy the
  [Kubernetes Index Job](https://kubernetes.io/blog/2021/04/19/introducing-indexed-jobs).
  This job encapsulates inference of DeepSeek R1 671B model using SGLang. The chart generates the manifest, adhering to best practices for using GPUDirect-TCPXO with Google Kubernetes Engine (GKE), which includes setting optimal values for NVIDIA NCCL and the TCPXO NCCL plugin.

## Prerequisites

Before running this recipe, ensure your environment is configured as follows:

- A GKE cluster with the following setup:
    - An A3 Mega node pool (2 nodes, 16 GPUs)
    - Topology-aware scheduling enabled
- An Artifact Registry repository to store the Docker image.
- A Google Cloud Storage (GCS) bucket to store results.
  *Important: This bucket must be in the same region as the GKE cluster*.
- A client workstation with the following pre-installed:
   - Google Cloud SDK
   - Helm
   - kubectl
- To access the [DeepSeek R1 671B model](https://huggingface.co/deepseek-ai/DeepSeek-R1) through Hugging Face, you'll need a Hugging Face token. Follow these steps to generate a new token if you don't have one already:
   - Create a [Hugging Face account](https://huggingface.co/), if you don't already have one.
   - Click Your **Profile > Settings > Access Tokens**.
   - Select **New Token**.
   - Specify a Name of your choice and a Role of at least `Read`.
   - Select **Generate a token**.
   - Copy the generated token to your clipboard.

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
  export SGLANG_IMAGE=sglang
  export SGLANG_VERSION=v0.4.3.post2-cu125-srt
  ```
  Replace the following values:

  - `<PROJECT_ID>`: your Google Cloud project ID
  - `<REGION>`: the region where you want to run Cloud Build
  - `<CLUSTER_REGION>`: the region where your cluster is located
  - `<CLUSTER_NAME>`: the name of your GKE cluster
  - `<GCS_BUCKET>`: the name of your Cloud Storage bucket. Do not include the `gs://` prefix
  - `<ARTIFACT_REGISTRY>`: the full name of your Artifact
    Registry in the following format: *LOCATION*-docker.pkg.dev/*PROJECT_ID*/*REPOSITORY*
  - `<SGLANG_IMAGE>`: the name of the SGLang image
  - `<SGLANG_VERSION>`: the version of the SGLang image

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
export RECIPE_ROOT=$REPO_ROOT/inference/a3mega/deepseek-r1-671b/sglang-serving-gke
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
    cd $REPO_ROOT/src/docker/sglang
    gcloud builds submit --region=${REGION} \
        --config cloudbuild.yml \
        --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY,_SGLANG_IMAGE=$SGLANG_IMAGE,_SGLANG_VERSION=$SGLANG_VERSION \
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

## Multi-node serving of DeepSeek R1 671B on A3 Mega nodes

The recipe serves DeepSeek R1 671B model using SGLang on multiple A3 Mega nodes in native FP8 mode

To start the serving, the recipe does the following steps:
1. Downloads the full DeepSeek R1 671B model checkpoints from [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1)
2. Starts SGLang server on two A3 Mega nodes, each with 8 GPUs and setting up necessary communication between the nodes.
3. Loads the model checkpoints on multiple nodes and apply SGLang optimizations.
4. Server is ready to respond to requests.

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

2. Install the LeaderWorkerSet API (LWS). Please follow the instructions [here](https://github.com/kubernetes-sigs/lws/blob/main/docs/setup/install.md#install-a-released-version) to install LWS.

3. Install the helm chart to prepare the model.

    ```bash
    cd $RECIPE_ROOT
    /usr/local/bin/helm/helm install -f values.yaml \
    --set job.image.repository=${ARTIFACT_REGISTRY}/${SGLANG_IMAGE} \
    --set clusterName=${CLUSTER_NAME} \
    --set job.image.tag=${SGLANG_VERSION} \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    $USER-serving-deepseek-r1-model \
    $REPO_ROOT/src/helm-charts/a3mega/sglang-inference
    ```

4. To view the logs for the deployment, you can run
    ```bash
    kubectl logs -f job/$USER-serving-deepseek-r1-model
    ```

5. Verify if the deployment has started by running
    ```bash
    kubectl get deployment/$USER-serving-deepseek-r1-model
    ```

6. Once the deployment has started, you will see logs similar to this:
    ```bash
    [2025-02-19 16:39:10 DP7 TP7] Prefill batch. #new-seq: 1, #new-token: 7, #cached-token: 0, cache hit rate: 0.00%, token usage: 0.00, #running-req: 0, #queue-req: 0
    [2025-02-19 16:39:11 DP7 TP7] Using configuration from /sgl-workspace/sglang/python/sglang/srt/layers/quantization/configs/N=576,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json for W8A8 Block FP8 kernel.
    [2025-02-19 16:39:16] INFO:     127.0.0.1:36440 - "POST /generate HTTP/1.1" 200 OK
    [2025-02-19 16:39:23] INFO:     127.0.0.1:36454 - "POST /generate HTTP/1.1" 200 OK
    [2025-02-19 16:39:29] INFO:     127.0.0.1:52874 - "POST /generate HTTP/1.1" 200 OK
    [2025-02-19 16:39:36] INFO:     127.0.0.1:52888 - "POST /generate HTTP/1.1" 200 OK
    [2025-02-19 16:39:42] INFO:     127.0.0.1:49466 - "POST /generate HTTP/1.1" 200 OK
    [2025-02-19 16:39:50] INFO:     127.0.0.1:53222 - "POST /generate HTTP/1.1" 200 OK
    [2025-02-19 16:39:56] INFO:     127.0.0.1:53238 - "POST /generate HTTP/1.1" 200 OK
    [2025-02-19 16:40:03] INFO:     127.0.0.1:53292 - "POST /generate HTTP/1.1" 200 OK
    [2025-02-19 16:40:10] INFO:     127.0.0.1:46284 - "POST /generate HTTP/1.1" 200 OK
    [2025-02-19 16:40:10] The server is fired up and ready to roll!
    ```

7. To make API requests to the service, you can port forward the service to your local machine.

    ```bash
    kubectl port-forward svc/$USER-serving-deepseek-r1-model-svc 30000:30000
    ```
8. Make the API requests to the service.

    ```bash
    curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model":"default",
      "messages":[
          {
            "role":"system",
            "content":"You are a helpful AI assistant"
          },
          {
            "role":"user",
            "content":"How many r are there in strawberry ?"
          }
      ],
      "temperature":0.6,
      "top_p":0.95,
      "max_tokens":2048
    }'
    ```

    If everything is setup correctly, you should a response similar to this:
    ```json
      {
        "id":"dd176721e73246b5a0ce0490fd9ba798",
        "object":"chat.completion",
        "created":1738368064,
        "model":"default",
        "choices":[
            {
              "index":0,
              "message":{
                  "role":"assistant",
                  "content":"<think>\nOkay, let's figure out how many times the letter \"r\" appears in the word \"strawberry.\" First, I need to spell out the word and check each letter one by one.\n\nSo, the word is S-T-R-A-W-B-E-R-R-Y. Let me write it out slowly to make sure I don't miss any letters. S, T, R, A, W, B, E, R, R, Y. Wait, let me count again. S (1), T (2), R (3), A (4), W (5), B (6), E (7), R (8), R (9), Y (10). Hmm, that's 10 letters total. Now, I need to count how many times the letter \"r\" appears.\n\nStarting from the beginning: S - no. T - no. R - that's the first R. Then A, W, B, E. Next comes R again, that's the second R. Then another R right after, so that's the third R. Finally, Y. So in total, there are three R's in \"strawberry.\"\n\nWait, let me double-check. Spelling it out: S-T-R-A-W-B-E-R-R-Y. The R is at the third position, then after E, there's two R's in a row. So that's three R's. Yeah, that seems right. I think that's correct. Maybe I should write it out again to confirm.\n\nS T R A W B E R R Y. Positions 3, 8, and 9 are R's. So three times. Yep, that's three R's. I don't think I missed any. The answer should be three.\n</think>\n\nThe word \"strawberry\" contains **3** instances of the letter **r**.  \n\n**Breakdown:**  \nS - T - **R** - A - W - B - E - **R** - **R** - Y.",
                  "tool_calls":null
              },
              "logprobs":null,
              "finish_reason":"stop",
              "matched_stop":1
            }
        ],
        "usage":{
            "prompt_tokens":17,
            "total_tokens":435,
            "completion_tokens":418,
            "prompt_tokens_details":null
        }
      }
    ```
    The thoughts of the models are enclosed in `<think>` tags which can be parsed out to get the reasoning of the model.

9. You may also make use of the utility script `stream_chat.sh` to stream responses in real time
    ```bash
    ./stream_chat.sh "Which is bigger 9.9 or 9.11 ?"
    ```

10. To run benchmarks for inference, you can use the default benchamrking tool from SGLang like this
    ```bash
    kubectl exec -it $USER-serving-deepseek-r1-model-0 -- /bin/bash -c "python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 1100 --random-input 1000 --random-output 1000 --host 0.0.0.0 --port 30000 --output-file /gcs/benchmark_logs/sglang/ds_1000_1000_1100_output.jsonl"
    ```

    Once the benchmark is done, you can find the results in the GCS Bucket. You should see logs similar to this:

    ```bash
    ============ Serving Benchmark Result ============
    Backend:                                 sglang
    Traffic request rate:                    inf
    Max reqeuest concurrency:                not set
    Successful requests:                     1100
    Benchmark duration (s):                  ...
    Total input tokens:                      1100000
    Total generated tokens:                  1100000
    Total generated tokens (retokenized):    1096494
    Request throughput (req/s):              xxx
    Input token throughput (tok/s):          xxxx
    Output token throughput (tok/s):         xxxx
    Total token throughput (tok/s):          xxxx
    Concurrency:                             xxx
    ----------------End-to-End Latency----------------
    Mean E2E Latency (ms):                   xxxxxxx
    Median E2E Latency (ms):                 xxxxxxx
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          xxxxxxx
    Median TTFT (ms):                        xxxxxxx
    P99 TTFT (ms):                           xxxxxxx
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          xxxxxxx
    Median TPOT (ms):                        xxxxxxx
    P99 TPOT (ms):                           xxxxxxx
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           xxxxxxx
    Median ITL (ms):                         xxxxxxx
    P99 ITL (ms):                            xxxxxxx
    ==================================================
    ```

### Cleanup

To clean up the resources created by this recipe, complete the following steps:

1. Uninstall the helm chart.

    ```bash
    /usr/local/bin/helm/helm uninstall $USER-serving-deepseek-r1-model
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
    --set job.image.repository=${ARTIFACT_REGISTRY}/${SGLANG_IMAGE} \
    --set job.image.tag=${SGLANG_VERSION} \
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
    $USER-serving-deepseek-r1-model \
    $REPO_ROOT/src/helm-charts/a3mega/sglang-inference
```