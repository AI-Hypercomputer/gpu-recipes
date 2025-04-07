# Multi node inference benchmark of DeepSeek R1 671B with vLLM on A3 Mega GKE Node Pool

This recipe outlines the steps to benchmark inference of a DeepSeek R1 671B model using [vLLM](https://github.com/vllm-project/vllm) on an [A3 Mega GKE Node pool](https://cloud.google.com/kubernetes-engine) with multiple nodes.

The recipe uses [LeaderWorkerSet API](https://github.com/kubernetes-sigs/lws) in Kubernetes to spin up multiple nodes and handle distributed inference workload. LWS enables treating multiple Pods as a group, simplifying the management of distributed model serving.

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- LeaderWorkerSet Deployment - Helm chart is used to configure and deploy multi-node inference
  using the [LeaderWorkerSet API](https://github.com/kubernetes-sigs/lws) provisioning leader
  and worker pods for distributed inference of the DeepSeek R1 671B model using vLLM. The chart generates the manifest, adhering to best practices for using GPUDirect-TCPXO with Google Kubernetes Engine (GKE), which includes setting optimal values for NVIDIA NCCL and the TCPXO NCCL plugin.

## Prerequisites

Before running this recipe, ensure your environment is configured as follows:

- A GKE cluster with the following setup:
    - An A3 Mega node pool (2 nodes, 16 GPUs)
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
  export VLLM_IMAGE=vllm-openai
  export VLLM_VERSION=v0.7.3
  ```
  Replace the following values:

  - `<PROJECT_ID>`: your Google Cloud project ID
  - `<REGION>`: the region where you want to run Cloud Build
  - `<CLUSTER_REGION>`: the region where your cluster is located
  - `<CLUSTER_NAME>`: the name of your GKE cluster
  - `<GCS_BUCKET>`: the name of your Cloud Storage bucket. Do not include the `gs://` prefix
  - `<ARTIFACT_REGISTRY>`: the full name of your Artifact
    Registry in the following format: *LOCATION*-docker.pkg.dev/*PROJECT_ID*/*REPOSITORY*
  - `<VLLM_IMAGE>`: the name of the vLLM image
  - `<VLLM_VERSION>`: the version of the vLLM image. We recommended running the recipe with vLLM v0.7.3.

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
export RECIPE_ROOT=$REPO_ROOT/inference/a3mega/deepseek-r1-671b/vllm-serving-gke
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
    cd $REPO_ROOT/src/docker/vllm
    gcloud builds submit --region=${REGION} \
        --config cloudbuild.yml \
        --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY,_VLLM_IMAGE=$VLLM_IMAGE,_VLLM_VERSION=$VLLM_VERSION \
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

The recipe serves DeepSeek R1 671B model using vLLM on multiple A3 Mega nodes in native FP8 mode

To start the serving, the recipe does the following steps:
1. Downloads the full DeepSeek R1 671B model checkpoints from [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1)
2. Starts vLLM server on two A3 Mega nodes, each with 8 GPUs and setting up necessary communication between the nodes.
3. Loads the model checkpoints on multiple nodes and apply vLLM optimizations.
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

2. Install the LeaderWorkerSet API (LWS). Please follow the instructions [here](https://github.com/kubernetes-sigs/lws/blob/main/docs/setup/install.md#install-a-released-version) to install a specific version of LWS API.

    ```bash
    kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/latest/download/manifests.yaml
    ```

    Validate that the LeaderWorkerSet controller is running in the lws-system namespace, using the following command:

    ```bash
    kubectl get pod -n lws-system
    ```

    The output is similar to the following:
    ```bash
    NAME                                      READY   STATUS    RESTARTS   AGE
    lws-controller-manager-56956867cb-4km9g   1/1     Running   0          24h
    ```

3. Install the helm chart to prepare the model.

    ```bash
    cd $RECIPE_ROOT
    helm install -f values.yaml \
    --set job.image.repository=${ARTIFACT_REGISTRY}/${VLLM_IMAGE} \
    --set clusterName=${CLUSTER_NAME} \
    --set job.image.tag=${VLLM_VERSION} \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    $USER-serving-deepseek-r1-model \
    $REPO_ROOT/src/helm-charts/a3mega/vllm-inference/multi-host
    ```

4. To view the logs for the service, you can run
    ```bash
    kubectl logs -f service/$USER-serving-deepseek-r1-model-svc
    ```

5. Verify if the service has started by running
    ```bash
    kubectl get service/$USER-serving-deepseek-r1-model-svc
    ```

6. Once the deployment has started, you will see logs similar to this:
    ```bash
    INFO 03-03 15:00:50 api_server.py:958] Starting vLLM API server on http://0.0.0.0:8000
    INFO 03-03 15:00:50 launcher.py:23] Available routes are:
    INFO 03-03 15:00:50 launcher.py:31] Route: /openapi.json, Methods: HEAD, GET
    INFO 03-03 15:00:50 launcher.py:31] Route: /docs, Methods: HEAD, GET
    INFO 03-03 15:00:50 launcher.py:31] Route: /docs/oauth2-redirect, Methods: HEAD, GET
    INFO 03-03 15:00:50 launcher.py:31] Route: /redoc, Methods: HEAD, GET
    INFO 03-03 15:00:50 launcher.py:31] Route: /health, Methods: GET
    INFO 03-03 15:00:50 launcher.py:31] Route: /ping, Methods: POST, GET
    INFO 03-03 15:00:50 launcher.py:31] Route: /tokenize, Methods: POST
    INFO 03-03 15:00:50 launcher.py:31] Route: /detokenize, Methods: POST
    INFO 03-03 15:00:50 launcher.py:31] Route: /v1/models, Methods: GET
    INFO 03-03 15:00:50 launcher.py:31] Route: /version, Methods: GET
    INFO 03-03 15:00:50 launcher.py:31] Route: /v1/chat/completions, Methods: POST
    INFO 03-03 15:00:50 launcher.py:31] Route: /v1/completions, Methods: POST
    INFO 03-03 15:00:50 launcher.py:31] Route: /v1/embeddings, Methods: POST
    INFO 03-03 15:00:50 launcher.py:31] Route: /pooling, Methods: POST
    INFO 03-03 15:00:50 launcher.py:31] Route: /score, Methods: POST
    INFO 03-03 15:00:50 launcher.py:31] Route: /v1/score, Methods: POST
    INFO 03-03 15:00:50 launcher.py:31] Route: /v1/audio/transcriptions, Methods: POST
    INFO 03-03 15:00:50 launcher.py:31] Route: /rerank, Methods: POST
    INFO 03-03 15:00:50 launcher.py:31] Route: /v1/rerank, Methods: POST
    INFO 03-03 15:00:50 launcher.py:31] Route: /v2/rerank, Methods: POST
    INFO 03-03 15:00:50 launcher.py:31] Route: /invocations, Methods: POST
    INFO:     Started server process [12010]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    ```

7. To make API requests to the service, you can port forward the service to your local machine.

    ```bash
    kubectl port-forward svc/$USER-serving-deepseek-r1-model-svc 8000:8000
    ```

8. Make the API requests to the service.

    ```bash
    curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model":"deepseek-ai/DeepSeek-R1",
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
      "id": "chatcmpl-d598bb668aa94e8cb2663f107cce8031",
      "object": "chat.completion",
      "created": 1741042950,
      "model": "deepseek-ai/DeepSeek-R1",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "reasoning_content": null,
            "content": "Okay, the user is asking how many times the letter \"r\" appears in the word \"strawberry\". Let me start by writing out the word and checking each letter one by one.\n\nS-T-R-A-W-B-E-R-R-Y. Let me count each letter. The first letters are S, T, then R. That's one R. Then A, W, B, E, another R. That's two. Then another R right after, making it three. And finally Y. Wait, let me check again to make sure I didn't miss any. S-T-R (1), A-W-B-E-R (2), R (3), Y. So total of three R's? But wait, maybe I'm splitting it wrong. Let's break it down:\n\nStrawberry is spelled S-T-R-A-W-B-E-R-R-Y. The letters are S, T, R, A, W, B, E, R, R, Y. So positions 3, 8, and 9 are R's. That's three R's. But maybe the user thinks there are two? Let me confirm by writing the word again: s t r a w b e r r y. Yeah, after the B and E, there's R and then another R, so two R's at the end. Wait, that's two R's there. Let me count again. S (1), T (2), R (3), A (4), W (5), B (6), E (7), R (8), R (9), Y (10). So R is at positions 3, 8, and 9. That's three R's. But I thought it was two. Hmm, maybe I made a mistake. Let me look up the correct spelling of \"strawberry\". Yes, it's S-T-R-A-W-B-E-R-R-Y. So after the B and E, there are two R's. So total R's: one in the beginning (third letter) and two at the end (eighth and ninth letters). So that's three R's in total. Wait, but when I spell it out, it's S-T-R-A-W-B-E-R-R-Y. So R appears three times. But sometimes people might miscount. Let me verify once more. Each letter in order:\n\n1. S\n2. T\n3. R\n4. A\n5. W\n6. B\n7. E\n8. R\n9. R\n10. Y\n\nYes, three R's. So the answer is 3. But I need to make sure the user is asking for lowercase or uppercase, but the question is about the letter \"r\" in the word, which is case-insensitive here. So regardless of case, the count remains the same. Therefore, the correct answer is three.\n</think>\n\nThe word \"strawberry\" contains **3** instances of the letter **r**.  \n\nHere's the breakdown:  \nS - T - **R** - A - W - B - E - **R** - **R** - Y.",
            "tool_calls": []
          },
          "logprobs": null,
          "finish_reason": "stop",
          "stop_reason": null
        }
      ],
      "usage": {
        "prompt_tokens": 19,
        "total_tokens": 667,
        "completion_tokens": 648,
        "prompt_tokens_details": null
      },
      "prompt_logprobs": null
    }
    ```
    The thoughts of the models are enclosed in `<think>` tags which can be parsed out to get the reasoning of the model.

9. You may also make use of the utility script `stream_chat.sh` to stream responses in real time
    ```bash
    ./stream_chat.sh "Which is bigger 9.9 or 9.11 ?"
    ```

10. To run benchmarks for inference, you can use the default benchmarking tool from vLLM like this
    ```bash
    kubectl exec -it service/$USER-serving-deepseek-r1-model-svc -- python3 vllm/benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-R1 --dataset-name random --ignore-eos --num-prompts 1100 --random-input-len 1000 --random-output-len 1000 --port 8000 --backend vllm
    ```

    Once the benchmark is done, you can find the results in the logs similar to this:

    ```bash
    ============ Serving Benchmark Result ============
    Successful requests:                     1100
    Benchmark duration (s):                  xxx.xx
    Total input tokens:                      1100000
    Total generated tokens:                  1100000
    Request throughput (req/s):              x.xx
    Output token throughput (tok/s):         xxxx.xx
    Total Token throughput (tok/s):          xxxx.xx
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          xxxx.xx
    Median TTFT (ms):                        xxxx.xx
    P99 TTFT (ms):                           xxxx.xx
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          xxx.xx
    Median TPOT (ms):                        xxx.xx
    P99 TPOT (ms):                           xxx.xx
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           xxx.xx
    Median ITL (ms):                         xxx.xx
    P99 ITL (ms):                            xxx.xx
    ==================================================
    ```

### Cleanup

To clean up the resources created by this recipe, complete the following steps:

1. Uninstall the helm chart.

    ```bash
    helm uninstall $USER-serving-deepseek-r1-model
    ```

2. Delete the Kubernetes Secret.

    ```bash
    kubectl delete secret hf-secret
    ```

### Running the recipe on a cluster that does not use the default configuration.

If you created your cluster using the [GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-mega.md), it is configured with default settings that include the names for networks and subnetworks used for communication between:

- The host to external services.
- GPU-to GPU communication.

For clusters with this default configuration, the Helm chart can automatically generate the [required networking annotations in a Pod's metadata](https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom#configure-pod-manifests-rdma). Therefore, you can use the streamlined command to install the chart, as described in the the [Multi node inference benchmark of DeepSeek R1 671B with vLLM on A3 Mega GKE Node Pool](#multi-node-inference-benchmark-of-deepseek-r1-671b-with-vllm-on-a3-mega-gke-node-pool) section.

To configure the correct networking annotations for a cluster that uses non-default names for GKE Network resources, you must provide the names of the GKE Network resources in you cluster  when installing the chart. Use the following example command, remembering to replace the example values with the actual names of your cluster's GKE Network resources:

```bash
cd $RECIPE_ROOT
helm  install -f values.yaml \
    --set job.image.repository=${ARTIFACT_REGISTRY}/${VLLM_IMAGE} \
    --set job.image.tag=${VLLM_VERSION} \
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
    $REPO_ROOT/src/helm-charts/a3mega/vllm-inference
```
