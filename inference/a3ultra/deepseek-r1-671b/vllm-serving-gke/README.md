# Single node inference benchmark of DeepSeek R1 671B with vLLM on A3 Ultra GKE Node Pool

This recipe outlines the steps to benchmark the inference of a DeepSeek R1 671B model using [vLLM](https://github.com/vllm-project/vllm) on an [A3 Ultra GKE Node pool](https://cloud.google.com/kubernetes-engine) with a single node.
## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- Job configuration and deployment - Helm chart is used to configure and deploy the
  [Kubernetes Index Job](https://kubernetes.io/blog/2021/04/19/introducing-indexed-jobs).
  This job encapsulates the inference of the DeepSeek R1 671B model using vLLM.
  The chart generates the job's manifest, which adhere to best practices for using RDMA Over Ethernet (RoCE)
  with Google Kubernetes Engine (GKE).

## Prerequisites

To prepare the required environment, see
[GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-ultra.md).

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
- To access the [DeepSeek R1 671B model](https://huggingface.co/deepseek-ai/DeepSeek-R1) through Hugging Face, you'll need a Hugging Face token. Follow these steps to generate a new token if you don't have one already:
   - Create a [Hugging Face account](https://huggingface.co/), if you don't already have one.
   - Click Your **Profile > Settings > Access Tokens**.
   - Select **New Token**.
   - Specify a Name and a Role of at least `Read`.
   - Select **Generate a token**.
   - Copy the generated token to your clipboard.

## Run the recipe

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
  - `<VLLM_VERSION>`: the version of the vLLM image

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
export RECIPE_ROOT=$REPO_ROOT/inference/a3ultra/deepseek-r1-671b/vllm-serving-gke
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

## Single A3 Ultra Node Serving of DeepSeek R1 671B

The recipe serves DeepSeek R1 671B model using vLLM on a single A3 Ultra node in native FP8 mode.

To start the serving, the recipe launches vLLM server that does the following steps:
1. Downloads the full DeepSeek R1 671B model checkpoints from [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1).
2. Loads the model checkpoints and apply vLLM optimizations.
3. Server is ready to respond to requests.

The recipe uses the helm chart to run the above steps.

1. Create Kubernetes Secret with a Hugging Face token to enable the job to download the model checkpoints.

    ```bash
    export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
    ```

    ```bash
    kubectl create secret generic hf-secret \
    --from-literal=hf_api_token=${HF_TOKEN} \
    --dry-run=client -o yaml | kubectl apply -f -
    ```

2. Install the helm chart to prepare the model.

    ```bash
    cd $RECIPE_ROOT
    helm install -f values.yaml \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set job.image.repository=${ARTIFACT_REGISTRY}/${VLLM_IMAGE} \
    --set job.image.tag=${VLLM_VERSION} \
    $USER-serving-deepseek-r1-model \
    $REPO_ROOT/src/helm-charts/a3ultra/vllm-inference
    ```

3. To view the logs for the deployment, run
    ```bash
    kubectl logs -f deployment/$USER-serving-deepseek-r1-model-serving
    ```

4. Verify if the deployment has started by running
    ```bash
    kubectl get deployment/$USER-serving-deepseek-r1-model-serving
    ```

5. Once the deployment has started, you'll see logs similar to:
    ```bash
    INFO 03-03 11:46:53 api_server.py:958] Starting vLLM API server on http://0.0.0.0:8000
    INFO 03-03 11:46:53 launcher.py:23] Available routes are:
    INFO 03-03 11:46:53 launcher.py:31] Route: /openapi.json, Methods: HEAD, GET
    INFO 03-03 11:46:53 launcher.py:31] Route: /docs, Methods: HEAD, GET
    INFO 03-03 11:46:53 launcher.py:31] Route: /docs/oauth2-redirect, Methods: HEAD, GET
    INFO 03-03 11:46:53 launcher.py:31] Route: /redoc, Methods: HEAD, GET
    INFO 03-03 11:46:53 launcher.py:31] Route: /health, Methods: GET
    INFO 03-03 11:46:53 launcher.py:31] Route: /ping, Methods: GET, POST
    INFO 03-03 11:46:53 launcher.py:31] Route: /tokenize, Methods: POST
    INFO 03-03 11:46:53 launcher.py:31] Route: /detokenize, Methods: POST
    INFO 03-03 11:46:53 launcher.py:31] Route: /v1/models, Methods: GET
    INFO 03-03 11:46:53 launcher.py:31] Route: /version, Methods: GET
    INFO 03-03 11:46:53 launcher.py:31] Route: /v1/chat/completions, Methods: POST
    INFO 03-03 11:46:53 launcher.py:31] Route: /v1/completions, Methods: POST
    INFO 03-03 11:46:53 launcher.py:31] Route: /v1/embeddings, Methods: POST
    INFO 03-03 11:46:53 launcher.py:31] Route: /pooling, Methods: POST
    INFO 03-03 11:46:53 launcher.py:31] Route: /score, Methods: POST
    INFO 03-03 11:46:53 launcher.py:31] Route: /v1/score, Methods: POST
    INFO 03-03 11:46:53 launcher.py:31] Route: /v1/audio/transcriptions, Methods: POST
    INFO 03-03 11:46:53 launcher.py:31] Route: /rerank, Methods: POST
    INFO 03-03 11:46:53 launcher.py:31] Route: /v1/rerank, Methods: POST
    INFO 03-03 11:46:53 launcher.py:31] Route: /v2/rerank, Methods: POST
    INFO 03-03 11:46:53 launcher.py:31] Route: /invocations, Methods: POST
    INFO:     Started server process [1]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    ```

6. To make API requests to the service, you can port forward the service to your local machine.

    ```bash
    kubectl port-forward svc/$USER-serving-deepseek-r1-model-svc 8000:8000
    ```
7. Make the API requests to the service.

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
      "max_tokens":128
    }'
    ```

    If everything is setup correctly, you should a response similar to this:
    ```json
    {
      "id":"chatcmpl-cb53f9c2200c47399698d7f3ac2512b7",
      "object":"chat.completion",
      "created":1741031403,
      "model":"deepseek-ai/DeepSeek-R1",
      "choices":[
          {
            "index":0,
            "message":{
                "role":"assistant",
                "reasoning_content":null,
                "content":"Okay, let's see. The user is asking how many times the letter \"r\" appears in the word \"strawberry\". Hmm, first, I need to make sure I spell the word correctly. Strawberry. S-T-R-A-W-B-E-R-R-Y. Let me write that out to check each letter.\n\nStarting with S, then T, then R. That's the first R. Next letters are A, W, B, E, then R again. Wait, after E, there's R-R-Y. So that's two R's in a row at the end. Let me count again. S-T-R (1), then later E-R-R (2 and 3). So that's three R's total? Wait, no, let me break it down step by step.\n\nBreaking the word into letters:\n\nS, T, R, A, W, B, E, R, R, Y.\n\nSo positions:\n\n1. S\n\n2. T\n\n3. R (1st R)\n\n4. A\n\n5. W\n\n6. B\n\n7. E\n\n8. R (2nd R)\n\n9. R (3rd R)\n\n10. Y\n\nWait, that's three R's? But when I say the word \"strawberry\", it's pronounced with two R sounds. But maybe the spelling has three? Let me check again. The word is spelled S-T-R-A-W-B-E-R-R-Y. So after the B and E, there are two R's before the Y. So that's R at position 3, then two R's at positions 8 and 9. Wait, that would make three R's. But that doesn't seem right. Maybe I'm miscounting. Let me write each letter with their order:\n\n1. S\n\n2. T\n\n3. R\n\n4. A\n\n5. W\n\n6. B\n\n7. E\n\n8. R\n\n9. R\n\n10. Y\n\nSo yes, positions 3, 8, and 9. That's three R's. But I thought strawberry had two R's. Maybe I'm confusing it with another word. Let me verify. Maybe the correct spelling is S-T-R-A-W-B-E-R-R-Y. So yes, two R's after the E. So in total, three R's. Wait, no. Let's see: S-T-R-A-W-B-E-R-R-Y. So after the E comes two R's and then Y. So that's two R's there, and the first R is in the third position. So total of three R's. Hmm. But when I look up the spelling, strawberry is spelled with two R's. Wait, maybe I'm making a mistake here. Let me check an actual dictionary.\n\n[Assuming the assistant can't access external resources, but from prior knowledge.] Okay, the correct spelling of strawberry is S-T-R-A-W-B-E-R-R-Y. So the letters are: S, T, R, A, W, B, E, R, R, Y. That's 10 letters. The R's are at positions 3, 8, and 9. So three R's. But wait, maybe I'm overcounting. Let's see: the word is straw-berry. \"Berry\" is spelled B-E-R-R-Y. So \"berry\" has two R's. Then the \"straw\" part has an R. So total of three R's. So the answer should be three. But maybe the user thinks it's two. Let me confirm again. S-T-R-A-W-B-E-R-R-Y. Yes, three R's. So the answer is 3.\n</think>\n\nThe word \"strawberry\" is spelled **S-T-R-A-W-B-E-R-R-Y**. Breaking it down:  \n- The third letter is **R**.  \n- The eighth and ninth letters are both **R** (from the \"berry\" part).  \n\nSo, there are **3** instances of the letter **r** in \"strawberry\". 🍓",
                "tool_calls":[

                ]
            },
            "logprobs":null,
            "finish_reason":"stop",
            "stop_reason":null
          }
      ],
      "usage":{
          "prompt_tokens":19,
          "total_tokens":868,
          "completion_tokens":849,
          "prompt_tokens_details":null
      },
      "prompt_logprobs":null
    }
    ```
    The model's thoughts are enclosed in `<think>` tags, which can be parsed out to get the reasoning of the model.

8. You may also make use of the utility script `stream_chat.sh` to stream responses in real time
    ```bash
    ./stream_chat.sh "Which is bigger 9.9 or 9.11 ?"
    ```

9. To run benchmarks for inference, use the default benchmarking tool from vLLM

    ```bash
    kubectl exec -it deployments/$USER-serving-deepseek-r1-model-serving -- python3 vllm/benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-R1 --dataset-name random --ignore-eos --num-prompts 1100 --random-input-len 1000 --random-output-len 1000 --port 8000 --backend vllm
    ```

    Once the benchmark is done, you can find the results in the GCS Bucket. You should see logs similar to:

    ```bash
    ============ Serving Benchmark Result ============
    Successful requests:                     xxxx
    Benchmark duration (s):                  xxx.xx
    Total input tokens:                      xxxxxxx
    Total generated tokens:                  xxxxxxx
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

### Running the recipe on a cluster that doesn't use the default configuration.

If you created your cluster using the [GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-ultra.md), it's configured with default settings that include the names for networks and subnetworks used for communication between:

- The host to external services
- GPU-to GPU communication

For clusters with this default configuration, the Helm chart can automatically generate the [required networking annotations in a Pod's metadata](https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom#configure-pod-manifests-rdma). Therefore, you can use the streamlined command to install the chart, as described in the the [Single A3 Ultra Node Benchmarking using FP8 Quantization](#single-a3-ultra-node-benchmarking-using-fp8-quantization) section.

To configure the correct networking annotations for a cluster that uses non-default names for GKE Network resources, you must provide the names of the GKE Network resources in your cluster when installing the chart. Use the following example command, remembering to replace the example values with the actual names of your cluster's GKE Network resources:

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
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
    $USER-benchmark-deepseek-r1-model \
    $REPO_ROOT/src/helm-charts/a3ultra/vllm-inference
```