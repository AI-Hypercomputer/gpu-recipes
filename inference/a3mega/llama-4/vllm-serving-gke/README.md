# Single node inference benchmark of Llama 4 models with vLLM on A3 Mega GKE Node Pool

This recipe outlines the steps to benchmark inference of a Llama 4 herd of models using [vLLM](https://github.com/vllm-project/vllm) on an [A3 Mega GKE Node pool](https://cloud.google.com/kubernetes-engine) on a single node.

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- Kubernetes Deployment - Helm chart is used to configure and deploy the
  [Kubernetes Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment).
  This deployment encapsulates the inference of the Llama 4 models using vLLM.
  The chart generates the manifest, adhering to best practices for using GPUDirect-TCPXO with Google Kubernetes Engine (GKE), which includes setting optimal values for NVIDIA NCCL and the TCPXO NCCL plugin with Google Kubernetes Engine (GKE).

## Prerequisites

Before running this recipe, ensure your environment is configured as follows:

- A GKE cluster with the following setup:
    - An A3 Mega node pool (1 nodes, 8 GPUs)
- An Artifact Registry repository to store the Docker image.
- A Google Cloud Storage (GCS) bucket to store results.
  *Important: This bucket must be in the same region as the GKE cluster*.
- A client workstation with the following pre-installed:
   - Google Cloud SDK
   - Helm
   - kubectl
- To access the [Llama 4 models](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) through Hugging Face, you'll need to get access to the gated model and create the Hugging Face token. Follow these steps to generate a new token if you don't have one already:
   - Create a [Hugging Face account](https://huggingface.co/), if you don't already have one.
   - Click Your **Profile > Settings > Access Tokens**.
   - Select **New Token**.
   - Specify a Name and a Role of at least `Read`.
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
  export VLLM_VERSION=v0.8.3
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
  - `<VLLM_VERSION>`: the version of the vLLM image. We recommended running the recipe with vLLM v0.8.3.

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
export RECIPE_ROOT=$REPO_ROOT/inference/a3mega/llama-4/vllm-serving-gke
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

## Single node serving of Llama 4 models on a A3 Mega node

The recipe serves various Llama 4 models using vLLM on a single A3 Mega nodes in full precision and FP8 mode.

Llama 4 models are offered in various sizes and precision and have 17B tokens when activated. This recipe is compatible with the following set of models.

| Model Name | Total Size | Precision | Context Length |
|------------|------|-----------|----------------|
| [Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E) | 109B | BF16 | 1M |
| [Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | 109B | BF16 | 1M |
| [Llama-4-Maverick-17B-128E](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E) | 400B | FP8 | 430k |
| [Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) | 400B | FP8 | 430k|


To start the serving, the recipe does the following steps:
1. Downloads the full model checkpoints from Hugging Face.
2. Loads the model checkpoints and apply vLLM optimizations.
3. Server is ready to respond to requests.

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

2. Install the helm chart to serve the model

    a. To serve the `Llama-4-Scout-17B-16E` and `Llama-4-Scout-17B-16E-Instruct` models, run
      ```bash
        cd $RECIPE_ROOT
        helm install -f values.yaml \
        --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
        --set job.image.repository=${ARTIFACT_REGISTRY}/${VLLM_IMAGE} \
        --set model.name=meta-llama/Llama-4-Scout-17B-16E-Instruct \
        --set vllm.serverArgs.max-model-len=1000000 \
        --set job.image.tag=${VLLM_VERSION} \
        $USER-serving-llama-4 \
        $REPO_ROOT/src/helm-charts/a3mega/vllm-inference/single-host
      ```

      b. To serve the `Llama-4-Scout-17B-128E-Instruct-FP8` model, run
      ```bash
        cd $RECIPE_ROOT
        helm install -f values.yaml \
        --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
        --set job.image.repository=${ARTIFACT_REGISTRY}/${VLLM_IMAGE} \
        --set model.name=meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
        --set vllm.serverArgs.max-model-len=430000 \
        --set job.image.tag=${VLLM_VERSION} \
        $USER-serving-llama-4 \
        $REPO_ROOT/src/helm-charts/a3mega/vllm-inference/single-host
      ```

3. To view the logs for the deployment, you can run
    ```bash
    kubectl logs -f deployment/$USER-serving-llama-4-serving
    ```

4. Verify if the service has started by running
    ```bash
    kubectl get deployment/$USER-serving-llama-4-serving
    ```

5. Once the deployment has started, you will see logs similar to this:
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

6. To make API requests to the service, you can port forward the service to your local machine.

    ```bash
    kubectl port-forward svc/$USER-serving-llama-4-svc 8000:8000
    ```

7. Make the API requests to the service.

    ```bash
    curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model":"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
      "messages":[
          {
            "role":"system",
            "content":"You are a helpful AI assistant"
          },
          {
            "role":"user",
            "content":"What is the meaning of life ?"
          }
      ],
      "temperature":0.6,
      "top_p":0.95,
      "max_tokens":2048
    }' \
    -o response.json
    ```

    If everything is setup correctly, you should a response similar to this:
    ```json
    {
      "id": "chatcmpl-1c8390497fb6409aac2ae5da4681611b",
      "object": "chat.completion",
      "created": 1743981666,
      "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "reasoning_content": null,
            "content": "The meaning of life is a profound and subjective question that has been debated and explored by philosophers, theologians, scientists, and thinkers across cultures and centuries. There is no one definitive answer, as it can vary greatly from person to person based on individual beliefs, values, experiences, and cultural backgrounds. However, here are some perspectives:\n\n1. **Philosophical Perspectives**: Philosophers have offered various interpretations. Existentialists like Jean-Paul Sartre and Martin Heidegger suggest that life has no inherent meaning, and it's up to each individual to create their own purpose. In contrast, others like Aristotle believed that the meaning of life is to achieve eudaimonia, often translated as happiness or flourishing, through living a virtuous life.\n\n2. **Religious and Spiritual Perspectives**: Many religions offer a sense of meaning that is tied to fulfilling a divine purpose or achieving spiritual enlightenment or salvation. For example, in Christianity, the meaning of life is often seen as loving and serving God and others. In Buddhism, it might involve reaching enlightenment through the Four Noble Truths and the Eightfold Path.\n\n3. **Scientific and Psychological Perspectives**: From a biological standpoint, the meaning of life could be seen as survival and reproduction, ensuring the continuation of one's genetic lineage. Psychologists like Viktor Frankl, who survived Nazi concentration camps, suggest that finding meaning in life is crucial for psychological resilience and well-being. He proposed that meaning can be found in work, love, and suffering, emphasizing the importance of finding purpose even in adversity.\n\n4. **Cultural and Personal Perspectives**: Different cultures and individuals may find meaning in various aspects of life, such as family, community, creativity, personal achievement, or the pursuit of happiness. Some may find meaning in contributing to society, while others might find it in personal fulfillment or the simple act of living.\n\nUltimately, the meaning of life is a deeply personal inquiry that each individual must explore for themselves. It can evolve over time as one's experiences, beliefs, and values change. What is consistent across many perspectives is the importance of finding or creating a sense of purpose and fulfillment that resonates with one's own existence.",
            "tool_calls": []
          },
          "logprobs": null,
          "finish_reason": "stop",
          "stop_reason": null
        }
      ],
      "usage": {
        "prompt_tokens": 28,
        "total_tokens": 461,
        "completion_tokens": 433,
        "prompt_tokens_details": null
      },
      "prompt_logprobs": null
    }
    ```

8. You may also make use of the utility script `stream_chat.sh` to stream responses in real time
    ```bash
    ./stream_chat.sh "what is the meaning of life ?" "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    ```

9. To run benchmarks for inference, you can use the default benchmarking tool from vLLM like this
    ```bash
    kubectl exec -it deployment/$USER-serving-llama-4-serving -- python3 vllm/benchmarks/benchmark_serving.py --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --dataset-name random --ignore-eos --num-prompts 1100 --random-input-len 1000 --random-output-len 1000 --port 8000 --backend vllm --max-concurrency 64
    ```

    Once the benchmark is done, you can find the results in the logs similar to this:

    ```bash
    ============ Serving Benchmark Result ============
    Successful requests:                     1100
    Benchmark duration (s):                  xxx.xx
    Total input tokens:                      1100000
    Total generated tokens:                  1100000
    Request throughput (req/s):              x.xx
    Output token throughput (tok/s):         xxx.xx
    Total Token throughput (tok/s):          xxx.xx
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          xxx.xx
    Median TTFT (ms):                        xxx.xx
    P99 TTFT (ms):                           xxx.xx
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          xx.xx
    Median TPOT (ms):                        xx.xx
    P99 TPOT (ms):                           xx.xx
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           xx.xx
    Median ITL (ms):                         xx.xx
    P99 ITL (ms):                            xx.xx
    ==================================================
    ```

### Cleanup

To clean up the resources created by this recipe, complete the following steps:

1. Uninstall the helm chart.

    ```bash
    helm uninstall $USER-serving-llama-4
    ```

2. Delete the Kubernetes Secret.

    ```bash
    kubectl delete secret hf-secret
    ```