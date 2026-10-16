# Disaggregated Multi-Node Inference with NVIDIA Dynamo on A4X-MAX GKE

This document outlines the steps to deploy and benchmark Large Language Models (LLMs) using the [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) disaggregated inference platform on [A4X-MAX (GB300) GKE Node pools](https://cloud.google.com/kubernetes-engine).

Dynamo provides a disaggregated architecture that separates prefill and decode operations for optimized inference performance, supporting multi-node configurations from 2 nodes (8 GPUs) up to full NVL72 domains (18 nodes / 72 GPUs). Dynamo supports multiple inference framework backends including [vLLM](https://docs.nvidia.com/dynamo/latest/components/backends/vllm/README.html) and [SGLang](https://docs.nvidia.com/dynamo/latest/components/backends/sglang/README.html). In this recipe, we focus on serving with the SGLang backend and profiling across both agentic multi-turn (`aiperf` / `agentx`) and synthetic workloads.

<a name="table-of-contents"></a>
## Table of Contents

* [1. Test Environment](#test-environment)
* [2. Environment Setup (One-Time)](#environment-setup)
  * [2.1. Clone the Repository](#clone-repo)
  * [2.2. Configure Environment Variables](#configure-vars)
  * [2.3. Connect to your GKE Cluster](#connect-cluster)
  * [2.4. Create Secrets](#create-secrets)
  * [2.5. Install Dynamo Platform](#install-platform)
  * [2.6. Setup GCS Bucket for GKE](#setup-gcsfuse)
  * [2.7. Configure Dynamo Image](#configure-dynamo-image)
* [3. Deploy with SGLang Backend](#deploy-sglang)
  * [3.1. Supported Models and Deployment Categories](#supported-models)
  * [3.2. Category 1: High Interactivity / Low Latency Deployment](#sglang-high-interactivity)
  * [3.3. Category 2: High Throughput Multi-Node Deployment](#sglang-high-throughput)
* [4. Inference Request and Benchmarking](#inference-request)
  * [4.1. Verify Deployment Readiness](#verify-readiness)
  * [4.2. Deploy the Benchmark Client](#deploy-bench-client)
  * [4.3. Category 1: Agentic Multi-Turn Profiling with AIPerf (AgentX Dataset)](#benchmark-aiperf-agentx)
  * [4.4. Category 2: Synthetic Random-Dataset Benchmarking](#benchmark-synthetic)
* [5. Monitoring and Troubleshooting](#monitoring)
* [6. Cleanup](#cleanup)

<a name="test-environment"></a>
## 1. Test Environment

[Back to Top](#table-of-contents)

This recipe has been tested with the following configuration:

* **GKE Cluster**:
    * GPU node pools with [a4x-maxgpu-4g-metal](https://docs.cloud.google.com/compute/docs/gpus#gb300-gpus) machines:
      * For 1P1D low-latency deployment: 2 machines with 4 GPUs each (8 GPUs total)
      * For multi-node high-throughput deployment (e.g., 4P4D / 3P5D on 32 GPUs, or 10P8D on 72 GPUs): 8 to 18 machines with 4 GPUs each (32 to 72 GPUs total)
    * [Workload Identity Federation for GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/workload-identity) enabled
    * [Cloud Storage FUSE CSI driver for GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/cloud-storage-fuse-csi-driver) enabled

<a name="environment-setup"></a>
## 2. Environment Setup (One-Time)

[Back to Top](#table-of-contents)

<a name="clone-repo"></a>
### 2.1. Clone the Repository

```bash
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=$(pwd)
export RECIPE_ROOT=$REPO_ROOT/inference/a4xmax/disaggregated-serving/dynamo
```

<a name="configure-vars"></a>
### 2.2. Configure Environment Variables

```bash
export PROJECT_ID=<PROJECT_ID>
export CLUSTER_REGION=<REGION_of_your_cluster>
export CLUSTER_NAME=<YOUR_GKE_CLUSTER_NAME>
export NAMESPACE=dynamo-cloud
export NGC_API_KEY=<YOUR_NGC_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
export RELEASE_VERSION=1.0.1
export GCS_BUCKET=<YOUR_GCS_BUCKET>
export NODE_POOL=<YOUR_GB300_NODE_POOL>

# Set the project for gcloud commands
gcloud config set project $PROJECT_ID
```

Replace the following values:

| Variable | Description | Example |
| -------- | ----------- | ------- |
| `PROJECT_ID` | Your Google Cloud Project ID | `gcp-project-12345` |
| `CLUSTER_REGION` | The GCP region where your GKE cluster is located | `us-central1` |
| `CLUSTER_NAME` | The name of your GKE cluster | `a4xmax-cluster` |
| `NGC_API_KEY` | Your NVIDIA NGC API key (get from [NGC](https://ngc.nvidia.com)) | `nvapi-xxx...` |
| `HF_TOKEN` | Your Hugging Face access token | `hf_xxx...` |
| `GCS_BUCKET` | Your GCS bucket name (without `gs://` prefix for Helm `bucketName`) | `my-model-bucket` |
| `NODE_POOL` | The GB300 node pool to run every worker in. All workers must share one NVLink domain | `a4xmax-np-1` |

The Helm chart deploys into `dynamo.namespace` from the values file (`dynamo-cloud`). The install commands below pass `--set dynamo.namespace=${NAMESPACE}` so the release follows `NAMESPACE`. Resource names (ComputeDomain, deployment, `bench-client`) are fixed, so use a separate namespace for each release that runs at the same time.

<a name="connect-cluster"></a>
### 2.3. Connect to your GKE Cluster

```bash
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

<a name="create-secrets"></a>
### 2.4. Create Secrets

Create the namespace:
```bash
kubectl create namespace ${NAMESPACE}
kubectl config set-context --current --namespace=$NAMESPACE
```

Create the Docker registry secret for NVIDIA Container Registry:
```bash
kubectl create secret docker-registry nvcr-secret \
  --namespace=${NAMESPACE} \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=${NGC_API_KEY}
```

Create the secret for the Hugging Face token:
```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

<a name="install-platform"></a>
### 2.5. Install Dynamo Platform (One-Time Setup)

Skip this step if the Dynamo Platform is already installed in the cluster. The operator watches all namespaces, and its CRDs are cluster-scoped, so don't install a second, older platform release on a shared cluster. The Reflection Beam recipe was reproduced on an existing `dynamo-platform` `1.4.2` install.

Add the NVIDIA Helm repository:
```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
  --username='$oauthtoken' --password=${NGC_API_KEY}
helm repo update
```

Install the Dynamo Platform with Grove & Kai scheduler enabled:
```bash
helm install https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace ${NAMESPACE} -f orchestrator-values.yaml --no-hooks --timeout 10m
```

Verify the installation:
```bash
kubectl get pods -n ${NAMESPACE}
```

Before moving forward, verify that all platform Pods have transitioned to the `Running` state. The expected output should look similar to this, with every pod `Ready` and `Running`:
```bash
$ kubectl get pods
NAME                                                              READY   STATUS    RESTARTS       AGE
admission-7486f477d9-59nd7                                        1/1     Running   0              117m
binder-87b67884d-jrf56                                            1/1     Running   0              117m
dynamo-platform-dynamo-operator-controller-manager-6b9576dcqfg2   1/1     Running   0              117m
dynamo-platform-nats-0                                            2/2     Running   0              110m
grove-operator-cf9498f87-s7zq5                                    1/1     Running   3 (110m ago)   117m
kai-operator-756446499c-ff9tj                                     1/1     Running   0              117m
kai-scheduler-default-84bc497d65-5mh9f                            1/1     Running   0              117m
pod-grouper-6c7db95fcf-zc9n9                                      1/1     Running   0              116m
podgroup-controller-f57775757-tjjk9                               1/1     Running   0              116m
queue-controller-5c8756b857-dqvh2                                 1/1     Running   0              117m
```

<a name="setup-gcsfuse"></a>
### 2.6. Setup GCS Bucket for GKE (One-Time Setup)

It is recommended to utilize [gcsfuse](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cloud-storage-fuse-csi-driver-setup) to facilitate model access and mitigate [Hugging Face rate limiting](https://huggingface.co/docs/hub/en/rate-limits#hub-rate-limits) issues.

Find the service account (usually annotated to default):
```bash
kubectl get serviceaccounts ${NAMESPACE} -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.annotations.iam\.gke\.io/gcp-service-account}{"\n"}{end}'
```

Configure the service account email:
```bash
export SERVICE_ACCOUNT_EMAIL=$(kubectl get serviceaccount/default -n ${NAMESPACE} -o jsonpath='{.metadata.annotations.iam\.gke\.io/gcp-service-account}')
```

Authorize the service account:
```bash
gcloud iam service-accounts add-iam-policy-binding ${SERVICE_ACCOUNT_EMAIL} \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:${PROJECT_ID}.svc.id.goog[${NAMESPACE}/default]"
```

Grant read access to the bucket:
```bash
gcloud storage buckets add-iam-policy-binding gs://${GCS_BUCKET} \
    --member "serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role "roles/storage.objectViewer"
```

Stage your model checkpoint files into the GCS bucket so they mount under `/data/model/<HF_MODEL_ID>` via GCS FUSE.

For example, to stage Reflection Beam ([`reflection/Beam-1.0-501B-A23B-NVFP4`](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4)):

```bash
pip install -U "huggingface_hub[cli]"
hf download reflection/Beam-1.0-501B-A23B-NVFP4 --local-dir ./reflection/Beam-1.0-501B-A23B-NVFP4
gcloud storage cp -r ./reflection gs://${GCS_BUCKET}/
```

If you stage a model at a different path in the bucket, pass `--set workload.modelPath=/data/model/<path>` to `helm install`.

<a name="configure-dynamo-image"></a>
### 2.7. Configure Dynamo Image

Instead of building the Dynamo container image manually, use the pre-built SGLang runtime release artifact (or a custom model-specific runtime image if required by your model).

For more details, refer to the [NVIDIA Release Artifacts documentation](https://docs.nvidia.com/dynamo/dev/resources/release-artifacts#sglang-runtime).
Set your environment to use the runtime image:
```bash
export DYNAMO_IMAGE="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.1-cuda13"
```

<a name="deploy-sglang"></a>
## 3. Deploy with SGLang Backend

[Back to Top](#table-of-contents)

<a name="supported-models"></a>
### 3.1. Supported Models and Deployment Categories

This recipe is organized into deployment categories that pair a Helm topology (`values_*.yaml`) with model-specific prefill and decode serving configurations under `$REPO_ROOT/src/frameworks/a4xmax/dynamo-configs/`. You can deploy any supported model by setting its `HF_MODEL_ID`, `VALUES_FILE`, `PREFILL_CONFIG`, and `DECODE_CONFIG`.

| Model Name | Hugging Face ID / Path (`HF_MODEL_ID`) | Deployment Category | Topology & GPUs | Helm Values File (`VALUES_FILE`) | Prefill Config (`PREFILL_CONFIG`) | Decode Config (`DECODE_CONFIG`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek-R1 671B (FP8)** | `deepseek-ai/DeepSeek-R1` | Low-Latency / without DeepEP | 1P1D (2 nodes / 8 GPUs) | `values_wo_deepep.yaml` | `deepseek-r1-fp8-1p1d-prefill.yaml` | `deepseekr1-fp8-1p1d-decode.yaml` |
| **DeepSeek-R1 671B (FP8)** | `deepseek-ai/DeepSeek-R1` | High-Throughput / with DeepEP | 10P8D (18 nodes / 72 GPUs) | `values_deepep.yaml` | `deepseekr1-fp8-10p8d-prefill.yaml` | `deepseekr1-fp8-10p8d-decode.yaml` |
| **Reflection Beam 501B (NVFP4)** | [`reflection/Beam-1.0-501B-A23B-NVFP4`](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4) | High Throughput | 4P4D (8 nodes / 32 GPUs) | `beam_high_throughput.yaml` | `beam_high_throughput_prefill.yaml` | `beam_high_throughput_decode.yaml` |

> [!NOTE]
> The Reflection Beam High Interactivity recipe runs 8 aggregated workers, so it lives in the [aggregated Dynamo recipe](../../aggregated-serving/dynamo/README.md).

> [!TIP]
> To add or deploy another model, stage its weights in `gs://${GCS_BUCKET}`, add its prefill and decode YAML files under `$REPO_ROOT/src/frameworks/a4xmax/dynamo-configs/`, and pass `--set workload.model=${HF_MODEL_ID}` along with `--set-file prefill_serving_config=...` and `--set-file decode_serving_config=...` to `helm install`.

<a name="sglang-high-interactivity"></a>
### 3.2. Category 1: High Interactivity / Low Latency Deployment

This category deploys A4X-MAX nodes targeting low Time-to-First-Token (TTFT) and low Time-Per-Output-Token (TPOT) for interactive workloads.

#### DeepSeek-R1 671B (FP8)

1. Set the model and configuration variables:

```bash
export HF_MODEL_ID="deepseek-ai/DeepSeek-R1"
export VALUES_FILE="values_wo_deepep.yaml"
export PREFILL_CONFIG="deepseek-r1-fp8-1p1d-prefill.yaml"
export DECODE_CONFIG="deepseekr1-fp8-1p1d-decode.yaml"
export RELEASE_NAME="$USER-dynamo-a4xmax-1p1d"
```

2. Install the Helm release:

```bash
cd $RECIPE_ROOT
helm install -f ${VALUES_FILE} \
  --set workload.model=${HF_MODEL_ID} \
  --set workload.image=${DYNAMO_IMAGE} \
  --set volumes.gcsfuse.bucketName=${GCS_BUCKET} \
  --set dynamo.namespace=${NAMESPACE} \
  --set-file prefill_serving_config=$REPO_ROOT/src/frameworks/a4xmax/dynamo-configs/${PREFILL_CONFIG} \
  --set-file decode_serving_config=$REPO_ROOT/src/frameworks/a4xmax/dynamo-configs/${DECODE_CONFIG} \
  ${RELEASE_NAME} \
  $REPO_ROOT/src/helm-charts/a4xmax/inference-templates/dynamo-deployment
```

<a name="sglang-high-throughput"></a>
### 3.3. Category 2: High Throughput Multi-Node Deployment

This category deploys multi-node A4X-MAX topologies across an NVL72 ComputeDomain to maximize total and output token throughput at high concurrency.

#### DeepSeek-R1 671B (FP8)

1. Set the model and configuration variables:

```bash
export HF_MODEL_ID="deepseek-ai/DeepSeek-R1"
export VALUES_FILE="values_deepep.yaml"
export PREFILL_CONFIG="deepseekr1-fp8-10p8d-prefill.yaml"
export DECODE_CONFIG="deepseekr1-fp8-10p8d-decode.yaml"
export RELEASE_NAME="$USER-dynamo-a4xmax-multi-node"
```

2. Install the Helm release:

```bash
cd $RECIPE_ROOT
helm install -f ${VALUES_FILE} \
  --set workload.model=${HF_MODEL_ID} \
  --set workload.image=${DYNAMO_IMAGE} \
  --set volumes.gcsfuse.bucketName=${GCS_BUCKET} \
  --set dynamo.namespace=${NAMESPACE} \
  --set-file prefill_serving_config=$REPO_ROOT/src/frameworks/a4xmax/dynamo-configs/${PREFILL_CONFIG} \
  --set-file decode_serving_config=$REPO_ROOT/src/frameworks/a4xmax/dynamo-configs/${DECODE_CONFIG} \
  ${RELEASE_NAME} \
  $REPO_ROOT/src/helm-charts/a4xmax/inference-templates/dynamo-deployment
```

#### Reflection Beam 501B (High Throughput)

1. Set the model and configuration variables:

```bash
export HF_MODEL_ID="reflection/Beam-1.0-501B-A23B-NVFP4"
export VALUES_FILE="beam_high_throughput.yaml"
export PREFILL_CONFIG="beam_high_throughput_prefill.yaml"
export DECODE_CONFIG="beam_high_throughput_decode.yaml"
export RELEASE_NAME="$USER-dynamo-a4xmax-beam-high-throughput"
```

2. Install the Helm release:

```bash
cd $RECIPE_ROOT
helm install -f ${VALUES_FILE} \
  --set workload.model=${HF_MODEL_ID} \
  --set volumes.gcsfuse.bucketName=${GCS_BUCKET} \
  --set dynamo.namespace=${NAMESPACE} \
  --set dynamo.nodeSelector."cloud\.google\.com/gke-nodepool"=${NODE_POOL} \
  --set-file prefill_serving_config=$REPO_ROOT/src/frameworks/a4xmax/dynamo-configs/${PREFILL_CONFIG} \
  --set-file decode_serving_config=$REPO_ROOT/src/frameworks/a4xmax/dynamo-configs/${DECODE_CONFIG} \
  ${RELEASE_NAME} \
  $REPO_ROOT/src/helm-charts/a4xmax/inference-templates/dynamo-deployment
```

<a name="inference-request"></a>
## 4. Inference Request and Benchmarking

[Back to Top](#table-of-contents)

<a name="verify-readiness"></a>
### 4.1. Verify Deployment Readiness

Before sending inference or benchmark requests, verify that all platform and worker pods have initialized and are in the `Running` state:

```bash
kubectl get pods -n ${NAMESPACE}
```

Ensure that all platform components and `dynamo-disagg` worker pods (decode, prefill, and frontend) show a `Running` status and that all containers within them are `Ready` (e.g., `1/1` or `2/2`):

```text
NAME                                                              READY   STATUS    RESTARTS       AGE
admission-7486f477d9-59nd7                                        1/1     Running   0              117m
binder-87b67884d-jrf56                                            1/1     Running   0              117m
dynamo-disagg10p8d-0-decode-0-decode-ldr-nlbhf                    2/2     Running   0              52m
...
dynamo-disagg10p8d-0-decode-0-decode-wkr-ln6hm                    2/2     Running   0              51m
dynamo-disagg10p8d-0-frontend-8gh64                               2/2     Running   0              38m
...
dynamo-disagg10p8d-0-frontend-svf2c                               2/2     Running   0              38m
dynamo-disagg10p8d-0-prefill-0-prefill-ldr-vgd95                  2/2     Running   0              62m
...
dynamo-disagg10p8d-0-prefill-4-prefill-wkr-bbnqr                  2/2     Running   0              61m
dynamo-platform-dynamo-operator-controller-manager-6b9576dcqfg2   1/1     Running   0              117m
dynamo-platform-nats-0                                            2/2     Running   0              110m
grove-operator-cf9498f87-s7zq5                                    1/1     Running   3 (110m ago)   117m
kai-operator-756446499c-ff9tj                                     1/1     Running   0              117m
kai-scheduler-default-84bc497d65-5mh9f                            1/1     Running   0              117m
pod-grouper-6c7db95fcf-zc9n9                                      1/1     Running   0              116m
podgroup-controller-f57775757-tjjk9                               1/1     Running   0              116m
queue-controller-5c8756b857-dqvh2                                 1/1     Running   0              117m
```

<a name="deploy-bench-client"></a>
### 4.2. Deploy the Benchmark Client

Deploy the standalone `bench-client` pod, which installs `aiperf` (for AgentX multi-turn profiling) alongside the Dynamo synthetic benchmarking tools:

```bash
cd $RECIPE_ROOT
kubectl apply -f bench_client.yaml -n ${NAMESPACE}
```

Wait for `bench-client` to finish setup:
```bash
kubectl logs -f pod/bench-client -n ${NAMESPACE}
```

Proceed once you see `--- SETUP COMPLETE. POD IS READY. ---`.

<a name="benchmark-aiperf-agentx"></a>
### 4.3. Category 1: Agentic Multi-Turn Profiling with AIPerf (AgentX Dataset)

Use [`aiperf`](https://github.com/ai-dynamo/aiperf) with the public SemiAnalysis AgentX coding trajectory dataset (`semianalysis_cc_traces_weka_062126`) to evaluate long-context, multi-turn prefix caching and decode interactivity.

1. Configure target service, worker metrics endpoints, model, and concurrency variables for the benchmark client (for example `CONCURRENCY=32`, `96`, or `192`):

```bash
# Replace FRONTEND_HOST with your Dynamo frontend service or pod IP
export FRONTEND_HOST="$(kubectl get svc -n ${NAMESPACE} -o name | grep frontend | head -n1 | cut -d/ -f2)"
export WORKER_METRICS="$(kubectl get pods -n ${NAMESPACE} -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.podIP}{"\n"}{end}' | awk '/-decode-|-prefill-/ && $2 != "" {printf "%s:9090/metrics ", $2}')"
export MODEL_NAME="${HF_MODEL_ID}"
export TOKENIZER_PATH="${HF_MODEL_ID}"
export CONCURRENCY=32
```

2. Run `aiperf` AgentX profiling from the `bench-client` pod:

```bash
kubectl exec -it bench-client -n ${NAMESPACE} -- bash -c "
  aiperf profile \
    --scenario inferencex-agentx-mvp \
    --url http://${FRONTEND_HOST}:8000 \
    --endpoint /v1/chat/completions \
    --model ${MODEL_NAME} \
    --tokenizer ${TOKENIZER_PATH} \
    --concurrency ${CONCURRENCY} \
    --benchmark-duration 3600 \
    --failed-request-threshold 0.10 \
    --warmup-requests-per-lane 10 \
    --trace-idle-gap-cap-seconds 300 \
    --warmup-grace-period 1800 \
    --tokenizer-trust-remote-code \
    --server-metrics ${WORKER_METRICS} \
    --output-artifact-dir /logs/agentic/aiperf_artifacts/stage_${CONCURRENCY} \
    --public-dataset semianalysis_cc_traces_weka_062126
"
```

<a name="benchmark-synthetic"></a>
### 4.4. Category 2: Synthetic Random-Dataset Benchmarking

For fixed-shape synthetic throughput and latency testing (e.g., `ISL=8192`, `OSL=1024`), use either the Dynamo `benchmark_serving.py` script in `bench-client` or `sglang.bench_serving`.

#### 4.4.1. Low-Latency Deployment (1P1D, 8 GPUs) via `bench-client`

```bash
kubectl exec -it bench-client -n ${NAMESPACE} -- bash -c "
  cd /workspace/dynamo/examples/backends/sglang/slurm_jobs/scripts/vllm && \
  python3 -u benchmark_serving.py \
    --host $USER-dynamo-a4xmax-1p1d-frontend \
    --port 8000 \
    --model ${HF_MODEL_ID} \
    --tokenizer ${HF_MODEL_ID} \
    --backend 'dynamo' \
    --endpoint /v1/completions \
    --disable-tqdm \
    --dataset-name random \
    --num-prompts 2560 \
    --random-input-len 8192 \
    --random-output-len 1024 \
    --random-range-ratio 0.8 \
    --ignore-eos \
    --request-rate inf \
    --percentile-metrics ttft,tpot,itl,e2el \
    --max-concurrency 512
"
```

#### 4.4.2. High-Throughput Multi-Node Deployment (10P8D, 72 GPUs) via `sglang.bench_serving`

```bash
kubectl exec -n ${NAMESPACE} $USER-dynamo-multi-node-serving-frontend -- \
  python3 -u -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --base-url http://localhost:8000 \
    --model "${HF_MODEL_ID}" \
    --tokenizer "/data/model/${HF_MODEL_ID}" \
    --dataset-name random \
    --num-prompts 10240 \
    --random-input-len 8192 \
    --random-range-ratio 0.8 \
    --random-output-len 1024 \
    --max-concurrency 2048
```

<a name="monitoring"></a>
## 5. Monitoring and Troubleshooting

[Back to Top](#table-of-contents)

View logs for different components (replace with your deployment name):

You can find the exact pod name by:
```bash
kubectl get pods -n ${NAMESPACE}
```

Frontend logs:
```bash
kubectl logs -f <frontend-pod-name> -c main -n ${NAMESPACE}
```

Decode worker logs:
```bash
kubectl logs -f <decode-worker-pod-name> -c main -n ${NAMESPACE}
```

Prefill worker logs:
```bash
kubectl logs -f <prefill-worker-pod-name> -c main -n ${NAMESPACE}
```

Common issues:

* **Pods stuck in Pending**: Check if nodes have sufficient resources (especially for multi-node deployments)
* **Model download slow**: Large models like DeepSeekR1 671B can take 30 minutes to download
* **Multi-node issues**: Verify network connectivity between nodes and proper subnet configuration
* **Deepep timeout issue**: Recompile DeepEP to patch NUM_CPU_TIMEOUT_SECS and NUM_TIMEOUT_CYCLES in csrc/kernels/configs.cuh during the image build.

<a name="cleanup"></a>
## 6. Cleanup

[Back to Top](#table-of-contents)

List deployed releases:
```bash
helm list -n ${NAMESPACE} --filter $USER-dynamo-
```

Uninstall specific deployments:
```bash
helm uninstall ${RELEASE_NAME} -n ${NAMESPACE}
kubectl delete -f $RECIPE_ROOT/bench_client.yaml -n ${NAMESPACE} --ignore-not-found=true
```

Uninstall Dynamo platform (if no longer needed):
```bash
helm uninstall dynamo-platform -n ${NAMESPACE}
```

Delete namespace and secrets:
```bash
kubectl delete namespace ${NAMESPACE}
```

Clean up downloaded charts:
```bash
rm -f dynamo-platform-${RELEASE_VERSION}.tgz
```
