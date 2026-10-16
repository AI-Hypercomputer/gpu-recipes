# Aggregated Multi-Node Inference with NVIDIA Dynamo on A4X-MAX GKE

This document outlines the steps to deploy and benchmark Large Language Models (LLMs) using [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) in aggregated mode on [A4X-MAX (GB300) GKE Node pools](https://cloud.google.com/kubernetes-engine).

In aggregated mode, every Dynamo worker runs both prefill and decode for its requests. The Dynamo frontend uses a KV-aware router to send each request to the worker that already holds the longest matching prefix in its KV cache. This recipe uses the [SGLang](https://docs.nvidia.com/dynamo/latest/components/backends/sglang/README.html) backend and profiles an agentic multi-turn workload with [`aiperf`](https://github.com/ai-dynamo/aiperf). For prefill/decode disaggregated serving, see the [disaggregated Dynamo recipe](../../disaggregated-serving/dynamo/README.md).

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
  * [3.2. High Interactivity Deployment](#sglang-high-interactivity)
* [4. Inference Request and Benchmarking](#inference-request)
  * [4.1. Verify Deployment Readiness](#verify-readiness)
  * [4.2. Deploy the Benchmark Client](#deploy-bench-client)
  * [4.3. Agentic Multi-Turn Profiling with AIPerf (AgentX Dataset)](#benchmark-aiperf-agentx)
* [5. Monitoring and Troubleshooting](#monitoring)
* [6. Cleanup](#cleanup)

<a name="test-environment"></a>
## 1. Test Environment

[Back to Top](#table-of-contents)

This recipe has been tested with the following configuration:

* **GKE Cluster**:
    * GPU node pools with [a4x-maxgpu-4g-metal](https://docs.cloud.google.com/compute/docs/gpus#gb300-gpus) machines:
      * 8 machines with 4 GPUs each (32 GPUs total), one Dynamo worker per machine
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
export RECIPE_ROOT=$REPO_ROOT/inference/a4xmax/aggregated-serving/dynamo
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

Skip this step if the Dynamo Platform is already installed in the cluster. The operator watches all namespaces, and its CRDs are cluster-scoped, so don't install a second, older platform release on a shared cluster. This recipe was reproduced on an existing `dynamo-platform` `1.4.2` install.

Add the NVIDIA Helm repository:
```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
  --username='$oauthtoken' --password=${NGC_API_KEY}
helm repo update
```

Install the Dynamo Platform with Grove & Kai scheduler enabled:
```bash
cd $RECIPE_ROOT
helm install https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace ${NAMESPACE} -f orchestrator-values.yaml --no-hooks --timeout 10m
```

Verify that all platform pods are `Running` and `Ready`:
```bash
kubectl get pods -n ${NAMESPACE}
```

<a name="setup-gcsfuse"></a>
### 2.6. Setup GCS Bucket for GKE (One-Time Setup)

It is recommended to utilize [gcsfuse](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cloud-storage-fuse-csi-driver-setup) to facilitate model access and mitigate [Hugging Face rate limiting](https://huggingface.co/docs/hub/en/rate-limits#hub-rate-limits) issues.

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

The bucket mounts at `/data/model` in every worker. Stage the target model ([`reflection/Beam-1.0-501B-A23B-NVFP4`](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4)) and the DFlash2 drafter ([`reflection/Beam-1.0-501B-A23B-DFlash2`](https://huggingface.co/reflection/Beam-1.0-501B-A23B-DFlash2)) into the bucket:

```bash
pip install -U "huggingface_hub[cli]"

# Target model: mounts at /data/model/<HF_MODEL_ID>
hf download reflection/Beam-1.0-501B-A23B-NVFP4 --local-dir ./reflection/Beam-1.0-501B-A23B-NVFP4

# DFlash2 drafter: mounts at the speculative-draft-model-path in beam_high_int_agg.yaml
hf download reflection/Beam-1.0-501B-A23B-DFlash2 --local-dir ./reflection/Beam-1.0-501B-A23B-DFlash2

gcloud storage cp -r ./reflection gs://${GCS_BUCKET}/
```

The workers load the target model from `/data/model/<HF_MODEL_ID>`. If you staged it at a different path in the bucket, pass `--set workload.modelPath=/data/model/<path>` to `helm install`.

<a name="configure-dynamo-image"></a>
### 2.7. Configure Dynamo Image

The runtime image comes from the Helm values file (`workload.image` and `dynamo.frontend.image`). To use a different image, add `--set workload.image=<IMAGE>` to `helm install`.

The chart sets `runtimeVersionOverride` to `dynamo.releaseVersion` (`1.0.1`) because the Dynamo operator can't infer the runtime version from a custom image tag.

<a name="deploy-sglang"></a>
## 3. Deploy with SGLang Backend

[Back to Top](#table-of-contents)

<a name="supported-models"></a>
### 3.1. Supported Models and Deployment Categories

This recipe reuses the Dynamo Helm chart at `$REPO_ROOT/src/helm-charts/a4xmax/inference-templates/dynamo-deployment`. Setting `dynamo.prefillWorker.replicas: 0` in the values file renders only unified workers (no `Prefill` service and no `subComponentType`). The worker serving configuration is passed through `decode_serving_config`.

| Model Name | Hugging Face ID / Path (`HF_MODEL_ID`) | Deployment Category | Topology & GPUs | Helm Values File (`VALUES_FILE`) | Serving Config (`SERVING_CONFIG`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Reflection Beam 501B (NVFP4)** | [`reflection/Beam-1.0-501B-A23B-NVFP4`](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4) | High Interactivity (DFlash2) | 8 aggregated workers (8 nodes / 32 GPUs) | `beam_high_int.yaml` | `beam_high_int_agg.yaml` |

<a name="sglang-high-interactivity"></a>
### 3.2. High Interactivity Deployment

This category deploys 8 aggregated SGLang workers (`TP=4`, `EP=4`, one per node) behind the Dynamo KV-aware router (`DYN_ROUTER_MODE=kv`, `DYN_KV_CACHE_BLOCK_SIZE=64`). Each worker uses DFlash2 speculative decoding ([`reflection/Beam-1.0-501B-A23B-DFlash2`](https://huggingface.co/reflection/Beam-1.0-501B-A23B-DFlash2)) with 8 draft tokens to lower Time-Per-Output-Token (TPOT).

#### Reflection Beam 501B (High Interactivity)

1. Set the model and configuration variables:

```bash
export HF_MODEL_ID="reflection/Beam-1.0-501B-A23B-NVFP4"
export VALUES_FILE="beam_high_int.yaml"
export SERVING_CONFIG="beam_high_int_agg.yaml"
export RELEASE_NAME="$USER-dynamo-a4xmax-beam-high-int"
```

2. Install the Helm release:

```bash
cd $RECIPE_ROOT
helm install -f ${VALUES_FILE} \
  --set workload.model=${HF_MODEL_ID} \
  --set volumes.gcsfuse.bucketName=${GCS_BUCKET} \
  --set dynamo.namespace=${NAMESPACE} \
  --set dynamo.nodeSelector."cloud\.google\.com/gke-nodepool"=${NODE_POOL} \
  --set-file decode_serving_config=$REPO_ROOT/src/frameworks/a4xmax/dynamo-configs/${SERVING_CONFIG} \
  ${RELEASE_NAME} \
  $REPO_ROOT/src/helm-charts/a4xmax/inference-templates/dynamo-deployment
```

<a name="inference-request"></a>
## 4. Inference Request and Benchmarking

[Back to Top](#table-of-contents)

<a name="verify-readiness"></a>
### 4.1. Verify Deployment Readiness

Before sending inference or benchmark requests, verify that the frontend pod and all 8 worker pods are `Running` and that all containers are `Ready`:

```bash
kubectl get pods -n ${NAMESPACE}
```

<a name="deploy-bench-client"></a>
### 4.2. Deploy the Benchmark Client

Deploy the standalone `bench-client` pod, which installs `aiperf` for AgentX multi-turn profiling:

```bash
cd $RECIPE_ROOT
kubectl apply -f bench_client.yaml -n ${NAMESPACE}
```

Wait for `bench-client` to finish setup:
```bash
kubectl logs -f pod/bench-client -n ${NAMESPACE}
```

Proceed once you see `--- SETUP COMPLETE. POD IS READY. ---`.

> [!NOTE]
> The client pod requests a `300Gi` memory limit and schedules on an `arm64` GPU node. A `48Gi` limit was OOM-killed at `CONCURRENCY=192` while loading the AgentX trace dataset.

<a name="benchmark-aiperf-agentx"></a>
### 4.3. Agentic Multi-Turn Profiling with AIPerf (AgentX Dataset)

Use [`aiperf`](https://github.com/ai-dynamo/aiperf) with the public SemiAnalysis AgentX coding trajectory dataset (`semianalysis_cc_traces_weka_062126`) to evaluate long-context, multi-turn prefix caching and decode interactivity.

1. Configure the target service, worker metrics endpoints, model, and concurrency (for example `CONCURRENCY=32`, `96`, or `192`):

```bash
export FRONTEND_HOST="$(kubectl get svc -n ${NAMESPACE} -o name | grep frontend | head -n1 | cut -d/ -f2)"
export WORKER_METRICS="$(kubectl get pods -n ${NAMESPACE} -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.podIP}{"\n"}{end}' | awk '/-decode-/ && $2 != "" {printf "%s:9090/metrics ", $2}')"
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

<a name="monitoring"></a>
## 5. Monitoring and Troubleshooting

[Back to Top](#table-of-contents)

Find the exact pod names:
```bash
kubectl get pods -n ${NAMESPACE}
```

Frontend logs:
```bash
kubectl logs -f <frontend-pod-name> -n ${NAMESPACE}
```

Worker logs:
```bash
kubectl logs -f <decode-worker-pod-name> -n ${NAMESPACE}
```

Common issues:

* **Pods stuck in Pending**: Check that 8 `a4x-maxgpu-4g-metal` nodes are free in the same NVL72 domain.
* **Worker fails to load the drafter**: Check that the drafter checkpoint is staged at the `speculative-draft-model-path` in `beam_high_int_agg.yaml`.
* **Slow first startup**: The first load reads the checkpoint through GCS FUSE into the local SSD file cache. Later restarts on the same node read from the cache.

<a name="cleanup"></a>
## 6. Cleanup

[Back to Top](#table-of-contents)

List deployed releases:
```bash
helm list -n ${NAMESPACE} --filter $USER-dynamo-
```

Uninstall the deployment and the benchmark client:
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
