# Disaggregated Multi-Node Inference with NVIDIA Dynamo on A4X GKE

This document outlines the steps to deploy and serve Large Language Models (LLMs) using [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) disaggregated inference platform on [A4X (GB200) GKE Node pools](https://cloud.google.com/kubernetes-engine).

Dynamo provides a disaggregated architecture that separates prefill and decode operations for optimized inference performance across multi-node NVL72 domains. Dynamo supports various inference framework backends like [vLLM](https://docs.nvidia.com/dynamo/latest/components/backends/vllm/README.html) and [SGLang](https://docs.nvidia.com/dynamo/latest/components/backends/sglang/README.html). In this recipe, we focus on serving `nvidia/DeepSeek-R1-0528-NVFP4-v2` (`ISL=8192`, `OSL=1024`) using the SGLang backend.

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
  * [3.1. SGLang Deployment without DeepEP (20 GPUs, 1P4D Low Latency)](#sglang-wo-deepep)
  * [3.2. SGLang Deployment with DeepEP (72 GPUs, 10P1D High Throughput)](#sglang-deepep)
* [4. Inference Request](#inference-request)
* [5. Monitoring and Troubleshooting](#monitoring)
* [6. Cleanup](#cleanup)

<a name="test-environment"></a>
## 1. Test Environment

[Back to Top](#table-of-contents)

This recipe has been tested with the following configuration:

* **GKE Cluster**:
    * GPU node pools with [a4x-highgpu-4g](https://docs.cloud.google.com/compute/docs/gpus#gb200-gpus) machines (4 GB200 GPUs per node):
      * For `1p4d` low-latency deployment (without DeepEP): 5 machines with 4 GPUs each (20 GPUs total: 1 Prefill worker × 4 GPUs + 4 Decode workers × 4 GPUs)
      * For `10p1d` high-throughput deployment (with DeepEP): 18 machines with 4 GPUs each (72 GPUs total: 10 Prefill workers × 4 GPUs + 1 Decode worker × 32 GPUs)
    * [Workload Identity Federation for GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/workload-identity) enabled
    * [Cloud Storage FUSE CSI driver for GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/cloud-storage-fuse-csi-driver) enabled

> [!IMPORTANT]
> To prepare the required environment, see the [GKE environment setup guide](../../../../docs/configuring-environment-gke-a4x.md).

<a name="environment-setup"></a>
## 2. Environment Setup (One-Time)

[Back to Top](#table-of-contents)

<a name="clone-repo"></a>
### 2.1. Clone the Repository

```bash
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=$(pwd)
export RECIPE_ROOT=$REPO_ROOT/inference/a4x/disaggregated-serving/dynamo
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
export RELEASE_VERSION=1.4.2
export GCS_BUCKET=<YOUR_GCS_BUCKET>

# Set the project for gcloud commands
gcloud config set project $PROJECT_ID
```

Replace the following values:

| Variable | Description | Example |
| -------- | ----------- | ------- |
| `PROJECT_ID` | Your Google Cloud Project ID | `gcp-project-12345` |
| `CLUSTER_REGION` | The GCP region where your GKE cluster is located | `us-central1` |
| `CLUSTER_NAME` | The name of your GKE cluster | `a4x-cluster` |
| `NGC_API_KEY` | Your NVIDIA NGC API key (get from [NGC](https://ngc.nvidia.com)) | `nvapi-xxx...` |
| `HF_TOKEN` | Your Hugging Face access token | `hf_xxx...` |
| `GCS_BUCKET` | Your GCS bucket name | `gs://xxx` |

<a name="connect-cluster"></a>
### 2.3. Connect to your GKE Cluster

```bash
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

<a name="create-secrets"></a>
### 2.4. Create Secrets

Create the namespace:
```bash
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
kubectl config set-context --current --namespace=$NAMESPACE
```

Create the Docker registry secret for NVIDIA Container Registry:
```bash
kubectl create secret docker-registry nvcr-secret \
  --namespace=${NAMESPACE} \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=${NGC_API_KEY} \
  --dry-run=client -o yaml | kubectl apply -f -
```

Create the secret for the Hugging Face token:
```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE} \
  --dry-run=client -o yaml | kubectl apply -f -
```

<a name="install-platform"></a>
### 2.5. Install Dynamo Platform (One-Time Setup)

> [!NOTE]
> If you are using a shared cluster where `dynamo-platform` is already installed in `dynamo-cloud`, verify it with `helm list -n ${NAMESPACE}` and `kubectl get pods -n ${NAMESPACE}`, and skip to Step 2.6.

Add the NVIDIA Helm repository:
```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
  --username='$oauthtoken' --password=${NGC_API_KEY}
helm repo update
```

Fetch the Dynamo Platform Helm chart:
```bash
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
```

Install the Dynamo Platform with Grove & KAI scheduler enabled:
```bash
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace ${NAMESPACE} \
  --set global.grove.install=true \
  --set global.kai-scheduler.install=true
```

Verify the installation:
```bash
kubectl get pods -n ${NAMESPACE}
```

Wait until all platform pods show a `Running` status before proceeding.

<a name="setup-gcsfuse"></a>
### 2.6. Setup GCS Bucket for GKE (One-Time Setup)

It is recommended to utilize [gcsfuse](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cloud-storage-fuse-csi-driver-setup) to facilitate model access and mitigate [Hugging Face rate limiting](https://huggingface.co/docs/hub/en/rate-limits#hub-rate-limits) issues.

Find the service account (usually annotated to default):
```bash
kubectl get serviceaccounts -n ${NAMESPACE} -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.annotations.iam\.gke\.io/gcp-service-account}{"\n"}{end}'
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
gcloud storage buckets add-iam-policy-binding ${GCS_BUCKET} \
    --member "serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role "roles/storage.objectViewer"
```

Label the `default` service account so the Dynamo operator grants it Kubernetes discovery RBAC permissions:
```bash
kubectl label serviceaccount default -n ${NAMESPACE} nvidia.com/dynamo-component-pod=true --overwrite
```

Download `nvidia/DeepSeek-R1-0528-NVFP4-v2` model files into your GCS bucket (under `nvidia/DeepSeek-R1-0528-NVFP4-v2`) and pass your GCS bucket name when installing the Helm release.

<a name="configure-dynamo-image"></a>
### 2.7. Configure Dynamo Image

Instead of building the Dynamo container image manually, use the official pre-built SGLang runtime image.

For more details, refer to the [NVIDIA Release Artifacts documentation](https://docs.nvidia.com/dynamo/dev/resources/release-artifacts#sglang-runtime).
Set your environment to use the pre-built image:
```bash
export DYNAMO_IMAGE="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0-dev.1-cuda13"
```

<a name="deploy-sglang"></a>
## 3. Deploy with SGLang Backend

[Back to Top](#table-of-contents)

Deploy Dynamo with the SGLang backend for high-performance inference using Kubernetes service discovery (`DYN_DISCOVERY_BACKEND=kubernetes`, `DYN_EVENT_PLANE=zmq`, `DYN_REQUEST_PLANE=tcp`) and Mooncake MNNVL/UCX KV cache transfer.

<a name="sglang-wo-deepep"></a>
### 3.1. SGLang Deployment without DeepEP (20 GPUs, 1P4D Low Latency)

The 5-node deployment (`1p4d`) uses 20 GPUs across 5 A4X machines (1 Prefill worker × 4 GPUs TP=4 EP=1 + 4 Decode workers × 4 GPUs TP=4 EP=1 without DeepEP), targeting low latency (`ISL=8192`, `OSL=1024`, `concurrency=4`).

#### DeepSeek-R1 NVFP4 Model (`1p4d`, 20 GPUs)

Deploy `nvidia/DeepSeek-R1-0528-NVFP4-v2` across 5 nodes:

```bash
cd $RECIPE_ROOT
helm install $USER-dynamo-a4x-1p4d \
  $REPO_ROOT/src/helm-charts/a4x/inference-templates/dynamo-deployment \
  -n ${NAMESPACE} -f values_wo_deepep.yaml \
  --set workload.image=${DYNAMO_IMAGE} \
  --set volumes.gcsfuse.bucketName=${GCS_BUCKET} \
  --set-file prefill_serving_config=$REPO_ROOT/src/frameworks/a4x/dynamo-configs/deepseekr1-fp4-1p4d-prefill.yaml \
  --set-file decode_serving_config=$REPO_ROOT/src/frameworks/a4x/dynamo-configs/deepseekr1-fp4-1p4d-decode.yaml
```

<a name="sglang-deepep"></a>
### 3.2. SGLang Deployment with DeepEP (72 GPUs, 10P1D High Throughput)

The 18-node deployment (`10p1d`) uses a full 72-GPU NVL72 rack across 18 A4X machines (10 Prefill workers × 4 GPUs TP=4 EP=1 + 1 Decode worker × 32 GPUs across 8 nodes TP=32 DP=32 EP=32 with DeepEP `low_latency`) for high throughput (`ISL=8192`, `OSL=1024`, `concurrency=2048`).

#### DeepSeek-R1 NVFP4 Model (`10p1d`, 72 GPUs)

```bash
cd $RECIPE_ROOT
helm install $USER-dynamo-a4x-10p1d \
  $REPO_ROOT/src/helm-charts/a4x/inference-templates/dynamo-deployment \
  -n ${NAMESPACE} -f values_deepep.yaml \
  --set workload.image=${DYNAMO_IMAGE} \
  --set volumes.gcsfuse.bucketName=${GCS_BUCKET} \
  --set-file prefill_serving_config=$REPO_ROOT/src/frameworks/a4x/dynamo-configs/deepseekr1-fp4-10p1d-prefill.yaml \
  --set-file decode_serving_config=$REPO_ROOT/src/frameworks/a4x/dynamo-configs/deepseekr1-fp4-10p1d-decode.yaml
```

<a name="inference-request"></a>
## 4. Inference Request

[Back to Top](#table-of-contents)

Check if the pods are in `Running` status before sending inference requests:

```bash
kubectl get pods -n ${NAMESPACE}
```

Deploy the benchmark client and send benchmark requests:
```bash
kubectl apply -f bench_client.yaml -n ${NAMESPACE}
```

Send the benchmark request via `bench-client` (for `1p4d` low latency, or set `--host $USER-dynamo-a4x-10p1d-frontend` and `--max-concurrency 2048` for `10p1d`):

```bash
kubectl exec -it bench-client -n ${NAMESPACE} -- bash -c "cd /workspace/dynamo/examples/backends/sglang/slurm_jobs/scripts/vllm && python3 -u benchmark_serving.py \
  --host $USER-dynamo-a4x-1p4d-frontend \
  --port 8000 \
  --model nvidia/DeepSeek-R1-0528-NVFP4-v2 \
  --tokenizer nvidia/DeepSeek-R1-0528-NVFP4-v2 \
  --backend 'dynamo' \
  --endpoint /v1/completions \
  --disable-tqdm \
  --dataset-name random \
  --num-prompts 40 \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --random-range-ratio 0 \
  --ignore-eos \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl,e2el \
  --max-concurrency 4"
```

Or send a benchmark request directly from a frontend pod:

```bash
FRONTEND_POD=$(kubectl get pods -n ${NAMESPACE} -l nvidia.com/dynamo-component-type=frontend -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${NAMESPACE} ${FRONTEND_POD} -- python3 -u -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --base-url http://localhost:8000 \
  --model "nvidia/DeepSeek-R1-0528-NVFP4-v2" \
  --tokenizer /data/model/nvidia/DeepSeek-R1-0528-NVFP4-v2 \
  --dataset-name random \
  --num-prompts 20480 \
  --random-input-len 8192 \
  --random-range-ratio 0 \
  --random-output-len 1024 \
  --max-concurrency 2048
```

<a name="monitoring"></a>
## 5. Monitoring and Troubleshooting

[Back to Top](#table-of-contents)

View logs for different components:

Find the pod names:
```bash
kubectl get pods -n ${NAMESPACE}
```

Frontend logs:
```bash
kubectl logs -f -l nvidia.com/dynamo-component-type=frontend -n ${NAMESPACE}
```

Decode worker logs:
```bash
kubectl logs -f -l nvidia.com/dynamo-sub-component-type=decode -n ${NAMESPACE}
```

Prefill worker logs:
```bash
kubectl logs -f -l nvidia.com/dynamo-sub-component-type=prefill -n ${NAMESPACE}
```

Common issues:

* **Pods stuck in Pending**: Check if nodes have sufficient resources (5 nodes for `1p4d`, 18 nodes for `10p1d`) and reside in the same NVL72 node pool. You can pin a specific node pool via `--set nodeSelector."cloud\.google\.com/gke-nodepool"=<NODEPOOL_NAME>`.
* **Model download slow**: Ensure your GCS bucket is in the same region as the GKE cluster and gcsfuse file caching is enabled.
* **Multi-node issues**: Verify network connectivity between nodes and proper RDMA/MNNVL ComputeDomain configuration.

<a name="cleanup"></a>
## 6. Cleanup

[Back to Top](#table-of-contents)

List deployed releases:
```bash
helm list -n ${NAMESPACE} --filter $USER-dynamo-
```

Uninstall specific deployments:
```bash
helm uninstall $USER-dynamo-a4x-1p4d -n ${NAMESPACE}
helm uninstall $USER-dynamo-a4x-10p1d -n ${NAMESPACE}
kubectl delete -f bench_client.yaml -n ${NAMESPACE} --ignore-not-found
```

Uninstall Dynamo platform (if no longer needed):
```bash
helm uninstall dynamo-platform -n ${NAMESPACE}
```

Delete namespace and secrets:
```bash
kubectl delete namespace ${NAMESPACE}
```
