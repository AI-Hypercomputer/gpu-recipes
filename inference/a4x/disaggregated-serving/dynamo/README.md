# Disaggregated Multi-Node Inference with NVIDIA Dynamo on A4X GKE

This document outlines the steps to deploy and serve Large Language Models (LLMs) using [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) disaggregated inference platform on [A4X GKE Node pools](https://cloud.google.com/kubernetes-engine).

Dynamo provides a disaggregated architecture that separates prefill and decode operations for optimized inference performance, supporting both single-node (4 GPUs) and multi-node NVL72 (72 GPUs) configurations. Dynamo also supports various inference framework backends like [vLLM](https://docs.nvidia.com/dynamo/latest/components/backends/vllm/README.html) and [SGLang](https://docs.nvidia.com/dynamo/latest/components/backends/sglang/README.html). In this recipe, we will focus on serving using the SGLang backend. 

<a name="table-of-contents"></a>
## Table of Contents

* [1. Test Environment](#test-environment)
* [2. Environment Setup (One-Time)](#environment-setup)
  * [2.1. Clone the Repository](#clone-repo)
  * [2.2. Configure Environment Variables](#configure-vars)
  * [2.3. Connect to your GKE Cluster](#connect-cluster)
  * [2.4. Create Secrets](#create-secrets)
  * [2.5. Install Dynamo Platform](#install-platform)
  * [2.6. Setup GCS Bucket for GKE ](#setup-gcsfuse)
* [3. Deploy with SGLang Backend](#deploy-sglang)
  * [3.1. SGLang Deployment without DeepEP(8 GPUs)](#sglang-wo-deepep)
  * [3.2. SGLang Deployment with DeepEP(72 GPUs)](#sglang-deepep)
* [4. Inference Request](#inference-request)
* [5. Monitoring and Troubleshooting](#monitoring)
* [6. Cleanup](#cleanup)

<a name="test-environment"></a>
## 1. Test Environment

[Back to Top](#table-of-contents)

This recipe has been tested with the following configuration:

* **GKE Cluster**:
    * GPU node pools with [a4x-highgpu-4g](https://docs.cloud.google.com/compute/docs/gpus#gb200-gpus) machines:
      * For multi-node deployment: 4 machines with 4 GPUs each (16 GPUs total)
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
export RELEASE_VERSION=0.7.0
export GCS_BUCKET=<YOUR_CGS_BUCKET>

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

Add the NVIDIA Helm repository:
```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
  --username='$oauthtoken' --password=${NGC_API_KEY}
helm repo update
```

Fetch the Dynamo Helm charts:
```bash
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
```

Install the Dynamo CRDs:
```bash
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz \
  --namespace default \
  --wait \
  --atomic
```

Install the Dynamo Platform with Grove & Kai scheduler enabled:
```bash
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace ${NAMESPACE} --set grove.enabled=true --set kai-scheduler.enabled=true
```

Verify the installation:
```bash
kubectl get pods -n ${NAMESPACE}
```

Wait until all pods show a `Running` status before proceeding.

<a name="setup-gcsfuse"></a>
### 2.6. Setup GCS Bucket for GKE (One-Time Setup)

It is recommended to utilize [gcsfuse](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cloud-storage-fuse-csi-driver-setup) to facilitate model access and mitigate [huggingface rate limiting](https://huggingface.co/docs/hub/en/rate-limits#hub-rate-limits) issues.

Find the service account (usually annotated to default):
```bash
kubectl get serviceaccounts ${NAMESPACE} -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.annotations.iam\.gke\.io/gcp-service-account}{"\n"}{end}'
```

Config the service account email:
```bash
export SERVICE_ACCOUNT_EMAIL=$(kubectl get serviceaccount/default -n ${NAMESPACE} -o jsonpath='{.metadata.annotations.iam\.gke\.io/gcp-service-account}')
```

Authorize the service account:
```bash
gcloud iam service-accounts add-iam-policy-binding ${SERVICE_ACCOUNT_EMAIL} \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:project_id.svc.id.goog[${NAMESPACE}/default]"
```

Grant read access to the bucket:
```bash
gcloud storage buckets add-iam-policy-binding ${GCS_BUCKET} \
    --member "serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role "roles/storage.objectViewer"
```

Downloading model files into the gcs bucket and set your gcs bucket name in values.yaml file.

<a name="deploy-sglang"></a>
## 3. Deploy with SGLang Backend

[Back to Top](#table-of-contents)

Deploy Dynamo with SGLang backend for high-performance inference. 

<a name="sglang-wo-deepep"></a>
### 3.1. SGLang Deployment without DeepEP (8 GPUs)

Two nodes deployment uses 8 GPUs across 2 A4X machines, targeting low latency. 

#### DeepSeekR1 671B Model

Deploy DeepSeekR1-671B across 2 nodes for testing and validation.  Note the use of `--set-file prefill_serving_config` and `--set-file decode_serving_config` pointing to the correct model config file.

```bash
cd $RECIPE_ROOT
helm install -f values_wo_deepep.yaml \
--set-file prefill_serving_config=$REPO_ROOT/src/frameworks/a4x/dynamo-configs/deepseekr1-fp8-1p1d-prefill.yaml \
--set-file decode_serving_config=$REPO_ROOT/src/frameworks/a4x/dynamo-configs/deepseekr1-fp8-1p1d-decode.yaml \
$USER-dynamo-a4x-1p1d \
$REPO_ROOT/src/helm-charts/a4x/inference-templates/dynamo-deployment
```

<a name="sglang-deepep"></a>
### 3.2. SGLang Deployment with DeepEP (72 GPUs)

Multi-node deployment uses 72 GPUs across 18 A4X machines, providing increased capacity for larger models or higher throughput. 

#### DeepSeekR1 671B Model

Deploy DeepSeekR1-671B across 18 nodes for production workloads. Note the use of `--set-file prefill_serving_config` and `--set-file decode_serving_config` pointing to the correct model config file for a multi node deployment scenario: 

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
--set-file prefill_serving_config=$REPO_ROOT/src/frameworks/a4x/dynamo-configs/deepseekr1-fp8-10p8d-prefill.yaml \
--set-file decode_serving_config=$REPO_ROOT/src/frameworks/a4x/dynamo-configs/deepseekr1-fp8-10p8d-decode.yaml \
$USER-dynamo-a4x-multi-node \
$REPO_ROOT/src/helm-charts/a4x/inference-templates/dynamo-deployment
```

<a name="inference-request"></a>
## 4. Inference Request
[Back to Top](#table-of-contents)

Check if the pods are in `Running` status before sending inference requests. 

```bash
kubectl get pods -n ${NAMESPACE}
```

We can then deploy the benchmark client and send benchark request.
Deploy the benchmark client like this:
```bash
kubectl apply -f bench_client.yaml -n ${NAMESPACE}
```

And send the request like this: 

```bash
kubectl exec -it bench-client -- bash -c "cd /workspace/dynamo/examples/backends/sglang/slurm_jobs/scripts/vllm && python3 -u benchmark_serving.py     --host $USER-dynamo-a4x-1p1d-frontend   --port 8000     --model deepseek-ai/DeepSeek-R1     --tokenizer deepseek-ai/DeepSeek-R1     --backend 'dynamo'     --endpoint /v1/completions     --disable-tqdm     --dataset-name random     --num-prompts 2560     --random-input-len 8192     --random-output-len 1024     --random-range-ratio 0.8     --ignore-eos     --request-rate inf     --percentile-metrics ttft,tpot,itl,e2el     --max-concurrency 512"
```

Or we can send a benchmark request to a frontend pod like this:

```bash
kubectl exec -n ${NAMESPACE} $USER-dynamo-multi-node-serving-frontend -- python3 -u -m sglang.bench_serving    --backend sglang-oai-chat    --base-url http://localhost:8000    --model "deepseek-ai/DeepSeek-R1"    --tokenizer /data/model/deepseek-ai/DeepSeek-R1    --dataset-name random    --num-prompts 10240   --random-input-len 8192  --random-range-ratio 0.8   --random-output-len 1024   --max-concurrency 2048
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
kubectl logs -f deployment/$USER-dynamo-multi-node-serving-frontend -n ${NAMESPACE}
```

Decode worker logs:
```bash
kubectl logs -f deployment/$USER-dynamo-multi-node-serving-decode-worker -n ${NAMESPACE}
```

Prefill worker logs:
```bash
kubectl logs -f deployment/$USER-dynamo-multi-node-serving-prefill-worker -n ${NAMESPACE}
```

Common issues:

* **Pods stuck in Pending**: Check if nodes have sufficient resources (especially for multi-node deployments)
* **Model download slow**: Large models like DeepSeekR1 671B can take 30 minutes to download
* **Multi-node issues**: Verify network connectivity between nodes and proper subnet configuration

<a name="cleanup"></a>
## 6. Cleanup

[Back to Top](#table-of-contents)

List deployed releases:
```bash
helm list -n ${NAMESPACE} --filter $USER-dynamo-
```

Uninstall specific deployments:
```bash
helm uninstall $USER-dynamo-multi-node-serving -n ${NAMESPACE}
```

Uninstall Dynamo platform (if no longer needed):
```bash
helm uninstall dynamo-platform -n ${NAMESPACE}
helm uninstall dynamo-crds -n default
```

Delete namespace and secrets:
```bash
kubectl delete namespace ${NAMESPACE}
```

Clean up downloaded charts:
```bash
rm -f dynamo-crds-${RELEASE_VERSION}.tgz
rm -f dynamo-platform-${RELEASE_VERSION}.tgz
```

