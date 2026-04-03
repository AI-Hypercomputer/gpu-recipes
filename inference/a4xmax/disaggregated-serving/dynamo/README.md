# Disaggregated Multi-Node Inference with NVIDIA Dynamo on A4X-MAX GKE

This document outlines the steps to deploy and serve Large Language Models (LLMs) using [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) disaggregated inference platform on [A4X-MAX (GB300) GKE Node pools](https://cloud.google.com/kubernetes-engine).

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
  * [2.6. Setup GCS Bucket for GKE](#setup-gcsfuse)
  * [2.7. Configure Dynamo Image ](#configure-dynamo-image)
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
    * GPU node pools with [a4x-maxgpu-4g-metal](https://docs.cloud.google.com/compute/docs/gpus#gb300-gpus) machines:
      * For 1p1d deployment: 2 machines with 4 GPUs each (8 GPUs total)
      * For 10p8d deployment: 18 machines with 4 GPUs each (72 GPUs total)
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
export RELEASE_VERSION=1.0.1
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

Install the Dynamo Platform with Grove & Kai scheduler enabled:
```bash
helm install https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace ${NAMESPACE} -f orchestrator-values.yaml --no-hooks --timeout 10m
```

Verify the installation:
```bash
kubectl get pods -n ${NAMESPACE}
```

Before moving forward, verify that all platform Pods have successfully transitioned to the `Running` state. The expected output should look similar to this, with every pod fully `Ready` and `Running`:
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

<a name="configure-dynamo-image"></a>
### 2.7. Configure Dynamo Image

Instead of building the Dynamo container image manually, we will use the official pre-built SGLang runtime release artifact.

For more details, refer to the [NVIDIA Release Artifacts documentation](https://docs.nvidia.com/dynamo/dev/resources/release-artifacts#sglang-runtime).
Set your environment to use the pre-built image:
```bash
export DYNAMO_IMAGE="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.1-cuda13"
```

<a name="deploy-sglang"></a>
## 3. Deploy with SGLang Backend

[Back to Top](#table-of-contents)

Deploy Dynamo with SGLang backend for high-performance inference. 

<a name="sglang-wo-deepep"></a>
### 3.1. SGLang Deployment without DeepEP (8 GPUs)

Two nodes deployment uses 8 GPUs across 2 A4X-MAX machines, targeting low latency. 

#### DeepSeekR1 671B Model

Deploy DeepSeekR1-671B across 2 nodes for testing and validation.  Note the use of `--set-file prefill_serving_config` and `--set-file decode_serving_config` pointing to the correct model config file.

```bash
cd $RECIPE_ROOT
helm install -f values_wo_deepep.yaml \
--set workload.image=${DYNAMO_IMAGE} \
--set volumes.gcsfuse.bucketName=${GCS_BUCKET} \
--set-file prefill_serving_config=$REPO_ROOT/src/frameworks/a4xmax/dynamo-configs/deepseekr1-fp8-1p1d-prefill.yaml \
--set-file decode_serving_config=$REPO_ROOT/src/frameworks/a4xmax/dynamo-configs/deepseekr1-fp8-1p1d-decode.yaml \
$USER-dynamo-a4xmax-1p1d \
$REPO_ROOT/src/helm-charts/a4xmax/inference-templates/dynamo-deployment
```

<a name="sglang-deepep"></a>
### 3.2. SGLang Deployment with DeepEP (72 GPUs)

Multi-node deployment uses 72 GPUs across 18 A4X-MAX machines, providing increased capacity for larger models or higher throughput. 

#### DeepSeekR1 671B Model

Deploy DeepSeekR1-671B across 18 nodes for production workloads. Note the use of `--set-file prefill_serving_config` and `--set-file decode_serving_config` pointing to the correct model config file for a multi node deployment scenario: 

```bash
cd $RECIPE_ROOT
helm install -f values_deepep.yaml \
--set workload.image=${DYNAMO_IMAGE} \
--set volumes.gcsfuse.bucketName=${GCS_BUCKET} \
--set-file prefill_serving_config=$REPO_ROOT/src/frameworks/a4xmax/dynamo-configs/deepseekr1-fp8-10p8d-prefill.yaml \
--set-file decode_serving_config=$REPO_ROOT/src/frameworks/a4xmax/dynamo-configs/deepseekr1-fp8-10p8d-decode.yaml \
$USER-dynamo-a4xmax-multi-node \
$REPO_ROOT/src/helm-charts/a4xmax/inference-templates/dynamo-deployment
```

<a name="inference-request"></a>
## 4. Inference Request
[Back to Top](#table-of-contents)

Before sending inference requests, verify that all pods have fully initialized and are in the `Running` state.

Run the following command to check the status of the pods in your namespace:

```bash
kubectl get pods -n ${NAMESPACE}
```

The expected output should look similar to the example below. Ensure that all platform components and `dynamo-disagg` worker pods (decode, prefill, and frontend) show a `Running` status and that all containers within them are `Ready` (e.g., `1/1` or `2/2`).

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

We can then deploy the benchmark client and send benchark request.
Deploy the benchmark client like this:
```bash
kubectl apply -f bench_client.yaml -n ${NAMESPACE}
```

And send the request like this: 

```bash
kubectl exec -it bench-client -- bash -c "cd /workspace/dynamo/examples/backends/sglang/slurm_jobs/scripts/vllm && python3 -u benchmark_serving.py     --host $USER-dynamo-a4xmax-1p1d-frontend   --port 8000     --model deepseek-ai/DeepSeek-R1     --tokenizer deepseek-ai/DeepSeek-R1     --backend 'dynamo'     --endpoint /v1/completions     --disable-tqdm     --dataset-name random     --num-prompts 2560     --random-input-len 8192     --random-output-len 1024     --random-range-ratio 0.8     --ignore-eos     --request-rate inf     --percentile-metrics ttft,tpot,itl,e2el     --max-concurrency 512"
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
helm uninstall $USER-dynamo-multi-node-serving -n ${NAMESPACE}
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