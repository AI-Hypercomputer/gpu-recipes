# Aggregated Inference Serving on GKE with TensorRT-LLM (A4X MAX / GB300)

This recipe provides the deployment configuration to run TensorRT-LLM aggregated inference on GKE with A4X MAX (GB300) Superpods. It leverages a Static SPMD architecture, NUMA optimizations, and Google Infrastructure Bundle (gIB) routing to maximize throughput using FP4 or FP8 engines.

<a name="table-of-contents"></a>
## Table of Contents

* [1. Test Environment](#test-environment)
* [2. Environment Setup (One-Time)](#environment-setup)
  * [2.1. Clone the Repository](#clone-repo)
  * [2.2. Configure Environment Variables](#configure-vars)
  * [2.3. Connect to your GKE Cluster](#connect-cluster)
  * [2.4. Create Secrets](#create-secrets)
  * [2.5. Setup GCS Bucket for GKE](#setup-gcsfuse)
  * [2.6. Configure TensorRT-LLM Image](#configure-trtllm-image)
* [3. Deploy with TensorRT-LLM Engine](#deploy-tensorrt-llm)
  * [3.1. Engine Architecture & Critical Setup](#system-architecture)
  * [3.2. TensorRT-LLM Deployment (8 GPUs across 2 Nodes)](#tensorrt-llm-deployment)
* [4. Inference Request & Benchmarking](#inference-request)
* [5. Monitoring and Troubleshooting](#monitoring)
* [6. Cleanup](#cleanup)

<a name="test-environment"></a>
## 1. Test Environment

[Back to Top](#table-of-contents)

This recipe has been tested with the following configuration:

* **GKE Cluster**:
    * GPU node pools with [a4x-maxgpu-4g-metal](https://docs.cloud.google.com/compute/docs/gpus#gb300-gpus) machines.
    * 2 machines with 4 GPUs each (8 GPUs total) for inference aggregation.
    * [Workload Identity Federation for GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/workload-identity) enabled.
    * [Cloud Storage FUSE CSI driver for GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/cloud-storage-fuse-csi-driver) enabled.

<a name="environment-setup"></a>
## 2. Environment Setup (One-Time)

[Back to Top](#table-of-contents)

<a name="clone-repo"></a>
### 2.1. Clone the Repository

```bash
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=$(pwd)
export RECIPE_ROOT=$REPO_ROOT/inference/a4xmax/aggregated-serving/tensorrt-llm
```

<a name="configure-vars"></a>
### 2.2. Configure Environment Variables

```bash
export PROJECT_ID=<PROJECT_ID>
export CLUSTER_REGION=<REGION_of_your_cluster>
export CLUSTER_NAME=<YOUR_GKE_CLUSTER_NAME>
export NAMESPACE=default
export NGC_API_KEY=<YOUR_NGC_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
export GCS_BUCKET=<YOUR_GCS_BUCKET>1

# Set the project for gcloud commands
gcloud config set project $PROJECT_ID
```

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

<a name="setup-gcsfuse"></a>
### 2.5. Setup GCS Bucket for GKE (One-Time Setup)

It is recommended to utilize [gcsfuse](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cloud-storage-fuse-csi-driver-setup) to facilitate model access.

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

<a name="configure-trtllm-image"></a>
### 2.6. Configure TensorRT-LLM Image

We will use the official NVIDIA TensorRT-LLM container artifact from the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release). Set your environment to use the pre-built image:

```bash
export TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc5"
```

<a name="deploy-tensorrt-llm"></a>
## 3. Deploy with TensorRT-LLM Engine

[Back to Top](#table-of-contents)

<a name="system-architecture"></a>
### 3.1. Engine Architecture & Critical Setup
The Helm Chart utilizes specialized optimizations to squeeze multi-node FP8 and FP4 capabilities on the GB300:
1. **Engine Architecture (Static SPMD):** To avoid MPI `Comm_spawn` segfaults in the containerized Kubernetes environment, the dynamic orchestrator must be disabled (`TRTLLM_ORCHESTRATOR=0`), and the engine must be launched explicitly with `mpirun` wrapping the FastAPI server.
2. **NUMA CPU & Memory Isolation:** To prevent the OS from scattering worker threads across the 144-core Grace CPU (which causes massive cross-socket UPI latency), strict NUMA isolation is enforced (`--bind-to numa`, `TLLM_NUMA_AWARE_WORKER_AFFINITY=1`).
3. **gIB Network Routing:** To route NCCL traffic over the GKE RoCEv2 ethernet fabric instead of defaulting to InfiniBand algorithms, the manifest installs GCP gIB plugins and sources environment variables (`set_nccl_env.sh`).
4. **Hardware Fabric Enablers:** To allow the 8 GPUs to communicate across the physical Multi-Node NVLink (MNNVL) switches across the nodes, memory flags must be explicitly exported (`MC_FORCE_MNNVL=1`, `NCCL_MNNVL_ENABLE=1`).
5. **Multi-Node IP Resolution:** TensorRT-LLM hardcodes `127.0.0.1` for its TCP tensor-parallel IPC listeners. In a multi-node Kubernetes cluster via LeaderWorkerSet, strings are dynamically patched at runtime to substitute the leader's actual network address.
6. **Triton JIT Compiler Paths:** The paths to the CUDA headers and PTX assembler are forwarded for Custom Attention and DeepGEMM compilations during the CUDA graph warmup phase.
7. **Warmup Synchronization:** The runtime logic blocks incoming requests (via a polling curl loop) until the engine is fully warmed up to prevent premature OOM crashes.

1. Review and populate the `values.yaml` file variables representing the model identity.

2. Apply the deployment using Helm:
```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
--set volumes.gcsfuse.bucketName=${GCS_BUCKET} \
--set workload.image=${TRTLLM_IMAGE} \
$USER-a4xmax-trtllm \
$REPO_ROOT/src/helm-charts/a4xmax/inference-templates/tensorrt-llm-deployment
```

3. View deployment logs to verify the TRT-LLM engine successfully loads:
```bash
kubectl logs -l app=a4xmax-trtllm -c tensorrt-llm -f
```

<a name="inference-request"></a>
## 4. Inference Request & Benchmarking

[Back to Top](#table-of-contents)

We provide a separated benchmarking pod definition to isolate client-side logic and load generation from the primary lifecycle of the backend nodes.

1. Deploy the load testing client:
```bash
cd $RECIPE_ROOT
kubectl apply -f bench_client.yaml
```

2. Exec into the running pod:
```bash
kubectl exec -it bench-client -- bash
```

3. Run the benchmark script inside the pod container against the headless service:
```bash
python3 -m tensorrt_llm.serve.scripts.benchmark_serving \
  --backend openai --host a4xmax-trtllm-0.a4xmax-trtllm.default.svc.cluster.local --port 8000 \
  --model /cache/<YOUR_MODEL_ID> \
  --num-prompts 9216 --trust-remote-code --ignore-eos --random-ids \
  --max-concurrency 3072 --random-input-len 1024 --random-output-len 8192 \
  --random-range-ratio 0.0 --dataset-name random --save-result \
  --result-filename /workspace/1k8k_benchmark_serving_results.json
```
**Note:**
This configuration is hard-coded to run only a single benchmarking experiment for 9216 requests for 1024/8192 token input/output lengths. To easily run other experiments, you can adjust the combinations provided in the `values.yaml` file.

<a name="monitoring"></a>
## 5. Monitoring and Troubleshooting

[Back to Top](#table-of-contents)

<a name="view-logs"></a>
### 5.1. View Logs

To see the logs from the TRTLLM server (useful for debugging and confirming server load), use the `-f` flag to follow the log stream:

```bash
kubectl logs -f deployment/a4xmax-trtllm -c tensorrt-llm
```

You should see logs indicating preparing the model and the multi-node engine status.

<a name="cleanup"></a>
## 6. Cleanup

[Back to Top](#table-of-contents)

Delete the deployments:
```bash
helm uninstall $USER-a4xmax-trtllm
kubectl delete -f $RECIPE_ROOT/bench_client.yaml
```
