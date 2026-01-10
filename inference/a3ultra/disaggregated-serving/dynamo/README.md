# Disaggregated Inference with NVIDIA Dynamo on A3 Ultra GKE

This document outlines the steps to deploy and serve Large Language Models (LLMs) using [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) disaggregated inference platform on [A3 Ultra GKE Node pools](https://cloud.google.com/kubernetes-engine).

Dynamo provides a disaggregated architecture that separates prefill and decode operations for optimized inference performance, supporting both single-node (8 GPUs) and multi-node (16 GPUs) configurations. Dynamo also supports various inference framework backends like [vLLM](https://docs.nvidia.com/dynamo/latest/backends/vllm/README.html) and [SGLang](https://docs.nvidia.com/dynamo/latest/backends/sglang/README.html). In this recipe, we will focus on serving using the vLLM backend. 

<a name="table-of-contents"></a>
## Table of Contents

* [1. Test Environment](#test-environment)
* [2. Environment Setup (One-Time)](#environment-setup)
  * [2.1. Clone the Repository](#clone-repo)
  * [2.2. Configure Environment Variables](#configure-vars)
  * [2.3. Connect to your GKE Cluster](#connect-cluster)
  * [2.4. Create Secrets](#create-secrets)
  * [2.5. Install Dynamo Platform](#install-platform)
* [3. Deploy with vLLM Backend](#deploy-vllm)
  * [3.1. Single-Node vLLM Deployment (8 GPUs)](#vllm-single-node)
  * [3.2. Multi-Node vLLM Deployment (16 GPUs)](#vllm-multi-node)
* [4. Inference Request](#inference-request)
* [5. Advanced Configuration](#advanced-config)
  * [5.1. Custom Network Configuration](#custom-networks)
* [6. Monitoring and Troubleshooting](#monitoring)
* [7. Cleanup](#cleanup)

<a name="test-environment"></a>
## 1. Test Environment

[Back to Top](#table-of-contents)

This recipe has been optimized for and tested with the following configuration:

* **GKE Cluster**:
    * A [regional standard cluster](https://cloud.google.com/kubernetes-engine/docs/concepts/configuration-overview) version: `1.31.7-gke.1265000` or later
    * GPU node pools with [a3-ultragpu-8g](https://cloud.google.com/compute/docs/gpus#h200-gpus) machines:
      * For single-node deployment: 1 machine with 8 GPUs
      * For multi-node deployment: 2 machines with 8 GPUs each (16 GPUs total)
    * [Workload Identity Federation for GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/workload-identity) enabled
    * [Cloud Storage FUSE CSI driver for GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/cloud-storage-fuse-csi-driver) enabled
    * [DCGM metrics](https://cloud.google.com/kubernetes-engine/docs/how-to/dcgm-metrics) enabled

> [!IMPORTANT]
> To prepare the required environment, see the [GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-ultra.md).

<a name="environment-setup"></a>
## 2. Environment Setup (One-Time)

[Back to Top](#table-of-contents)

<a name="clone-repo"></a>
### 2.1. Clone the Repository

```bash
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=$(pwd)
export RECIPE_ROOT=$REPO_ROOT/inference/a3ultra/disaggregated-serving/dynamo
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
export RELEASE_VERSION=0.4.1

# Set the project for gcloud commands
gcloud config set project $PROJECT_ID
```

Replace the following values:

| Variable | Description | Example |
| -------- | ----------- | ------- |
| `PROJECT_ID` | Your Google Cloud Project ID | `gcp-project-12345` |
| `CLUSTER_REGION` | The GCP region where your GKE cluster is located | `us-central1` |
| `CLUSTER_NAME` | The name of your GKE cluster | `a3-ultra-cluster` |
| `NGC_API_KEY` | Your NVIDIA NGC API key (get from [NGC](https://ngc.nvidia.com)) | `nvapi-xxx...` |
| `HF_TOKEN` | Your Hugging Face access token | `hf_xxx...` |

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

Install the Dynamo Platform:
```bash
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace ${NAMESPACE}
```

Verify the installation:
```bash
kubectl get pods -n ${NAMESPACE}
```

Wait until all pods show a `Running` status before proceeding.

<a name="deploy-vllm"></a>
## 3. Deploy with vLLM Backend

[Back to Top](#table-of-contents)

Deploy Dynamo with vLLM backend for high-performance inference. Choose between single-node (8 GPUs) or multi-node (16 GPUs) configurations based on your requirements.

<a name="vllm-single-node"></a>
### 3.1. Single-Node vLLM Deployment (8 GPUs)

Single-node deployment uses 8 GPUs on one A3 Ultra machine, suitable for smaller models and workloads.

#### Llama 3.3 70B Model

Deploy Llama 3.3 70B Instruct model for testing and validation. 

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
  --set-file workload_launcher=$REPO_ROOT/src/launchers/dynamo-vllm-launcher.sh \
  --set-file serving_config=$REPO_ROOT/src/frameworks/a3ultra/dynamo-configs/llama-3.3-70b-single-node.yaml \
  --set workload.framework=vllm \
  --set workload.model.name=meta-llama/Llama-3.3-70B-Instruct \
  --set workload.image=nvcr.io/nvidia/ai-dynamo/vllm-runtime:${RELEASE_VERSION} \
  --set workload.gpus=8 \
  $USER-dynamo-single-node-serving \
  $REPO_ROOT/src/helm-charts/a3ultra/inference-templates/dynamo-deployment
```

<a name="vllm-multi-node"></a>
### 3.2. Multi-Node vLLM Deployment (16 GPUs)

Multi-node deployment uses 16 GPUs across two A3 Ultra machines, providing increased capacity for larger models or higher throughput.

#### Llama 3.3 70B Model

Deploy Llama 3.3 70B Instruct across multiple nodes for production workloads. Note the use of `--set-file serving_config` pointing to `llama-3.3-70b-multi-node.yaml` and `--set workload.gpus=16` for a multi node deployment scenario: 

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
  --set-file workload_launcher=$REPO_ROOT/src/launchers/dynamo-vllm-launcher.sh \
  --set-file serving_config=$REPO_ROOT/src/frameworks/a3ultra/dynamo-configs/llama-3.3-70b-multi-node.yaml \
  --set workload.framework=vllm \
  --set workload.model.name=meta-llama/Llama-3.3-70B-Instruct \
  --set workload.image=nvcr.io/nvidia/ai-dynamo/vllm-runtime:${RELEASE_VERSION} \
  --set workload.gpus=16 \
  $USER-dynamo-multi-node-serving \
  $REPO_ROOT/src/helm-charts/a3ultra/inference-templates/dynamo-deployment
```

<a name="inference-request"></a>
## 4. Inference Request
[Back to Top](#table-of-contents)

To make an inference request to test the server, we can first run a health check against the server using `curl`

```bash
kubectl exec -it -n ${NAMESPACE} deployment/$USER-dynamo-single-node-serving-frontend -- curl http://localhost:8000/health | jq
```

You should see a server status like this. Wait for it to be in a `healthy` state.

```json
{
  "instances": [
    {
      "component": "backend",
      "endpoint": "load_metrics",
      "instance_id": 3994861215823793160,
      "namespace": "dynamo",
      "transport": {
        "nats_tcp": "dynamo_backend.load_metrics-3770991c30298c08"
      }
    },
    {
      "component": "prefill",
      "endpoint": "clear_kv_blocks",
      "instance_id": 3994861215823793153,
      "namespace": "dynamo",
      "transport": {
        "nats_tcp": "dynamo_prefill.clear_kv_blocks-3770991c30298c01"
      }
    },
    {
      "component": "prefill",
      "endpoint": "generate",
      "instance_id": 3994861215823793153,
      "namespace": "dynamo",
      "transport": {
        "nats_tcp": "dynamo_prefill.generate-3770991c30298c01"
      }
    }
  ],
  "message": "No endpoints available",
  "status": "unhealthy"
}
``` 


Then we can send a request with some sample data like this for a single node scenario:

```bash
kubectl exec -it -n ${NAMESPACE} deployment/$USER-dynamo-single-node-serving-frontend  -- \
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "messages": [
    {
        "role": "user",
        "content": "what is the meaning of life ?"
    }
    ],
    "stream":false,
    "max_tokens": 30
  }' | jq
```

For a multi node scenrio, replace the deployment name with `$USER-dynamo-multi-node-serving-frontend` to send the requests.

<a name="advanced-config"></a>
## 5. Advanced Configuration

[Back to Top](#table-of-contents)

<a name="custom-networks"></a>
### 5.1. Custom Network Configuration

If you created your cluster using the [GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-ultra.md), it's configured with default settings that include the names for networks and subnetworks used for communication between:

- The host to external services
- GPU-to-GPU communication

For clusters with this default configuration, the Helm chart can automatically generate the [required networking annotations in a Pod's metadata](https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom#configure-pod-manifests-rdma). Therefore, you can use the streamlined commands shown in the deployment sections above.

#### Running on a Cluster with Non-Default Network Configuration

To configure the correct networking annotations for a cluster that uses non-default names for GKE Network resources, you must provide the names of the GKE Network resources in your cluster when installing the chart. Use the following example command, remembering to replace the example values with the actual names of your cluster's GKE Network resources:

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
  --set-file workload_launcher=$REPO_ROOT/src/launchers/dynamo-vllm-launcher.sh \
  --set-file serving_config=$REPO_ROOT/src/frameworks/a3ultra/dynamo-configs/llama-3.3-70b-single-node.yaml \
  --set workload.framework=vllm \
  --set workload.model.name=meta-llama/Llama-3.3-70B-Instruct \
  --set workload.image=nvcr.io/nvidia/ai-dynamo/vllm-runtime:${RELEASE_VERSION} \
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
  --set workload.gpus=8 \
  $USER-dynamo-single-node-serving \
  $REPO_ROOT/src/helm-charts/a3ultra/inference-templates/dynamo-deployment
```

<a name="monitoring"></a>
## 6. Monitoring and Troubleshooting

[Back to Top](#table-of-contents)

View logs for different components (replace with your deployment name):

Frontend logs:
```bash
kubectl logs -f deployment/$USER-dynamo-single-node-serving-frontend -n ${NAMESPACE}
```

Decode worker logs:
```bash
kubectl logs -f deployment/$USER-dynamo-single-node-serving-vllmdecode-worker -n ${NAMESPACE}
```

Prefill worker logs:
```bash
kubectl logs -f deployment/$USER-dynamo-single-node-serving-vllmprefill-worker -n ${NAMESPACE}
```

Common issues:

* **Pods stuck in Pending**: Check if nodes have sufficient resources (especially for multi-node deployments)
* **Model download slow**: Large models like Llama 70B can take 20-30 minutes to download
* **Multi-node issues**: Verify network connectivity between nodes and proper subnet configuration

<a name="cleanup"></a>
## 7. Cleanup

[Back to Top](#table-of-contents)

List deployed releases:
```bash
helm list -n ${NAMESPACE} --filter $USER-dynamo-
```

Uninstall specific deployments:
```bash
helm uninstall $USER-dynamo-single-node-serving -n ${NAMESPACE}
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