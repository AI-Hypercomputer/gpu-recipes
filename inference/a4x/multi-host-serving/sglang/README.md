# Multi-Host Inference of Kimi-K3 with SGLang on A4X GKE Node Pool

This recipe provides instructions and templates for serving Moonshot AI's **Kimi-K3** (2.8T parameter hybrid MoE vision-language model) using multi-host [SGLang](https://github.com/sgl-project/sglang) across 4 A4X nodes (16 GPUs) on [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine).

## Table of Contents

* [1. Test Environment](#test-environment)
* [2. High-Level Architecture](#architecture)
* [3. Environment Setup](#environment-setup)
* [4. Deployment Instructions](#deployment-instructions)
* [5. Inference Requests & Benchmarking](#inference-requests)
* [6. Cleanup](#cleanup)

<a name="test-environment"></a>
## 1. Test Environment

* **Orchestration**: Google Kubernetes Engine (GKE)
* **Compute**: 4 × `a4x-highgpu-4g` nodes (16 NVIDIA GB200 GPUs total)
* **Serving Engine**: `lmsysorg/sglang:kimi-k3`
* **Inter-Node Interconnect**: NVIDIA Multi-Node NVLink (MNNVL / NVL72) via `resource.nvidia.com/v1beta1` `ComputeDomain` DRA resource claim
* **Parallelism Strategy**: Tensor Parallelism (`TP=16`), Decode Context Parallelism (`DCP=16`), Mamba State Ratio `0.86`, Extra Slots `16`

<a name="architecture"></a>
## 2. High-Level Architecture

The deployment utilizes `LeaderWorkerSet` (LWS) with a 4-node pod group. A Kubernetes `ComputeDomain` resource (`resource.nvidia.com/v1beta1`) binds all 4 nodes into a single NVL72 clique for low-latency NVLink communication across nodes.

```
+-------------------------------------------------------------------------------+
|                           ComputeDomain (NVL72)                               |
|                                                                               |
|  +---------------------+   +---------------------+   +---------------------+  |
|  | Leader Pod (Rank 0) |---| Worker Pod (Rank 1) |---| Worker Pod (Rank 2) |  |
|  | 4 × GB200 GPUs      |   | 4 × GB200 GPUs      |   | 4 × GB200 GPUs      |  |
|  +---------------------+   +---------------------+   +---------------------+  |
|             \                                                   /             |
|              +-----------------+ Worker Pod (Rank 3) +---------+              |
|                                | 4 × GB200 GPUs      |                        |
|                                +---------------------+                        |
+-------------------------------------------------------------------------------+
```

<a name="environment-setup"></a>
## 3. Environment Setup

Ensure your GKE cluster is connected and your environment variables are exported:

```bash
export CLUSTER_NAME="a4x-baker"
export REGION="us-central1"
export NAMESPACE="default"

gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION}
```

<a name="deployment-instructions"></a>
## 4. Deployment Instructions

Deploy the LeaderWorkerSet Helm chart for Kimi-K3:

```bash
cd inference/a4x/multi-host-serving/sglang

helm install -f values_kimi_k3.yaml \
  --namespace ${NAMESPACE} \
  pirillo-sglang-kimi-k3 \
  ../../../../src/helm-charts/a4x/inference-templates/lws-deployment
```

Monitor pod status:

```bash
kubectl get pods -n ${NAMESPACE} -l leaderworkerset.sigs.k8s.io/name=pirillo-sglang-kimi-k3 -o wide
```

<a name="inference-requests"></a>
## 5. Inference Requests & Benchmarking

### 5.1. OpenAI-Compatible Chat Completion API (Vision & Text)

```bash
curl http://pirillo-sglang-kimi-k3-svc.default.svc.cluster.local:30100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K3",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image in detail."},
          {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png"}}
        ]
      }
    ]
  }'
```

### 5.2. Serving Benchmark (8k Input / 1k Output)

Run the SGLang benchmark client against port 30100:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host pirillo-sglang-kimi-k3-svc \
  --port 30100 \
  --dataset-name random \
  --num-prompts 100 \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --max-concurrency 32
```

<a name="cleanup"></a>
## 6. Cleanup

```bash
helm uninstall pirillo-sglang-kimi-k3 -n ${NAMESPACE}
```
