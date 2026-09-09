<!-- mdformat global-off -->
# Pretrain LLaMA 3.1 405B on A4X GKE Node Pools with NVIDIA Megatron-Bridge & NeMo 26.06.01 (256 GPUs, FP8-MX)

This recipe outlines the steps for running a **LLaMA 3.1 405B** pretraining workload on [A4X (NVIDIA GB200) GKE Node Pools](https://cloud.google.com/kubernetes-engine) using the [Megatron-Bridge pretraining framework](https://github.com/NVIDIA-NeMo/Megatron-Bridge) with **NeMo 26.06.01** and micro-scaling FP8 (`fp8_mx`).

## Measured Benchmark Performance

* **Measured Throughput**: **1,984.25 MODEL_TFLOP/s/GPU**
* **NVIDIA DGX-B200 Roofline Target**: 1,976.00 MODEL_TFLOP/s/GPU
* **% Reached (GCP / NVIDIA Roofline)**: **100.42%**
* **Steady-State Step Latency**: **62.53s / step**
* **Model FLOPs Utilization (MFU)**: **44.09% MFU** (based on 4,500 TFLOP/s FP8 peak)
* **Scale**: 256 GPUs (64 nodes $\times$ 4 GPUs)
* **uBench Dashboard**: [uBench Benchmark Dashboard](https://dashboards.corp.google.com/view/_bdba3bae_b77f_442d_b99a_3afb3c2aee9b?fb=run_id:in:megatron_bridge_training-nemo2606/llama31_405b_256gpus_fp8mx_seq8192_gbs1536-2026-08-20_091758-1f6d19dc-857d-466b-9988-d66662832c96)

## Parallelism & Sizing Configuration

* **Model Family**: `llama` (`llama31_405b`)
* **Precision / Compute Dtype**: `fp8_mx` (Micro-scaling FP8)
* **Global Batch Size ($GBS$)**: `1536`
* **Micro Batch Size ($MBS$)**: `1`
* **Sequence Length**: `8192`
* **Tensor Parallelism ($TP$)**: `4`
* **Pipeline Parallelism ($PP$)**: `16`
* **Virtual Pipeline Parallelism ($VPP$)**: `8`
* **Context Parallelism ($CP$)**: `1`
* **Expert Parallelism ($EP$)**: `1` (Dense model)
* **Data Parallelism ($DP$)**: `4` ($256 / (4 \times 16 \times 1) = 4$ model replicas)

## Orchestration & Deployment Tools

* **Orchestration**: [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine) with [Kueue Topology-Aware Scheduling (TAS)](https://kueue.sigs.k8s.io/) configured for 16-node subblock slicing (`cloud.google.com/gce-topology-subblock`).
* **Deployment**: Helm chart deploying a Kubernetes [JobSet](https://github.com/kubernetes-sigs/jobset) resource.
* **Containers**:
  * NeMo Container: `nvcr.io/nvidia/nemo:26.06.01`
  * GPUDirect-TCPX / gib Plugin: `us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-arm64:v1.1.2`

## Run the Recipe

### 1. Configure Environment Settings

```bash
export PROJECT_ID=<PROJECT_ID>
export CLUSTER_REGION=<CLUSTER_REGION>
export CLUSTER_NAME=<CLUSTER_NAME>
export GCS_BUCKET=<GCS_BUCKET> # Note: path should not be prefixed with gs://
export KUEUE_NAME=<KUEUE_NAME> # Default: tas-lq
export HF_TOKEN=<YOUR_HF_TOKEN>
```

Replace:
* `<PROJECT_ID>`: Your Google Cloud project ID.
* `<CLUSTER_REGION>`: Region where your A4X cluster is located.
* `<CLUSTER_NAME>`: Name of your A4X GKE cluster.
* `<GCS_BUCKET>`: Name of your Cloud Storage bucket (without `gs://`).
* `<KUEUE_NAME>`: Kueue local queue name (e.g. `tas-lq`).
* `<YOUR_HF_TOKEN>`: Your Hugging Face access token.

```bash
gcloud config set project $PROJECT_ID
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

### 2. Get the Recipe

```bash
git clone https://github.com/AI-Hypercomputer/gpu-recipes.git
cd gpu-recipes/training/a4x/llama31_405b/megatron-bridge-gke/nemo2606/256gpus-fp8mx-seq8192-gbs1536/recipe
```

### 3. Deploy the Workload

```bash
helm install llama31-405b-256gpus-fp8mx . \
  --set queue=${KUEUE_NAME} \
  --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
  --set volumes.gcsMounts[0].mountPath=/runtime-logs \
  --set workload.envs[0].name=ARTIFACT_DIR \
  --set workload.envs[0].value=/runtime-logs/llama31-405b-256gpus-fp8mx/artifacts
```

### 4. Monitor Progress

```bash
kubectl get jobset llama31-405b-256gpus-fp8mx
kubectl logs -l jobset.sigs.k8s.io/jobset-name=llama31-405b-256gpus-fp8mx -c workload -f
```
