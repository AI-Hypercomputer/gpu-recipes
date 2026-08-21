# Serve and benchmark Qwen3-32B on G4 (NVIDIA RTX PRO 6000) with vLLM

This recipe shows how to serve and benchmark Qwen3-32B on a single 8-GPU G4 instance
with [vLLM](https://github.com/vllm-project/vllm), using
[inference-perf](https://github.com/kubernetes-sigs/inference-perf) for standardized
throughput/latency measurement. Both **FP8** (`Qwen/Qwen3-32B-FP8`) and **NVFP4**
(`nvidia/Qwen3-32B-NVFP4`) checkpoints are covered.

The 32B model fits on a single 96 GB RTX PRO 6000, so for maximum aggregate throughput it
is served **data-parallel** (one replica per GPU, `--data-parallel-size 8`).

## Before you begin

### 1. Create a GCP VM with G4 GPUs

Make sure you have the following prerequisites:
*   [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) is initialized.
*   A project with GPU quota. See [Request a quota increase](https://cloud.google.com/docs/quota/view-request#requesting_higher_quota).
*   [Enable required APIs](https://console.cloud.google.com/flows/enableapi?apiid=compute.googleapis.com).

The following commands set up environment variables and create a GCE instance.
`MACHINE_TYPE` is set to `g4-standard-384` for an 8-GPU VM. The boot disk is set to 200GB
to accommodate the models and dependencies.

```bash
export VM_NAME="${USER}-g4-test"
export PROJECT_ID="your-project-id"
export ZONE="your-zone"
export MACHINE_TYPE="g4-standard-384"
export IMAGE_PROJECT="ubuntu-os-accelerator-images"
export IMAGE_FAMILY="ubuntu-accelerator-2404-amd64-with-nvidia-570"

gcloud compute instances create ${VM_NAME} \
  --machine-type=${MACHINE_TYPE} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --image-project=${IMAGE_PROJECT} \
  --image-family=${IMAGE_FAMILY} \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB
```

### 2. Connect to the VM

```bash
gcloud compute ssh ${VM_NAME?} --project=${PROJECT_ID?} --zone=${ZONE?}
# Verify the driver installation and available GPUs.
nvidia-smi
```

## Install dependencies

### 1. Install Docker

Follow the official documentation to install Docker on Ubuntu:
[Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/). Make
sure the Docker daemon is running.

### 2. Install NVIDIA Container Toolkit

Follow the official NVIDIA documentation:
[NVIDIA Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
This lets the container access the host NVIDIA driver.

## Serve the model (8 GPUs, data-parallel)

For maximum aggregate throughput on the balanced `1024/1024` workload, serve the model
data-parallel — one replica per GPU — so all 8 GPUs are saturated with independent request
streams.

### FP8 (`Qwen/Qwen3-32B-FP8`)

```bash
sudo docker run \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    --env "VLLM_ATTENTION_BACKEND=FLASHINFER" \
    --env "VLLM_USE_DEEP_GEMM=0" \
    --env "VLLM_MOE_USE_DEEP_GEMM=0" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-32B-FP8 \
    --kv-cache-dtype fp8 \
    --max-model-len 2560 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 8192 \
    --no-enable-prefix-caching \
    --block-size 256 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8
```

`VLLM_USE_DEEP_GEMM=0` / `VLLM_MOE_USE_DEEP_GEMM=0` are required for block-quantized FP8
checkpoints, which otherwise crash at startup on Blackwell (`Unknown SF transformation`).

### NVFP4 (`nvidia/Qwen3-32B-NVFP4`)

```bash
sudo docker run \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    --env "VLLM_ATTENTION_BACKEND=FLASHINFER" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model nvidia/Qwen3-32B-NVFP4 \
    --kv-cache-dtype fp8 \
    --max-model-len 2560 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 8192 \
    --no-enable-prefix-caching \
    --block-size 256 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8
```

Common flags:
- `--kv-cache-dtype fp8` — halves the KV-cache footprint.
- `--max-model-len 2560` — covers the 1024 + 1024 workload with a small buffer.
- `--tensor-parallel-size 1 --data-parallel-size 8` — one independent replica per GPU.
- `VLLM_ATTENTION_BACKEND=FLASHINFER` — attention backend used for the tuned config.

## Benchmark with inference-perf

[inference-perf](https://github.com/kubernetes-sigs/inference-perf) is a
model-server-agnostic benchmarking tool that reports standardized throughput and latency
metrics. Install it and run against the server above:

```bash
pip install inference-perf
inference-perf --config_file inference-perf-config.yml
```

The provided [`inference-perf-config.yml`](./inference-perf-config.yml) drives 4000
requests at concurrency 2000 with ISL/OSL 1024/1024 against `Qwen/Qwen3-32B-FP8`. For the
NVFP4 server, set `server.model_name` and `tokenizer.pretrained_model_name_or_path` to
`nvidia/Qwen3-32B-NVFP4`. See the
[inference-perf configuration guide](https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/config.md)
to sweep other shapes or concurrency levels.

## Clean up

This command deletes the GCE instance and all its disks.

```bash
gcloud compute instances delete ${VM_NAME?} --zone=${ZONE?} --project=${PROJECT_ID} --quiet --delete-disks=all
```
