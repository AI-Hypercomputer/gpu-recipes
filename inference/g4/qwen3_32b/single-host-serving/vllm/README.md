# Single host inference benchmark of Qwen3-32B (NVFP4) with vLLM on G4

This recipe shows how to serve and benchmark the [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) model in **NVFP4** precision using [vLLM](https://github.com/vllm-project/vllm) on a single GCP VM with G4 GPUs, driving load with the [inference-perf](https://github.com/kubernetes-sigs/inference-perf) benchmarking tool. For more information on G4 machine types, see the [GCP documentation](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-machine-types).

This is a high-throughput configuration: the model is served **data-parallel across all 8 GPUs** (one replica per GPU), which maximizes aggregate output throughput on the PCIe-connected G4 platform.

## Benchmark results

Measured on `g4-standard-384` (8× NVIDIA RTX PRO 6000 Blackwell), vLLM, NVFP4 weights + FP8 KV cache, ISL/OSL 1024/1024, concurrency 2000, 4000 requests:

| Metric | Value |
| --- | --- |
| Output throughput | ~27,300 tokens/s |
| Total throughput | ~55,000 tokens/s |
| Request throughput | ~27.1 req/s |
| TTFT (p50) | ~2.1 s |
| TPOT (p50) | ~69.7 ms |
| Successful requests | 4000 / 4000 (0 failures) |

## Before you begin

### 1. Create a GCP VM with G4 GPUs

Make sure you have the following prerequisites:
*   [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) is initialized.
*   You have a project with a GPU quota. See [Request a quota increase](https://cloud.google.com/docs/quota/view-request#requesting_higher_quota).
*   [Enable required APIs](https://console.cloud.google.com/flows/enableapi?apiid=compute.googleapis.com).

The following commands set up environment variables and create a GCE instance. The `MACHINE_TYPE` is set to `g4-standard-384` for an 8-GPU VM. The boot disk is set to 200GB to accommodate the model and dependencies.

```bash
export VM_NAME="${USER}-g4-vllm-qwen3-32b-nvfp4"
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

# Verify the driver installation and see the available GPUs.
nvidia-smi
```

## Serve the model

### 1. Install Docker and the NVIDIA Container Toolkit

Follow the official docs to install [Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then make sure the Docker daemon is running.

### 2. Launch the vLLM server

Serve `RedHatAI/Qwen3-32B-NVFP4` (a pre-quantized NVFP4 checkpoint) data-parallel across the 8 GPUs. The `expandable_segments` allocator setting avoids CUDA fragmentation OOMs in the FP4 GEMM activation buffers, and `gpu_memory_utilization=0.92` leaves headroom for them.

```bash
docker run --gpus all --ipc=host --network=host \
  -v /scratch/cache:/root/.cache \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  vllm/vllm-openai:latest \
  --model RedHatAI/Qwen3-32B-NVFP4 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --max-model-len 2560 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --block-size 256
```

Wait until the server logs `Application startup complete` before benchmarking.

## Run the benchmark with inference-perf

In a second shell on the same VM:

```bash
pip install inference-perf
```

Use the provided [`inference-perf-config.yml`](./inference-perf-config.yml) (concurrency 2000, 4000 requests, ISL/OSL 1024/1024):

```bash
inference-perf --config_file inference-perf-config.yml
```

inference-perf writes a summary report (throughput and TTFT/TPOT/ITL latency distributions) to the output directory. See the [inference-perf configuration guide](https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/config.md) to sweep other input/output shapes or concurrency levels.

## Clean up

```bash
gcloud compute instances delete ${VM_NAME?} --zone=${ZONE?} --project=${PROJECT_ID} --quiet --delete-disks=all
```
