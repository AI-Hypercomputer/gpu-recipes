# Serve Llama-3.1-70B-FP8 on G4 (NVIDIA RTX PRO 6000) with vLLM

This recipe serves [`neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8`](https://huggingface.co/neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8)
on a single 8-GPU G4 instance with vLLM, and benchmarks it with
[inference-perf](https://github.com/kubernetes-sigs/inference-perf).

The 70B model (~70 GB in FP8) does not fit on a single 96 GB RTX PRO 6000, so it is
served **tensor-parallel across all 8 GPUs** (`--tensor-parallel-size 8`).

## Prerequisites

- A G4 instance (`g4-standard-384`, 8× NVIDIA RTX PRO 6000).
- Docker with the NVIDIA container runtime.
- A Hugging Face token (`HF_TOKEN`) with access to the model.

## Serving on 8 GPUs (tensor-parallel)

G4 instances accelerate multi-GPU workloads with direct GPU
[peer-to-peer](https://cloud.google.com/blog/products/compute/g4-vms-p2p-fabric-boosts-multi-gpu-workloads/)
communication over the PCIe bus. To use it, set `NCCL_P2P_LEVEL=SYS` **before** starting
the server. Without it, tensor-parallel all-reduces fall back to host memory and decode
throughput drops by roughly an order of magnitude.

```bash
sudo docker run \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --env "NCCL_P2P_LEVEL=SYS" \
    --env "VLLM_USE_DEEP_GEMM=0" \
    --env "VLLM_MOE_USE_DEEP_GEMM=0" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
    --kv-cache-dtype fp8 \
    --max-model-len 2560 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 8192 \
    --no-enable-prefix-caching \
    --block-size 256 \
    --tensor-parallel-size 8
```

Notes on the arguments:
- `NCCL_P2P_LEVEL=SYS` — **required** on G4 to use the PCIe P2P fabric for TP all-reduces.
- `VLLM_USE_DEEP_GEMM=0` / `VLLM_MOE_USE_DEEP_GEMM=0` — block-quantized FP8 checkpoints
  crash at startup on Blackwell with DeepGEMM enabled (`Unknown SF transformation`);
  disabling DeepGEMM avoids the crash with no measurable throughput cost.
- `--kv-cache-dtype fp8` — halves KV-cache footprint, leaving ample headroom at 128/2048.
- `--max-model-len 2560` — covers the target 128 + 2048 workload with a small buffer.
- `--max-num-seqs 1024` — decode batch cap; the 128/2048 shape is KV-light so the server
  can hold a large decode batch.

## Benchmark with inference-perf

[inference-perf](https://github.com/kubernetes-sigs/inference-perf) is a
model-server-agnostic benchmarking tool that reports standardized throughput and latency
metrics.

```bash
pip install inference-perf
inference-perf --config_file inference-perf-config.yml
```

The provided [`inference-perf-config.yml`](./inference-perf-config.yml) drives 4000
requests at concurrency 960 with ISL/OSL 128/2048.

> **Important:** OSL=2048 requests take ~360 s end-to-end at high concurrency, which
> exceeds inference-perf's default 300 s request timeout. The config sets
> `load.request_timeout: 900` so long requests complete and are counted instead of being
> dropped as timeouts.


## Clean up

Delete the GCE instance and its disks when finished:

```bash
gcloud compute instances delete ${VM_NAME?} --zone=${ZONE?} --project=${PROJECT_ID?} --quiet --delete-disks=all
```
