#!/bin/bash
set -eux

# Critical: Force visibility and distributed discovery for B200
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SGLANG_AUTO_FORGE_TP=4
export HF_HOME=/ssd

# Standard Blackwell NCCL settings for GKE/A4X
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PIX

if [ -z "$MODEL_NAME" ]; then
  echo "Error: MODEL_NAME environment variable is not set."
  exit 1
fi

echo "Launching SGLang for Wan2.2 on 4x B200 GPUs (Blackwell)"

sglang serve \
  --model-path "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --tp 4 \
  --sp-degree 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --trust-remote-code \
  --dist-backend nccl \
  --text-encoder-cpu-offload
