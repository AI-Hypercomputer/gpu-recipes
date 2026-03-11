#!/bin/bash
set -eux

# 1. Force GPU Visibility and TP settings for B200
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SGLANG_AUTO_FORGE_TP=4
export HF_HOME=/ssd

# 2. Optimized NCCL settings for Blackwell/GKE
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PIX

if [ -z "$MODEL_NAME" ]; then
  echo "Error: MODEL_NAME environment variable is not set."
  exit 1
fi

echo "Launching SGLang for Wan2.2 on 4x B200 GPUs (Blackwell)"

# 3. CRITICAL: Extract ONLY the IPv4 address to fix the discovery bug
export MY_IP=$(hostname -I | awk '{print $1}')

# 4. Launch SGLang with Explicit Distributed Address
sglang serve \
  --model-path "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --tp 4 \
  --trust-remote-code \
  --dist-backend nccl \
  --dist-init-addr "$MY_IP:6000" \
  --text-encoder-cpu-offload
