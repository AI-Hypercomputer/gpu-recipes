#!/bin/bash
# launch-workload.sh
set -eux

# 1. Hardware Environment Setup
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/ssd

# 2. NCCL Optimization for Blackwell NVLink
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PIX

if [ -z "$MODEL_NAME" ]; then
  echo "Error: MODEL_NAME environment variable is not set."
  exit 1
fi

# 3. Dynamic IP Discovery for Distributed Init
# We use the internal Pod IP for the head-worker handshake
export MY_IP=$(hostname -I | awk '{print $1}')

echo "Launching SGLang for Wan2.2 on 4x B200 GPUs (Blackwell)"

# 4. Start Server
# No CPU offloading - keeping 14B model entirely in Blackwell VRAM
sglang serve \
  --model-path "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --tp 4 \
  --trust-remote-code \
  --dist-init-addr "$MY_IP:6000"
