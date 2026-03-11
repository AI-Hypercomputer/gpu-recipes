#!/bin/bash
# launch-workload.sh
set -eux

# 1. Environment & Path Setup
export HF_HOME=/ssd
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 2. Dynamic GPU Detection for TP
# This ensures the same script works for both 1-chip and 4-chip deployments
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs. Setting TP to $NUM_GPUS."

# 3. Optimized NCCL settings for Blackwell NVLink
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PIX

if [ -z "$MODEL_NAME" ]; then
  echo "Error: MODEL_NAME environment variable is not set."
  exit 1
fi

# 4. Dynamic IP Discovery for Distributed Init
export MY_IP=$(hostname -I | awk '{print $1}')

echo "Launching SGLang for $MODEL_NAME on $NUM_GPUS GPUs"

# 5. Start Server
# Removed CPU offloading to maximize Blackwell VRAM performance
sglang serve \
  --model-path "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --tp "$NUM_GPUS" \
  --trust-remote-code \
  --dist-init-addr "$MY_IP:6000"
