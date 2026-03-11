#!/bin/bash
set -eux

export HF_HOME=/ssd

if [ -z "$MODEL_NAME" ]; then
  echo "Error: MODEL_NAME environment variable is not set."
  exit 1
fi

echo "Launching SGLang for Wan2.2 on 4x B200 GPUs"

# We add --sp-degree 4 to satisfy: 2 (ulysses) * 2 (ring) = 4
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
