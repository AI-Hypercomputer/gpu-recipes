#!/bin/bash
# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

set -eux # Exit immediately on error

echo "SGLang server arguments received: $@"

export HF_HOME=/ssd

# MODEL_NAME comes from the Helm 'workload.model.name'
if [ -z "$MODEL_NAME" ]; then
  echo "Error: MODEL_NAME environment variable is not set."
  exit 1
fi

echo "Launching SGLang server for Wan2.2 (Blackwell)"
echo "Using MODEL_NAME: $MODEL_NAME"

# 1. We use 'sglang serve' because 'launch_server' doesn't support Video flags correctly.
# 2. We use '--model-path' as per your lead's specific patch.
# 3. We exclude "$@" to prevent the "Unsupported config file format: None" error.

sglang serve \
  --model-path "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --tp 4 \
  --trust-remote-code \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --text-encoder-cpu-offload

echo "Server bringup is complete."
