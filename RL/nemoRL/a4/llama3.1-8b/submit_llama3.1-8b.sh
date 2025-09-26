#!/bin/bash
WANDB_API_KEY='' # Update this with your WANDB API key
HF_TOKEN='' # Update this with your HF token

# --- Step 1: Find the Ray Head Pod ---
echo "Finding Ray head pod..."
export HEAD_POD_NAME=$(kubectl get pods --selector=ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
if [ -z "$HEAD_POD_NAME" ]; then
    echo "Error: No running Ray head pod found. Please check your cluster."
    exit 1
fi
echo "Found head pod: $HEAD_POD_NAME"
echo ""

# --- Step 2: Define the Job Script to Run ---
# This is the script that will be executed *inside* the head pod.
# It assumes the 'uv venv' setup from the values.yaml is already done.
JOB_SCRIPT=$(cat <<EOF
set -ex

echo "--- Running on Ray Head Pod ($HOSTNAME) ---"
cd /opt/nemo-rl

echo "Setting environment variables..."
export WANDB_API_KEY=$WANDB_API_KEY
export HF_TOKEN=$HF_TOKEN
export HF_HOME=/opt/nemo-rl/

###-----Example to launch llama 3.1 8b on 4 nodes (32 GPUs)----------
uv run python examples/run_grpo_math.py \
  --config examples/configs/grpo_math_8B.yaml \
  logger.wandb_enabled=True \
  cluster.num_nodes=4 \
  cluster.gpus_per_node=8 \
  logger.wandb.name='llama3.1-8b-grpo-4nodes'

echo "--- Job Finished ---"
EOF
)

# --- Step 3: Execute the Job ---
echo "Submitting job to $HEAD_POD_NAME..."
echo "$JOB_SCRIPT" | tr -d '\r' | kubectl exec -i $HEAD_POD_NAME -c ray-head -- /bin/bash

echo ""
echo "Job submission complete."