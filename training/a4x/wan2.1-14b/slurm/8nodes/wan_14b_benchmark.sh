#!/bin/bash
#SBATCH --job-name=wan-14b-benchmark
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --partition=a4x
#SBATCH --exclusive
#SBATCH --account=tony
#SBATCH --mem=0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# --- Prep Networking ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export RDZV_ENDPOINT="${MASTER_ADDR}:29500"

# --- Define Workload ---
# Use quoted 'EOF' to prevent local expansion of $ variables
read -r -d '' CMD_WORKLOAD <<'EOF'
set -e
cd /opt/

# --- Environment Config ---
export OMP_NUM_THREADS=1
# Ensure NCCL uses high-speed fabric (exclude docker bridge/loopback)
export NCCL_SOCKET_IFNAME=^lo,docker0 
export NCCL_DEBUG=INFO 

# --- 1. Setup Repos ---

# DFM (Pure Main - Fresh Clone)
rm -rf /opt/DFM
git clone https://github.com/NVIDIA-NeMo/DFM.git /opt/DFM
export DFM_PATH=/opt/DFM

# Megatron-Bridge (Pinned)
rm -rf /opt/Megatron-Bridge
git clone --no-checkout https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /opt/Megatron-Bridge
git -C /opt/Megatron-Bridge checkout 953aabf75c0500180dc14a6a76cf9e7e7c4baec7

# Megatron-LM (Pinned)
rm -rf /opt/Megatron-LM
git clone --no-checkout https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM
git -C /opt/Megatron-LM checkout 2d398b42fd4237fffb553109563d73ac099751c3

# --- 2. Install Deps ---
export PYTHONPATH="${DFM_PATH}/.:/opt/Megatron-Bridge/.:/opt/Megatron-LM"
pip install --upgrade diffusers==0.35.1 easydict imageio imageio-ffmpeg > /dev/null 2>&1

# --- 3. Run Training ---
echo "Starting Training on Node ${SLURM_NODEID}..."
cd ${DFM_PATH}

# Note: Using SLURM_NNODES/SLURM_GPUS_ON_NODE allows the script to adapt to SBATCH header changes automatically
NVTE_FUSED_ATTN=1 torchrun \
  --nnodes=${SLURM_NNODES} \
  --nproc_per_node=${SLURM_GPUS_ON_NODE} \
  --node_rank=${SLURM_NODEID} \
  --rdzv_id=${SLURM_JOB_ID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${RDZV_ENDPOINT} \
  examples/megatron/recipes/wan/pretrain_wan.py \
  --config-file examples/megatron/recipes/wan/conf/gb200_perf_pretrain_mock.yaml \
  --training-mode pretrain \
  --mock \
  train.train_iters=30 
EOF

# --- Execute ---
export SLURM_JOB_ID
export RDZV_ENDPOINT

srun \
  --container-image="nvcr.io#nvidia/nemo:25.11.00" \
  --container-mounts="/lustre/fsw/" \
  --container-writable \
  --no-container-mount-home \
  --export=ALL \
  bash -c "${CMD_WORKLOAD}"
