#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=llama3-70b-pretrain-a3u
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --output=logs/%x_%u_%j.out
#SBATCH --time=24:00:00
#SBATCH --open-mode=append

set -e

# Master Node Setup
nodes=( $( scontrol show hostnames ${SLURM_JOB_NODELIST} ) )
head_node=${nodes[0]}
export MASTER_ADDR=$head_node
export MASTER_PORT=6005
echo "Master Node: $MASTER_ADDR"

#  Infrastructure Config
export MGMT_IFACE=enp0s19
export PMIX_MCA_gds=^ds12

# Control Plane
export UCX_TLS=tcp,sm,self
export UCX_NET_DEVICES=$MGMT_IFACE

# Gloo
export GLOO_SOCKET_IFNAME=$MGMT_IFACE
export GLOO_TIMEOUT_SECONDS=1200

# Performance & Cache Setup
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# PATHS - Modify these if your setup differs
if [[ ! -d "${BASE_DIR}" ]]; then
    echo "Error: BASE_DIR ${BASE_DIR} not found. Please ensure it exists."
    exit 1
fi
CONTAINER="${BASE_DIR}/sqsh/nvidia+nemo+25.09.sqsh"
RUN_SCRIPT="${BASE_DIR}/Megatron-Bridge/scripts/performance/run_script.py"
CONFIG_FILE="${BASE_DIR}/Megatron-Bridge/scripts/performance/configs/llama3/llama3_70b_llm_pretrain.yaml"
TOTAL_GPUS=$(( SLURM_JOB_NUM_NODES * 8 ))


# CACHE SETUP
CACHE_ROOT="${LOCAL_SSD_DIR}/triton_cache"
INDUCTOR_ROOT="${LOCAL_SSD_DIR}/torchinductor_cache"
JOB_CACHE_DIR="${CACHE_ROOT}/${SLURM_JOBID}"
JOB_INDUCTOR_DIR="${INDUCTOR_ROOT}/${SLURM_JOBID}"

echo "Pre-creating cache directories on host..."
mkdir -p "${JOB_CACHE_DIR}"
mkdir -p "${JOB_INDUCTOR_DIR}"

echo "Submitting job on $SLURM_JOB_NUM_NODES nodes ($TOTAL_GPUS GPUs)..."

#  Launch
srun \
  --container-image "${CONTAINER}" \
  --container-mounts "${BASE_DIR}:${BASE_DIR},/usr/local/gib:/usr/local/gib" \
  --container-workdir "${BASE_DIR}" \
  --no-container-mount-home \
  --container-writable \
  --gres=gpu:8 \
  -l \
  --mpi=pmix \
  bash -c "

    # Ensure clean slate inside container
    unset NCCL_SOCKET_IFNAME
    unset NCCL_IB_DISABLE
    unset NCCL_GPUDIRECT_TCPX_FORCE_ACK

    # Distributed Ranks
    export WORLD_SIZE=\${SLURM_NTASKS}
    export RANK=\${SLURM_PROCID}
    export LOCAL_RANK=\${SLURM_LOCALID}
    export NODE_RANK=\${SLURM_NODEID}

    export TRITON_CACHE_DIR=${JOB_CACHE_DIR}/node_\${SLURM_NODEID}
    export TORCHINDUCTOR_CACHE_DIR=${JOB_INDUCTOR_DIR}/node_\${SLURM_NODEID}
    mkdir -p \${TRITON_CACHE_DIR} \${TORCHINDUCTOR_CACHE_DIR}

    export LD_LIBRARY_PATH=/usr/local/gib/lib64:\${LD_LIBRARY_PATH}

    # Source environment scripts
    if [ -f /usr/local/gib/scripts/set_nccl_env.sh ]; then
        source /usr/local/gib/scripts/set_nccl_env.sh
    fi

    echo \"Rank \${RANK} starting on host \$(hostname)...\"

    python ${RUN_SCRIPT} \
    --config_file ${CONFIG_FILE} \
    --model_name llama3 \
    --model_size 70b \
    --compute_dtype fp8 \
    --fp8_recipe cs \
    --gpu h100 \
    -a dummy -p dummy \
    -ng ${TOTAL_GPUS} \
    train.manual_gc=true \
    train.manual_gc_interval=100
  "
