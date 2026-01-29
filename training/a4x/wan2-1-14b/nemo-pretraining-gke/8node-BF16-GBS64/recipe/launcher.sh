#!/bin/bash

usage() {
cat << EOF
usage: bash ./launcher.sh [config-override [config-override ...]]
EOF
}

parse_args() {
  while [ "$1" != "" ]; do
    case $(grep -o "=" <<< "$1" | wc -l) in
        1) config_overrides+=("$1") ;;
        *) echo "Invalid config override: $1"; usage; exit 1 ;;
    esac
    shift
  done
  # Convert array to space-separated string
  config_overrides="${config_overrides[*]}"
}

config_overrides=()
parse_args "$@"

export LD_LIBRARY_PATH="$NCCL_PLUGIN_PATH"
ldconfig $LD_LIBRARY_PATH
echo "Added $LD_LIBRARY_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""

explicit_log_dir=${EXPLICIT_LOG_DIR:-workload_logs}
mkdir -p ${explicit_log_dir}

export NCCL_PLUGIN_PATH=/usr/local/gib/lib64
export LD_LIBRARY_PATH="/usr/local/gib/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"

if [[ -n "${NCCL_INIT_SCRIPT}" ]]; then
  source ${NCCL_INIT_SCRIPT}
fi

export NCCL_NET=gIB
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_ALGO=Ring,Tree
export NCCL_CUMEM_ENABLE=0
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=16

export CUDA_DEVICE_MAX_CONNECTIONS=1 
export NVTE_FUSED_ATTN=1
# Repos & Pinning (Wan Stable Build)
cd /opt
rm -rf DFM Megatron-Bridge Megatron-LM

# Fresh clone of DFM
git clone https://github.com/NVIDIA-NeMo/DFM.git /opt/DFM

# Pin Bridge & LM to known stable commits for this workflow
git clone --no-checkout https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /opt/Megatron-Bridge
git -C /opt/Megatron-Bridge checkout 953aabf75c0500180dc14a6a76cf9e7e7c4baec7

git clone --no-checkout https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM
git -C /opt/Megatron-LM checkout 2d398b42fd4237fffb553109563d73ac099751c3

# Fix automodel info logging flood
sed -i 's/logger.info/logger.warning/g' /opt/DFM/dfm/src/automodel/flow_matching/flow_matching_pipeline.py

# Fix data length issue with DFM
sed -i 's/length=1024/length=10**12/g' /opt/DFM/dfm/src/megatron/data/wan/wan_mock_datamodule.py

sed -i 's/length: int/length: int = 10**12/g' /opt/DFM/dfm/src/megatron/data/wan/wan_mock_datamodule.py
export PYTHONPATH="/opt/DFM:/opt/Megatron-Bridge:/opt/Megatron-LM"

pip install --no-cache-dir \
  diffusers==0.35.1 \
  easydict \
  imageio \
  imageio-ffmpeg \
  peft \
  "transformers<4.57.0" \
  nvidia-modelopt[hf] \
  > /dev/null 2>&1

worker_command=$(cat <<- EOM
  # Coordination: Let Rank 0 dump the network state for debugging
  if [ "\$RANK" -eq "0" ]; then
    echo "Initializing environment..." ;
    sleep 5 ; 
    env | grep -E "NCCL|JOB|MASTER" ;
  fi ;

  cd /opt/DFM ;

  # Performance: Bind to local CPU/Memory socket to fix 'unspecified launch failure'
  # Config: Use '+' prefix to ensure Hydra adds/overrides keys correctly
  numactl \\
    --cpunodebind=\$((LOCAL_RANK/2)) \\
    --membind=\$((LOCAL_RANK/2)) \\
  nice -10 \\
  python examples/megatron/recipes/wan/pretrain_wan.py \\
    --config-file examples/megatron/recipes/wan/conf/gb200_perf_pretrain_mock.yaml \\
    --training-mode pretrain \\
    --mock \\
    checkpoint.save_interval=0 \
    ${config_overrides}
EOM
)

echo "$worker_command" > worker_command.sh
chmod 777 worker_command.sh

echo "Launching Wan 2.1 14B on ${NNODES} nodes..."

torchrun \
  --nproc-per-node="${GPUS_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${JOB_COMPLETION_INDEX}" \
  --rdzv_id="${JOB_IDENTIFIER}" \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --no-python bash worker_command.sh

if [[ "$JOB_COMPLETION_INDEX" == "0" ]]; then
  mkdir -p ${ARTIFACT_DIR}
  cp -r ${explicit_log_dir}/* ${ARTIFACT_DIR}/ 2>/dev/null || true
  env > ${ARTIFACT_DIR}/environ.txt
fi