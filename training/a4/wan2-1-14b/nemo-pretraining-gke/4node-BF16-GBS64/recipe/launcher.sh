usage()
{
cat << EOF
usage: bash ./launcher.sh [config-override  [config-override ...]]
config-override  (Optional) A  NeMo configuration override. E.g. trainer.max_steps=10000.
EOF
}

parse_args() {
  while [ "$1" != "" ]; do
    case $(grep -o "=" <<< "$1" | wc -l) in
        1  )
        config_overrides+=("$1")
        ;;
        * )
            echo "Invalid config override: $1"
            usage
            exit 1
    esac
    shift
  done
  config_overrides="${config_overrides[*]}"
}

config_overrides=()
parse_args "$@"

if [ -z "${config_overrides}" ]; then
  echo "No NeMo config overrides specified"
else
  echo "NeMo config overrides:"
  echo "  ${config_overrides}"
fi

export LD_LIBRARY_PATH="$NCCL_PLUGIN_PATH"
ldconfig $LD_LIBRARY_PATH
echo "Added $LD_LIBRARY_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""

if [[ -n "${EXPLICIT_LOG_DIR}" ]]; then
  explicit_log_dir=${EXPLICIT_LOG_DIR}
else
  explicit_log_dir=/workload_logs
fi
echo "Logging to ${explicit_log_dir}"

if [[ -n "${TOKENIZER_PATH}" ]]; then
  echo "Getting tokenizer files"
  cp ${TOKENIZER_PATH}/* .
  echo ""
fi

echo "Launching Torch distributed on the node rank $JOB_COMPLETION_INDEX out of $NNODES nodes"

pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger

# Create the nsys directory.
mkdir -p ${explicit_log_dir}/nsys

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
  if [ "\$RANK" -eq "0" ]; then
    echo "Worker 0 is stalling for a few seconds.." ;
    sleep 3 ;
    echo "The detected environment within worker rank 0 is:" ;
    env | sed 's/^/  /' ;
  fi ;

  cd /opt/DFM ;

  numactl \
    --cpunodebind=\$((LOCAL_RANK/4)) \
    --membind=\$((LOCAL_RANK/4)) \
  nice -10 \
  python examples/megatron/recipes/wan/pretrain_wan.py \
    --config-file examples/megatron/recipes/wan/conf/gb200_perf_pretrain_mock.yaml \
    --training-mode pretrain \
    --mock \
    checkpoint.save_interval=0 \
    model.tensor_model_parallel_size=4 \
    model.context_parallel_size=2 \
    model.sequence_parallel=true \
    model.recompute_granularity=full \
    model.recompute_method=block \
    model.recompute_num_layers=40 \
    \${config_overrides}
EOM
)

echo "$worker_command" > worker_command.sh
chmod 777 worker_command.sh

torchrun \
--nproc-per-node="8" \
--nnodes="4" \
--node_rank="${JOB_COMPLETION_INDEX}" \
--rdzv_id="${JOB_IDENTIFIER}" \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
--no-python bash worker_command.sh


if [[ "$JOB_COMPLETION_INDEX" == "0" ]]; then
  mkdir -p ${ARTIFACT_DIR}
  cp -r ${explicit_log_dir}/* ${ARTIFACT_DIR}/
  env > ${ARTIFACT_DIR}/environ.txt
  ls ${ARTIFACT_DIR}
fi
echo "Training completed"
echo "Pod on $(hostname --fqdn) is exiting"