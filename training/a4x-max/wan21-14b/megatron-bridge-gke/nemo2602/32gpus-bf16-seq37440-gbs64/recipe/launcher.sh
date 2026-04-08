usage()
{
cat << EOF
usage: bash ./launcher.sh [config-override  [config-override ...]]
config-override  (Optional) A  NeMo configuration override. E.g. trainer.max_steps=10000.
EOF
}

parse_args() {
  while [[ "$1" != "" ]]; do
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

if [[ -z "${config_overrides[*]}" ]]; then
  echo "No NeMo config overrides specified"
else
  echo "NeMo config overrides:"
  echo "  ${config_overrides}"
fi

export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:$NCCL_PLUGIN_PATH:$LD_LIBRARY_PATH"
ldconfig "$LD_LIBRARY_PATH"
echo "Added $LD_LIBRARY_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""

if [[ -n "${EXPLICIT_LOG_DIR}" ]]; then
  explicit_log_dir="${EXPLICIT_LOG_DIR}"
else
  explicit_log_dir="workload_logs"
fi

# Ensure explicit_log_dir is an absolute path before any cd commands
if [[ "$explicit_log_dir" != /* ]]; then
  explicit_log_dir="${PWD}/${explicit_log_dir}"
fi
echo "Logging to ${explicit_log_dir}"

if [[ -n "${TOKENIZER_PATH}" ]]; then
  echo "Getting tokenizer files"
  cp "${TOKENIZER_PATH}"/* .
  echo ""
fi

echo "Launching Torch distributed on the node rank $JOB_COMPLETION_INDEX out of $NNODES nodes"

pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger

# Create the nsys directory.
mkdir -p "${explicit_log_dir}/nsys"

# Collect diagnostics to a single line
kv="\"kernel_version\": \"$(uname --kernel-release)\""
if command -v nvidia-smi &> /dev/null; then
  cuda_v=$(nvidia-smi -q -x | grep -Po '(?<=<cuda_version>).*(?=</cuda_version>)' || true)
  driver_v=$(nvidia-smi -q -x | grep -Po '(?<=<driver_version>).*(?=</driver_version>)' || true)
  vbios_v=$(nvidia-smi -q -x | grep -Po '(?<=<vbios_version>).*(?=</vbios_version>)' | head -n1 || true)
  kv="${kv}, \"cuda_version\": \"${cuda_v}\""
  kv="${kv}, \"driver_version\": \"${driver_v}\""
  kv="${kv}, \"vbios_version\": \"${vbios_v}\""
fi
echo "VERSION_DIAGNOSTICS: {${kv}}"


export NVTE_FUSED_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Repos & Pinning (Wan Stable Build 26.02)
cd /opt
rm -rf DFM Megatron-Bridge

# Fresh clone of DFM
git clone --no-checkout https://github.com/NVIDIA-NeMo/DFM.git /opt/DFM
git -C /opt/DFM checkout 7cb7634189b00cee16bbf0b3d5ee1f9098f6f9d8

# Pin Bridge utilizing the universal container commit map for Nemo 26.02
git clone --no-checkout https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /opt/Megatron-Bridge
git -C /opt/Megatron-Bridge checkout 9c9dd848966322fc3ed7706747ec13219ef49dda

# Init Megatron-core module cleanly instead of separate LM checkout
cd /opt/Megatron-Bridge && git submodule update --init --recursive

export PYTHONPATH="/opt/DFM:/opt/Megatron-Bridge"

# Python deps
python3 -m pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger

# Unset NVIDIA container global package constraints to prevent it from forcibly downgrading transformers<4.57
export PIP_CONSTRAINT=""
python3 -m pip install --upgrade diffusers==0.35.1
python3 -m pip install easydict imageio imageio-ffmpeg peft "transformers>=4.57.1" nvidia-modelopt[hf]

# Fix automodel info logging flood
sed -i 's/logger.info/logger.warning/g' /opt/DFM/dfm/src/automodel/flow_matching/flow_matching_pipeline.py || true


worker_command=$(cat <<- EOM
  if [ "\$RANK" -eq "0" ]; then
    echo "Worker 0 is stalling for a few seconds.." ;
    sleep 3 ;
    echo "The detected environment within worker rank 0 is:" ;
    env | sed 's/^/  /' ;
  fi ;

  cd /opt/DFM ;

  numactl \
    --cpunodebind=\$((LOCAL_RANK/2)) \
    --membind=\$((LOCAL_RANK/2)) \
  nsys profile \
    -t nvtx,cuda \
    --cuda-event-trace=false \
    --sample=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --kill none \
    -o "/${explicit_log_dir}/$JOB_IDENTIFIER/rank-\$RANK" \
    --force-overwrite true \
    --session-new "nsys-\$RANDOM-\$RANK" \
  nice -10 \
  python examples/megatron/recipes/wan/pretrain_wan.py \
    --training-mode pretrain \
    --mock \
    --config-file examples/megatron/recipes/wan/conf/gb300_perf_pretrain_mock.yaml \
    checkpoint.save_interval=0 \
    train.global_batch_size=64 \
    dataset.global_batch_size=64 \
    train.train_iters=30

EOM
)

echo "$worker_command" > worker_command.sh
chmod 777 worker_command.sh

torchrun \
--nproc-per-node="4" \
--nnodes="8" \
--node_rank="${JOB_COMPLETION_INDEX}" \
--rdzv_id="${JOB_IDENTIFIER}" \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
--no-python bash worker_command.sh


if [[ "$JOB_COMPLETION_INDEX" == "0" ]]; then
  mkdir -p "${ARTIFACT_DIR}"
  cp -r "${explicit_log_dir}"/* "${ARTIFACT_DIR}/"
  env > "${ARTIFACT_DIR}/environ.txt"
  ls "${ARTIFACT_DIR}"
fi
echo "Training completed"
echo "Pod on $(hostname --fqdn) is exiting"