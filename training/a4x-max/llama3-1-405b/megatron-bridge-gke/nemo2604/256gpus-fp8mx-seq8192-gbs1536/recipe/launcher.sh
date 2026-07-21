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

# Create the nsys directory.
mkdir -p "${explicit_log_dir}/nsys"

# Collect diagnostics
linux_kv="$(uname --kernel-release)"
cuda_driver_v=""
driver_v=""
vbios_v=""
if command -v nvidia-smi &> /dev/null; then
  cuda_driver_v=$(nvidia-smi -q -x | grep -Po '(?<=<cuda_version>).*(?=</cuda_version>)' || true)
  driver_v=$(nvidia-smi -q -x | grep -Po '(?<=<driver_version>).*(?=</driver_version>)' || true)
  vbios_v=$(nvidia-smi -q -x | grep -Po '(?<=<vbios_version>).*(?=</vbios_version>)' | head -n1 || true)
fi
nccl_v=$(python3 -c "import torch; v=torch.cuda.nccl.version() if hasattr(torch.cuda, 'nccl') else 'unknown'; print('.'.join(map(str, v)) if isinstance(v, tuple) else v)" || echo "unknown")
cuda_container_v=$(python3 -c "import torch; print(torch.version.cuda)" || echo "unknown")

kv="{\"linux_kernel_version\": \"${linux_kv}\""
kv="${kv}, \"cuda_driver_version\": \"${cuda_driver_v}\""
kv="${kv}, \"cuda_container_version\": \"${cuda_container_v}\""
kv="${kv}, \"gpu_driver_version\": \"${driver_v}\""
kv="${kv}, \"vbios_version\": \"${vbios_v}\""
kv="${kv}, \"nccl_version\": \"${nccl_v}\"}"

echo "VERSION_DIAGNOSTICS: ${kv}"


export HF_TOKEN=<YOUR_HF_TOKEN>
export NCCL_RAS_ENABLE=0

cd /opt
rm -rf Megatron-Bridge
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge
git checkout c810129341a84e58f4cbed3093f70668a088c028
git submodule update --init --recursive && sed -i 's/timeout=60/timeout=600/g' src/megatron/bridge/models/hf_pretrained/safe_config_loader.py
sed -i -e '/pretrain(config=recipe/i \    recipe.dist.distributed_timeout_minutes = 10' scripts/performance/run_script.py
ls



worker_command=$(cat <<- EOM
  if [ "\$RANK" -eq "0" ]; then
    echo "Worker 0 is stalling for a few seconds.." ;
    sleep 3 ;
    echo "The detected environment within worker rank 0 is:" ;
    env | sed 's/^/  /' ;
  else
    echo "Worker \$RANK is running" ;
  fi ;

  cd /opt/Megatron-Bridge ;

  numactl \
    --cpunodebind=\$((LOCAL_RANK/2)) \
    --membind=\$((LOCAL_RANK/2)) \
  nice -10 \
  python scripts/performance/run_script.py \
    --model_family_name llama \
    --model_recipe_name llama31_405b \
    --config_variant v2 \
    --gpu gb300 \
    --num_gpus 256 \
    --gpus_per_node 4 \
    --compute_dtype fp8_mx \
    --seq_length 8192 \
    --global_batch_size 1536 \
    --micro_batch_size 1 \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 8 \
    --virtual_pipeline_model_parallel_size 4 \
    --context_parallel_size 2 \
    --expert_model_parallel_size 1 \
    --expert_tensor_parallel_size 1 \
    --max_step 50 \
    logger.log_throughput=True
EOM
)

echo "$worker_command" > worker_command.sh
chmod 777 worker_command.sh

torchrun \
--nproc-per-node="4" \
--nnodes="64" \
--node_rank="${JOB_COMPLETION_INDEX}" \
--rdzv_id="${JOB_IDENTIFIER}" \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
--no-python bash worker_command.sh 2>&1 | python3 -u -c "import sys, time; [sys.stdout.write('[{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), line)) for line in iter(sys.stdin.readline, '')]" 


if [[ "$JOB_COMPLETION_INDEX" == "0" ]]; then
  mkdir -p "${ARTIFACT_DIR}"
  cp -r "${explicit_log_dir}"/* "${ARTIFACT_DIR}/"
  env > "${ARTIFACT_DIR}/environ.txt"
  ls "${ARTIFACT_DIR}"
fi
echo "Training completed"
echo "Pod on $(hostname --fqdn) is exiting"
