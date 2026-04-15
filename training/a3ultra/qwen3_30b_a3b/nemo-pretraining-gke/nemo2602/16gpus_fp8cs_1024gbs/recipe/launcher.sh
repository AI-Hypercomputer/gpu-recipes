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


export HF_TOKEN=${HF_TOKEN:-<YOUR_HF_TOKEN>}

cd /opt
rm -rf Megatron-Bridge
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge
git checkout f7a9428f301fa17ac374d5e7166a63b0aa4771af
git submodule update --init --recursive
sed -i -e '/pretrain(config=recipe/i \    recipe.dist.distributed_timeout_minutes = 10' scripts/performance/run_script.py
ls

cp $CUSTOM_SETUP_EXPERIMENT_SCRIPT_PATH scripts/performance/

worker_command=$(cat <<- EOM
  if [ "\$RANK" -eq "0" ]; then
    echo "Worker 0 is stalling for a few seconds.." ;
    sleep 3 ;
    echo "The detected environment within worker rank 0 is:" ;
    env | sed 's/^/  /' ;
  fi ;

  cd /opt/Megatron-Bridge ;

  numactl \
    --cpunodebind=\$((LOCAL_RANK/4)) \
    --membind=\$((LOCAL_RANK/4))           nsys profile \
    -t nvtx,cuda \
    --cuda-event-trace=false \
    --sample=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --kill none \
    -o "${explicit_log_dir}/$JOB_IDENTIFIER/rank-\$RANK" \
    --force-overwrite true \
    --session-new "nsys-\$RANDOM-\$RANK" \
  nice -10 \
  python scripts/performance/custom_setup_experiment.py \
    --model_family_name qwen \
    --model_recipe_name qwen3_30b_a3b \
    --config_variant v1 \
    --gpu h100 \
    --num_gpus 16 \
    --gpus_per_node 8 \
    --compute_dtype fp8_cs \
    --seq_length 4096 \
    --global_batch_size 1024 \
    --micro_batch_size 4 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --virtual_pipeline_model_parallel_size 12 \
    --context_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --expert_tensor_parallel_size 1 \
    --moe_a2a_overlap True \
    --max_steps 30

EOM
)

echo "$worker_command" > worker_command.sh
chmod 777 worker_command.sh

torchrun \
--nproc-per-node="8" \
--nnodes="2" \
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