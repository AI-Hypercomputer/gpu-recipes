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
export NCCL_IB_SPLIT_DATA_ON_QPS=1
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

  echo "Collect nvidia-smi telemetry"
  gpu_uuid=\$(nvidia-smi --id \$LOCAL_RANK --query-gpu uuid --format noheader)

  echo "NODE_NAME=${NODE_NAME}"
  echo "Rank \$RANK starting on GPU \$gpu_uuid"

  echo "Runtime logging: "
  echo "LOCAL_RANK: \$LOCAL_RANK"
  echo "RANK: \$RANK"

  telemetry_dir="/runtime-logs/${JOBSET_NAME}/nvidia-smi"
  mkdir -p \$telemetry_dir
  touch "\$telemetry_dir/\$RANK.csv"
  #touch "\$telemetry_dir/\$RANK_custom.csv"
  echo "Telemetry dir: \$telemetry_dir"
  ls "\$telemetry_dir"

  nv_smi_pid=\$!

  # power_dir="/runtime-logs/${JOBSET_NAME}/power"
  # mkdir -p \$power_dir
  # touch "\$power_dir/\$RANK.csv"
  # nvidia-smi --id \$gpu_uuid --query-gpu=index,timestamp,power.draw.average,power.draw.instant,module.power.draw.average,module.power.draw.instant --format=csv,noheader,nounits -lms 100 -f \$power_dir/\$RANK.csv &
  # nv_power_pid=\$!

  #cp /runtime-logs/custom_log_telemetry.sh .
  #chmod +x ./custom_log_telemetry.sh
  #./custom_log_telemetry.sh >> "\$telemetry_dir/\$RANK_custom.csv" &
  #custom_log_telemetry_pid=\$!

  if [ "\$LOCAL_RANK" -eq "0" ] || [ "\$LOCAL_RANK" -eq "2" ]; then
    # get CPU usage
    echo "Starting CPU usage capture"
    cpu_usage_dir="/runtime-logs/${JOBSET_NAME}/cpu-usage"
    mkdir -p \$cpu_usage_dir
    touch "\$cpu_usage_dir/cpu_usage_\$RANK.out"

    top -b -d 5 >> \$cpu_usage_dir/cpu_usage_\$RANK.out &
    export cpu_usage_pid=\$!

    # get NIC usage
    echo "Starting NIC usage capture"
    cp /runtime-logs/custom_log_nic.sh .
    chmod +x ./custom_log_nic.sh

    nic_usage_dir="/runtime-logs/${JOBSET_NAME}/nic-usage"
    mkdir -p "\$nic_usage_dir"
    touch "\$nic_usage_dir/nic_log_eth0_\$RANK.log"

    ./custom_log_nic.sh "eth0" >> "\$nic_usage_dir/nic_log_eth0_\$RANK.log" &
    nic_usage_pid_eth0=\$!
    echo "NIC RUN ID: \$nic_usage_pid_eth0"
  fi

  cd /opt/Megatron-Bridge ;

  numactl \
    --cpunodebind=\$((LOCAL_RANK/2)) \
    --membind=\$((LOCAL_RANK/2)) \
  nice -10 \
  python scripts/performance/run_script.py \
    --model_family_name kimi \
    --model_recipe_name kimi_k2 \
    --gpu gb300 \
    --num_gpus 512 \
    --gpus_per_node 4 \
    --compute_dtype fp8_mx \
    --seq_length 4096 \
    --global_batch_size 8192 \
    --micro_batch_size 2 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 4 \
    --virtual_pipeline_model_parallel_size 4 \
    --context_parallel_size 1 \
    --expert_model_parallel_size 64 \
    --expert_tensor_parallel_size 1 \
    --moe_flex_dispatcher_backend hybridep \
    --recompute_modules mla_up_proj \
    --max_step 50 \
    logger.log_throughput=True \
    train.manual_gc_interval=100


  kill -9 \$nv_smi_pid
  # kill -9 \$nv_power_pid
  #kill -9 \$custom_log_telemetry_pid

  if [ "\$LOCAL_RANK" -eq "0" ] || [ "\$LOCAL_RANK" -eq "2" ]; then
    kill -9 \$cpu_usage_pid
    kill -9 \$nic_usage_pid_eth0
  fi
  echo "Telemetry processes killed"
  ps aux
EOM
)

echo "$worker_command" > worker_command.sh
chmod 777 worker_command.sh

mkdir -p "/runtime-logs/${JOBSET_NAME}/logs"

torchrun \
--nproc-per-node="4" \
--nnodes="128" \
--node_rank="${JOB_COMPLETION_INDEX}" \
--rdzv_id="${JOB_IDENTIFIER}" \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
--no-python bash worker_command.sh 2>&1 | python3 -u -c "import sys, time; [sys.stdout.write('[{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), line)) for line in iter(sys.stdin.readline, '')]" | tee "/runtime-logs/${JOBSET_NAME}/logs/nemo_mb_RANK_${JOB_COMPLETION_INDEX}.log"


if [[ "$JOB_COMPLETION_INDEX" == "0" ]]; then
  mkdir -p "${ARTIFACT_DIR}"
  cp -r "${explicit_log_dir}"/* "${ARTIFACT_DIR}/"
  env > "${ARTIFACT_DIR}/environ.txt"
  ls "${ARTIFACT_DIR}"
fi
echo "Training completed"
echo "Pod on $(hostname --fqdn) is exiting"
