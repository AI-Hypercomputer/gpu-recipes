# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script is used to run NCCL tests on a GKE cluster with A3+ or A3 Ultra node pools.
#
set -x
set -e

usage()
{
cat << EOF
usage: bash ./nccl-test.sh [--benchmark <benchmark>] [--mask <mask>] [--begin_msg_size <begin_msg_size>] [--end_msg_size <end_msg_size>] [--factor <factor>] [--warmup_iters <warmup_iters>] [--run_iters <run_iters>]
EOF
}


parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      -t|--benchmark)
        benchmark="$2"
        shift 2
        ;;
      -m|--mask)
        mask="$2"
        shift 2
        ;;
      -g|--gpus_per_node)
        gpus_per_node="$2"
        shift 2
        ;;
      -b|--begin_msg_size)
        begin_msg_size="$2"
        shift 2
        ;;
      -e|--end_msg_size)
        end_msg_size="$2"
        shift 2
        ;;
      -f|--factor)
        factor="$2"
        shift 2
        ;;
      -w|--warmup_iters)
        warmup_iters="$2"
        shift 2
        ;;
      -n|--run_iters)
        run_iters="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit
        ;;
      *)
        echo "Invalid argument: $1"
        usage
        exit 1
        ;;
    esac
  done
}


clean_up() {
  echo "Notifying all ranks to exit"
  OMPI_MCA_orte_keep_fqdn_hostnames=t mpirun --allow-run-as-root --mca plm_rsh_no_tree_spawn true \
  --mca btl tcp,self --mca btl_tcp_if_include eth0 \
  -np ${#hostnames[@]} \
  --hostfile "/tmp/hostfiles/hostfile1" \
  bash -c "touch /tmp/master.done"
}

wait_for_tasks() {
  until [ -e /tmp/master.done ]; do
    echo "Waiting for test to complete ..."
    sleep 5
  done
  echo "Exiting pod on $(hostname --fqdn)"
  exit 0
}

configure_hostfiles() {
  local hostfile_dir=$1

  for i in $(seq 0 $(($NNODES-1))); do
    hostnames+=("$HOSTNAME_PREFIX$i.$DOMAIN_NAME")
  done
  echo "Hostnames: ${hostnames[@]}"
  sleep 30
  echo "Waiting for all hosts to be ready"
  for host in "${hostnames[@]}"; do
    echo "Waiting for $host to be ready"
    until ssh -p 222 -o StrictHostKeyChecking=no $host hostname; do
      echo Waiting for ${host}...
    done
    echo "Host $host is ready"
  done
  echo "All hosts are ready"

  echo "Generating hostfiles"
  mkdir -p $hostfile_dir
  rm -f "${hostfile_dir}/hostfile1"
  rm -f "${hostfile_dir}/hostfile8"
  touch "${hostfile_dir}/hostfile1"
  touch "${hostfile_dir}/hostfile8"
  for host in "${hostnames[@]}"; do
    echo "${host} port=222 slots=1" >> "${hostfile_dir}/hostfile1"
    echo "${host} port=222 slots=8" >> "${hostfile_dir}/hostfile8"
  done
}


run_test_ultra() {
  local benchmark=$1
  local mask=$2
  local begin_msg_size=$3
  local end_msg_size=$4
  local factor=$5
  local warmup_iters=$6
  local run_iters=$7
  local gpus_per_node=$8
  local log_path=$9
  local hostnames=( "${@:10}" )
  local nhosts="${#hostnames[@]}"
  local script_path=/usr/local/gib/scripts/set_nccl_env.sh
  local nccl_tests_dir=/third_party/nccl-tests/build

  OMPI_MCA_orte_keep_fqdn_hostnames=t \
  mpirun --allow-run-as-root \
  --mca btl tcp,self --mca btl_tcp_if_include eth0 --mca plm_rsh_no_tree_spawn true \
  --bind-to none \
  -np $(( ${gpus_per_node} * ${nhosts} )) \
  --hostfile "/tmp/hostfiles/hostfile${gpus_per_node}" \
  -x PATH -x LD_LIBRARY_PATH  \
  -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  -x NCCL_TESTS_SPLIT_MASK="${mask}" \
  bash -c \
    "source ${script_path}; \
     ${nccl_tests_dir}/${benchmark}_perf \
       -b ${begin_msg_size} -e ${end_msg_size} -f ${factor} \
       -w ${warmup_iters} -n ${run_iters}" 2>&1 |  tee ${log_path}
}


echo "Running on $(hostname --fqdn) is running"
echo "Assigned job index of $JOB_COMPLETION_INDEX"
echo "Job ID is $JOB_IDENTIFIER"
echo "Restarting ssh"
service ssh restart

export LD_LIBRARY_PATH="$NCCL_PLUGIN_PATH"
ldconfig $LD_LIBRARY_PATH
echo "Added $LD_LIBRARY_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""

echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

if [ "$JOB_COMPLETION_INDEX" -ne 0 ]; then
  wait_for_tasks
fi

benchmark="all_gather"
mask="0x0"
begin_msg_size="1K"
end_msg_size="16G"
factor="2"
warmup_iters="50"
run_iters="100"

parse_args "$@"

echo "Configuring hostfiles:"
hostfile_dir="/tmp/hostfiles"
echo "  Hostfile directory: ${hostfile_dir}"
hostnames=()
configure_hostfiles "$hostfile_dir"

nhosts="${#hostnames[@]}"
date_str=$(date "+%Y-%m-%d-%H-%M-%S")
log_file_name="${benchmark}_${nhosts}_${GPUS_PER_NODE}_${begin_msg_size}_${end_msg_size}_${factor}_${warmup_iters}_${run_iters}_${date_str}.log"
log_path="${LOG_DIR}/${log_file_name}"

mkdir -p "${LOG_DIR}"

echo "Running the test with the following parameters:"
echo "  Benchmark: $benchmark"
echo "  Mask: $mask"
echo "  Begin message size: $begin_msg_size"
echo "  End message size: $end_msg_size"
echo "  Factor: $factor"
echo "  Warmup iterations: $warmup_iters"
echo "  Run iterations: $run_iters"

args=("$benchmark" \
  "$mask" \
  "$begin_msg_size" \
  "$end_msg_size" \
  "$factor" \
  "$warmup_iters" \
  "$run_iters" \
  "$GPUS_PER_NODE" \
  "$log_path" \
  "${hostnames[@]}")

run_test_ultra "${args[@]}"

echo "Test completed"
clean_up