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

echo "Pod on $(hostname --fqdn) is running"
echo "Pod is assigned job index of $JOB_COMPLETION_INDEX"
echo "Job ID is $JOB_IDENTIFIER"
echo "Restarting ssh"
service ssh restart

echo "NCCL_LIB_DIR: ${NCCL_LIB_DIR}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

if [ "$JOB_COMPLETION_INDEX" -ne 0 ]; then
  wait_for_tasks
fi

echo "Configuring hostfiles:"
hostfile_dir="/tmp/hostfiles"
echo "  Hostfile directory: ${hostfile_dir}"
hostnames=()
configure_hostfiles "$hostfile_dir"

nhosts="${#hostnames[@]}"
date_str=$(date "+%Y-%m-%d-%H-%M-%S")
log_file_name="${BENCHMARK}_${nhosts}_${GPUS_PER_NODE}_${BEGIN_MESSAGE_SIZE}_${END_MESSAGE_SIZE}_${FACTOR}_${WARMUP_ITERATIONS}_${RUN_ITERATIONS}_${date_str}.log"
log_path="${LOG_DIR}/${log_file_name}"

echo "Running the test with the following parameters:"
echo "  Benchmark: $BENCHMARK"
echo "  Mask: $MASK"
echo "  Begin message size: $BEGIN_MESSAGE_SIZE"
echo "  End message size: $END_MESSAGE_SIZE"
echo "  Factor: $FACTOR"
echo "  Warmup iterations: $WARMUP_ITERATIONS"
echo "  Run iterations: $RUN_ITERATIONS"
echo "  Hosts: ${hostnames[@]}"
echo "  Log path: ${GCS_LOG_DIR}/${log_file_name}"

mkdir -p "${LOG_DIR}"

args=("$BENCHMARK" \
  "$MASK" \
  "$BEGIN_MESSAGE_SIZE" \
  "$END_MESSAGE_SIZE" \
  "$FACTOR" \
  "$WARMUP_ITERATIONS" \
  "$RUN_ITERATIONS" \
  "$GPUS_PER_NODE" \
  "$log_path" \
  "${hostnames[@]}")

run_test_ultra "${args[@]}"

echo "Test completed"
clean_up