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
# This script is used to run MaxText workloads.
#


usage()
{
cat << EOF
usage: bash ./maxtext-launcher.sh [config-override  [config-override ...]]
config-override  (Optional) A  MaxText configuration override. E.g. base_output_directory=gs://log-bucket/maxtext steps=10.
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

echo "MaxText configuration file:"
cat "${MAXTEXT_CONFIG_PATH}" | sed 's/^/| /'
echo ""


if [ -z "${config_overrides}" ]; then
  echo "No MaxText config overrides specified"
else
  echo "MaxText config overrides:"
  echo "  ${config_overrides}"
fi

export LD_LIBRARY_PATH="$NCCL_PLUGIN_PATH"
ldconfig $LD_LIBRARY_PATH
echo "Added $LD_LIBRARY_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""

export JAX_COORDINATOR_IP=$(nslookup "$MASTER_ADDR" 2>/dev/null | awk '/^Address: / { print $2 }' | head -n 1)
echo "JAX_COORDINATOR_IP: $JAX_COORDINATOR_IP"
export JAX_COORDINATOR_PORT=$MASTER_PORT
echo "JAX_COORDINATOR_PORT: $JAX_COORDINATOR_PORT"
export NODE_RANK=$JOB_COMPLETION_INDEX
export JAX_COORDINATOR_ADDRESS="$MASTER_ADDR"
echo "JAX_COORDINATOR_ADDRESS: $JAX_COORDINATOR_ADDRESS"
echo "XLA_FLAGS: $XLA_FLAGS"

python MaxText/train.py ${MAXTEXT_CONFIG_PATH}  ${config_overrides} run_name=${JOB_IDENTIFIER}

echo "Training completed"
echo "Pod on $(hostname --fqdn) is exiting"