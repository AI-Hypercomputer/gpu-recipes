#!/bin/bash
#SBATCH --job-name=megatron-pretrain-qwen3-235b-a22b
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0

# Exit early on failures
set -e

# Validate that the recipe location is setup correctly.
# Recipe is expected to be in "recipe" folder inside current working directory
RECIPE_DIR="$(pwd)/recipe"
LAUNCH_SCRIPT="${RECIPE_DIR}/launch_script.sh"
if [[ ! -f "${LAUNCH_SCRIPT}" ]]; then
    echo "Error: Recipe is not located correctly. The recipe is expected to be in "recipe" folder inside current working directory. We could not find the launch script there." >&2
    exit 1
fi
chmod +x "${LAUNCH_SCRIPT}"

# Enroot the image if it is not already enrooted.
export ENROOT_CONFIG_PATH=${HOME}/.config/enroot
ORIG_IMAGE=nvcr.io#nvidia/nemo:25.11
SQSH_IMAGE_PATH=${RECIPE_DIR}/sqsh/nvcr.io_nvidia_nemo:25.11
if [[ ! -f "${SQSH_IMAGE_PATH}" ]]; then
  mkdir -p "$(dirname "${SQSH_IMAGE_PATH}")"
  echo "enrooting $ORIG_IMAGE to ${SQSH_IMAGE_PATH}"
  enroot import --output "${SQSH_IMAGE_PATH}" -- "docker://${ORIG_IMAGE}"
fi

# get the master node
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_port=29500

ARTIFACT_DIR_HOME="/home/$USER/job_artifacts/${SLURM_JOB_ID}"
mkdir -p "$ARTIFACT_DIR_HOME"

export NNODES=$SLURM_NNODES
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export ARTIFACT_DIR=/artifacts
export JOB_NAME=megatron-pretrain-qwen3-235b-a22b
export JOB_IDENTIFIER=megatron-pretrain-qwen3-235b-a22b
export CUSTOM_SETUP_EXPERIMENT_SCRIPT_PATH=/recipe/custom_setup_experiment.py 


export PMIX_MCA_gds="^ds12"
export GLOO_SOCKET_IFNAME=enp0s3

srun --container-image="$SQSH_IMAGE_PATH" \
  --container-mounts="${RECIPE_DIR}:/recipe:mkdir,${ARTIFACT_DIR_HOME}:${ARTIFACT_DIR}:mkdir,/usr/local/gib:/usr/local/gib" \
  --container-workdir=/recipe \
  --container-writable \
  bash -c 'export JOB_COMPLETION_INDEX=$SLURM_NODEID; ./launch_script.sh'
