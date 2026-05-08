#!/bin/bash
#SBATCH --job-name=rishabhbaghel-ubench-s8tw
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0

# Exit early on failures
set -e

# Validate that the recipe location is setup correctly.
# Recipe is expected to be in "recipe" folder inside current working directory
RECIPE_DIR="$(pwd)/recipe"
LAUNCH_SCRIPT="$RECIPE_DIR/launch_script.sh"
if [[ ! -f "$LAUNCH_SCRIPT" ]]; then
    echo "Error: Recipe is not located correctly. The recipe is expected to be in "recipe" folder inside current working directory. We could not find the launch script there." >&2
    exit 1
fi
chmod +x "$LAUNCH_SCRIPT"

# Enroot the image if it is not already enrooted.
export ENROOT_CONFIG_PATH=$HOME/.config/enroot
ORIG_IMAGE=nvcr.io#nvidia/nemo:25.11
SQSH_IMAGE_PATH=/home/$USER/sqsh/nvcr.io_nvidia_nemo:25.11
if [[ ! -f "$SQSH_IMAGE_PATH" ]]; then
  mkdir -p "$(dirname "$SQSH_IMAGE_PATH")"
  echo "enrooting $ORIG_IMAGE to $SQSH_IMAGE_PATH"
  enroot import --output "$SQSH_IMAGE_PATH" -- "docker://${ORIG_IMAGE}"
fi

# sbcast the enrooted image to the worker nodes.
export WORKER_IMAGE_PATH="/dev/shm/nvcr.io_nvidia_nemo:25.11"

# get the master node
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_port=29500


ARTIFACT_DIR_HOME="/home/$USER/job_artifacts/${SLURM_JOB_ID}"
mkdir -p "$ARTIFACT_DIR_HOME"

export NNODES=$SLURM_NNODES
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export ARTIFACT_DIR=/artifacts
export JOB_NAME=rishabhbaghel-ubench-s8tw
export JOB_IDENTIFIER=rishabhbaghel-ubench-s8tw



srun --container-image="$WORKER_IMAGE_PATH" \
  --container-mounts="${RECIPE_DIR}:/recipe:mkdir,${ARTIFACT_DIR_HOME}:${ARTIFACT_DIR}:mkdir" \
  --container-workdir=/recipe \
  --container-writable \
  bash -c 'export JOB_COMPLETION_INDEX=$SLURM_NODEID; ./launch_script.sh'
  