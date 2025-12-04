# Single host inference benchmark of DeepSeek-R1 with TensorRT-LLM on G4

This recipe shows how to serve and benchmark the DeepSeek-R1 model using [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) on a single GCP VM with G4 GPUs. For more information on G4 machine types, see the [GCP documentation](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-machine-types).

## Before you begin

### 1. Create a GCP VM with G4 GPUs

First, we will create a Google Cloud Platform (GCP) Virtual Machine (VM) that has the necessary GPU resources.

Make sure you have the following prerequisites:
*   [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) is initialized.
*   You have a project with a GPU quota. See [Request a quota increase](https://cloud.google.com/docs/quota/view-request#requesting_higher_quota).
*   [Enable required APIs](https://console.cloud.google.com/flows/enableapi?apiid=compute.googleapis.com).

The following commands set up environment variables and create a GCE instance. The `MACHINE_TYPE` is set to `g4-standard-384` for a multi-GPU VM (8 GPUs). The boot disk is set to 200GB to accommodate the models and dependencies.

```bash
export VM_NAME="${USER}-g4-trtllm-deepseek-r1"
export PROJECT_ID="your-project-id"
export ZONE="your-zone"
export MACHINE_TYPE="g4-standard-384"
export IMAGE_PROJECT="ubuntu-os-accelerator-images"
export IMAGE_FAMILY="ubuntu-accelerator-2404-amd64-with-nvidia-570"

gcloud compute instances create ${VM_NAME} \
  --machine-type=${MACHINE_TYPE} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --image-project=${IMAGE_PROJECT} \
  --image-family=${IMAGE_FAMILY} \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB
```

### 2. Connect to the VM

Use `gcloud compute ssh` to connect to the newly created instance.

```bash
gcloud compute ssh ${VM_NAME?} --project=${PROJECT_ID?} --zone=${ZONE?}
```

```bash
# Run NVIDIA smi to verify the driver installation and see the available GPUs.
nvidia-smi
```

## Serve a model

### 1. Install Docker

Before you can serve the model, you need to have Docker installed on your VM. You can follow the official documentation to install Docker on Ubuntu:
[Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

After installing Docker, make sure the Docker daemon is running.

### 2. Install NVIDIA Container Toolkit

To enable Docker containers to access the GPU, you need to install the NVIDIA Container Toolkit.

You can follow the official NVIDIA documentation to install the container toolkit:
[NVIDIA Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### 3. Setup TensorRT-LLM

```bash
sudo apt-get update
sudo apt-get -y install git git-lfs

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout v1.2.0rc3
git submodule update --init --recursive
git lfs install
git lfs pull

# Build the Docker image
make -C docker release_build

# Run the Docker container
mkdir -p /scratch/cache
make -C docker release_run DOCKER_RUN_ARGS="-v /scratch:/scratch -v /scratch/cache:/root/.cache --ipc=host"
```

Now you are inside the container.

### 4. Download the Quantized Model

NVIDIA provides a pre-quantized version of DeepSeek-R1 on Hugging Face.

```bash
# Inside the container

# Download the pre-quantized model from Hugging Face
apt-get update && apt-get install -y huggingface-cli

huggingface-cli download nvidia/DeepSeek-R1-0528-NVFP4-v2 --local-dir /scratch/models/DeepSeek-R1-0528-NVFP4-v2
```

## Run Benchmarks

Create a script to run the benchmarks with different configurations.

```bash
# Inside the container

cat << 'EOF' > /scratch/run_benchmark.sh
#!/bin/bash

# Function to run benchmarks
run_benchmark() {
  local model_name=$1
  local model_path=$2
  local isl=$3
  local osl=$4
  local num_requests=$5
  local tp_size=$6
  local pp_size=$7
  local ep_size=$8

  echo "Running benchmark for $model_name with ISL=$isl, OSL=$osl, TP=$tp_size, PP=$pp_size, EP=$ep_size"

  dataset_file="/scratch/token-norm-dist_${model_name##*/}_${isl}_${osl}.json"

  python benchmarks/cpp/prepare_dataset.py --tokenizer=$model_path --stdout token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0 > $dataset_file

  # Save throughput output to a file
  trtllm-bench --model $model_name --model_path ${model_path} throughput --concurrency 128 --dataset $dataset_file --tp $tp_size --pp $pp_size --ep $ep_size --backend pytorch > "/scratch/output_${model_name##*/}_isl${isl}_osl${osl}_tp${tp_size}_pp${pp_size}_ep${ep_size}_throughput.txt"

  rm -f $dataset_file
}

model_name="deepseek-ai/DeepSeek-R1"
model_path="/scratch/models/DeepSeek-R1-0528-NVFP4-v2"
# Adjust TP/PP/EP sizes based on the number of GPUs (e.g., 8 for g4-standard-384)
TP_SIZE=8
PP_SIZE=1
EP_SIZE=1

run_benchmark "$model_name" "$model_path" 128 128 1024 $TP_SIZE $PP_SIZE $EP_SIZE
run_benchmark "$model_name" "$model_path" 128 2048 1024 $TP_SIZE $PP_SIZE $EP_SIZE
run_benchmark "$model_name" "$model_path" 128 4096 1024 $TP_SIZE $PP_SIZE $EP_SIZE
run_benchmark "$model_name" "$model_path" 500 2000 1024 $TP_SIZE $PP_SIZE $EP_SIZE
run_benchmark "$model_name" "$model_path" 1000 1000 1024 $TP_SIZE $PP_SIZE $EP_SIZE
run_benchmark "$model_name" "$model_path" 2048 128 1024 $TP_SIZE $PP_SIZE $EP_SIZE
run_benchmark "$model_name" "$model_path" 2048 2048 1024 $TP_SIZE $PP_SIZE $EP_SIZE
run_benchmark "$model_name" "$model_path" 5000 500 1024 $TP_SIZE $PP_SIZE $EP_SIZE
run_benchmark "$model_name" "$model_path" 20000 2000 1024 $TP_SIZE $PP_SIZE $EP_SIZE
EOF

chmod +x /scratch/run_benchmark.sh
/scratch/run_benchmark.sh
```

The benchmark results will be saved to files like `/scratch/output_DeepSeek-R1-0528-NVFP4-v2_isl128_osl128_tp8_pp1_ep1_throughput.txt`.

## Clean up

### 1. Exit the container

```bash
exit
```

### 2. Delete the VM

This command will delete the GCE instance and all its disks.

```bash
gcloud compute instances delete ${VM_NAME?} --zone=${ZONE?} --project=${PROJECT_ID} --quiet --delete-disks=all
```
