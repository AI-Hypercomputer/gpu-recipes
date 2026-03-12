# Single host inference benchmark of Wan2.2 with Sglang on G4

This recipe shows how to serve and benchmark the Wan-AI/Wan2.2-T2V-A14B & Wan-AI/Wan2.2-I2V-A14B model using [SGLang](https://github.com/sgl-project/sglang/tree/main) on a single GCP VM with G4 GPUs. For more information on G4 machine types, see the [GCP documentation](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-machine-types).

## Before you begin

### 1. Create a GCP VM with G4 GPUs

First, we will create a Google Cloud Platform (GCP) Virtual Machine (VM) that has the necessary GPU resources.

Make sure you have the following prerequisites:
*   [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) is initialized.
*   You have a project with a GPU quota. See [Request a quota increase](https://cloud.google.com/docs/quota/view-request#requesting_higher_quota).
*   [Enable required APIs](https://console.cloud.google.com/flows/enableapi?apiid=compute.googleapis.com).

The following commands set up environment variables and create a GCE instance. The `MACHINE_TYPE` is set to `g4-standard-384` for a multi-GPU VM (8 GPUs). The boot disk is set to 200GB to accommodate the models and dependencies.

```bash
export VM_NAME="${USER}-g4-sglang-wan2.2"
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

### 3. Setup Sglang

```bash
sudo apt-get update
sudo apt-get -y install git git-lfs

git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt

export IMAGE_URL="us-docker.pkg.dev/gke-ai-infra/gpu-recipes/sglang:latest"

# Run the Docker container
mkdir -p /scratch/cache
docker run --gpus all -it --rm \
  -v /scratch:/scratch \
  -v /scratch/cache:/root/.cache \
  --ipc=host \
  $IMAGE_URL \
  /bin/bash
```

Now you are inside the container.

### 4. Download the Model Weights

```bash
# Inside the container

# Download the base model from Hugging Face
apt-get update && apt-get install -y huggingface-cli

huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir /scratch/models/Wan2.2
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir /scratch/models/Wan2.2

```

## Run Benchmarks

Run the benchmarks with different configurations of both Text to Video & Image to Video generations.

```bash
# Inside the container
#Number of gpus-1 & frames-81
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers  --dit-layerwise-offload false --text-encoder-cpu-offload false --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false     --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." --save-output --num-gpus 1 --num-frames 81
```
```bash
#Number of gpus-4 & frames-93
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers  --dit-layerwise-offload false --text-encoder-cpu-offload false --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false     --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." --save-output --num-gpus 4  --tp-size 4 --num-frames 93
```
```bash
#Number of gpus-1 & frames-81
#Download Image from internet
sglang generate --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers --image-path assets/logo.png --dit-layerwise-offload false --text-encoder-cpu-offload false --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false --prompt "A curious raccoon" --save-output --num-gpus 1 --num-frames 81
```
```bash
#Number of gpus-3 & frames-93
sglang generate --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers --image-path assets/logo.png --dit-layerwise-offload false --text-encoder-cpu-offload false --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false --prompt "A curious raccoon" --save-output --num-gpus 4 --tp-size 4 --num-frames 93
```

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
