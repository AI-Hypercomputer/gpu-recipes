# Single host inference benchmark of Qwen3-32B-FP8 with vLLM on G4

This recipe shows how to serve and benchmark Qwen3-32B-FP8 model using [vLLM](https://github.com/vllm-project/vllm) on a single GCP VM with G4 GPUs. vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. For more information on G4 machine types, see the [GCP documentation](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-machine-types).

## Before you begin

### 1. Create a GCP VM with G4 GPUs

First, we will create a Google Cloud Platform (GCP) Virtual Machine (VM) that has the necessary GPU resources.

Make sure you have the following prerequisites:
*   [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) is initialized.
*   You have a project with a GPU quota. See [Request a quota increase](https://cloud.google.com/docs/quota/view-request#requesting_higher_quota).
*   [Enable required APIs](https://console.cloud.google.com/flows/enableapi?apiid=compute.googleapis.com).

The following commands set up environment variables and create a GCE instance. The `MACHINE_TYPE` is set to `g4-standard-48` for a single GPU VM, More information on different machine types can be found in the [GCP documentation](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines#g4-machine-types). The boot disk is set to 200GB to accommodate the models and dependencies.

```bash
export VM_NAME="${USER}-g4-test"
export PROJECT_ID="your-project-id"
export ZONE="your-zone"
# g4-standard-48 is for a single GPU VM. For a multi-GPU VM (e.g., 8 GPUs), you can use g4-standard-384.
export MACHINE_TYPE="g4-standard-48"
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

```
# Run NVIDIA smi to verify the driver installation and see the available GPUs.
nvidia-smi
```

## Install Dependencies

### 1. Install Docker

Before you can serve the model, you need to have Docker installed on your VM. You can follow the official documentation to install Docker on Ubuntu:
[Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

After installing Docker, make sure the Docker daemon is running.

### 2. Install NVIDIA Container Toolkit

To enable Docker containers to access the GPU, you need to install the NVIDIA Container Toolkit. This toolkit allows the container to interact with the NVIDIA driver on the host machine, making the GPU resources available within the container.

You can follow the official NVIDIA documentation to install the container toolkit:
[NVIDIA Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

##  Serve a Model

### 1. Serving on Single-Chip (1-GPU):

To run the vLLM server, you can use the following command:

```bash
sudo docker run \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    --entrypoint python3 \
    vllm/vllm-openai:latest \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B-FP8 \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 256 \
    --max-model-len 8192 \ 
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1 \ 
```
For the 32B model on a G4 (1-chip) instance, we recommend --max-num-batched-tokens 4096 --max-num-seqs 256 --max-model-len 8192.

Here's a breakdown of the arguments:
-   `--runtime nvidia --gpus all`: This makes the NVIDIA GPUs available inside the container.
-   `-v ~/.cache/huggingface:/root/.cache/huggingface`: This mounts the Hugging Face cache directory from the host to the container. This is useful for cachingdownloaded models.
-   `--env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN"`: This sets the Hugging Face Hub token as an environment variable in the container. This is required for downloading models that require authentication.
-   `-p 8000:8000`: This maps port 8000 on the host to port 8000 in the container.
-   `--ipc=host`: This allows the container to share the host's IPC namespace, which can improve performance.
-   `vllm/vllm-openai:latest`: This is the name of the official vLLM docker image.
-   `--model Qwen/Qwen3-32B-FP8`: The model to be served from Hugging Face.
-   `--kv-cache-dtype fp8`: Sets the data type for the key-value cache to FP8 to save GPU memory.
-   `--max-num-batched-tokens 4096`: This sets the maximum total tokens (input+output) the GPU processes in one go. It limits the GPU's immediate computational load.
-   `--max-num-seqs 256`: This sets the maximum number of concurrent requests (sequences) the VLLM scheduler keeps actively running in the GPU's KV cache. 
-   `--max-model-len 8192`: This defines the maximum total context size (I/tokens+O/tokens) allowed for any single request. It sets the model's context window limit.
-   `--gpu-memory-utilization 0.95`: The fraction of GPU memory to be used by vLLM.
-   `--tensor-parallel-size 1`: It specifies the number of gpu's to use.

### 2. Serving on Multi-Chip (8-GPU): 

The system will use the full 8-GPU capacity, which requires explicitly exporting the NCCL peer-to-peer ([P2P documentation](https://cloud.google.com/blog/products/compute/g4-vms-p2p-fabric-boosts-multi-gpu-workloads/)) setting.



```
The NCCL_P2P_LEVEL setting forces NVIDIA's communication library to utilize peer-to-peer transfers across the entire multi-GPU system, 
including those linked through the CPU/NUMA interconnects, which is crucial for accelerating multi-GPU workloads like those on Google Cloud's G4 VMs.
```
```
export NCCL_P2P_LEVEL=SYS
```

we will use the official vllm/vllm-openai:latest Docker image to run the server. 

```bash
sudo docker run \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-32B-FP8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 8 \
```

Here's a breakdown of the arguments:
-   `gpus all`: Exposes all 8 available GPUs to the Docker container.
-   `tensor-parallel-size 8`: This is the crucial setting, enabling Tensor Parallelism to split the 70B model weights and operations across all 8 GPUs.

For more information on the available engine arguments, you can refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/configuration/engine_args/), which includes different parallelism strategies that can be used with multi GPU setup.

After running the command, the model will be served. To run the benchmark, you will need to either run the server in the background by appending `&` to the command, or open a new terminal to run the benchmark command.

## Run Benchmarks for Qwen3-32B-FP8

### 1. Server Output

When the server is up and running, you should see output similar to the following.

```
(APIServer pid=XXXXXX) INFO XX-XX XX:XX:XX [launcher.py:XX] Route: /metrics, Methods: GET
(APIServer pid=XXXXXX) INFO:     Started server process [XXXXXX]
(APIServer pid=XXXXXX) INFO:     Waiting for application startup.
(APIServer pid=XXXXXX) INFO:     Application startup complete.
```

### 2. Run the benchmarks

To run the benchmark, you can use the following command:

```bash
sudo docker run \
    --runtime nvidia \
    --gpus all \
    --network="host" \
    --entrypoint vllm \
    vllm/vllm-openai:latest bench serve \
    --model Qwen/Qwen3-32B-FP8 \
    --dataset-name random \
    --random-input-len 2048 \
    --random-output-len 128 \
    --request-rate inf \
    --num-prompts 1000 \
    --ignore-eos
```
Here's a breakdown of the arguments:
- `--model Qwen/Qwen3-32B-FP8`: The model to benchmark.
- `--dataset-name random`: The dataset to use for the benchmark. `random` will generate random prompts.
- `--random-input-len 2048`: The length of the random input prompts.
- `--random-output-len 128`: The length of the generated output.
- `--request-rate inf`: The number of requests per second to send. `inf` sends requests as fast as possible.
- `--num-prompts 1000`: The total number of prompts to send.
- `--ignore-eos`: A flag to ignore the end-of-sequence token and generate a fixed number of tokens.

### 3. Example output

The output shows various performance metrics of the model, such as throughput and latency.

```bash
============ Serving Benchmark Result ============ 
Successful requests:                     XX
Request rate configured (RPS):           XX
Benchmark duration (s):                  XX
Total input tokens:                      XX
Total generated tokens:                  XX
Request throughput (req/s):              XX
Output token throughput (tok/s):         XX
Total Token throughput (tok/s):          XX
---------------Time to First Token----------------
Mean TTFT (ms):                          XX
Median TTFT (ms):                        XX
P99 TTFT (ms):                           XX
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          XX
Median TPOT (ms):                        XX
P99 TPOT (ms):                           XX
---------------Inter-token Latency----------------
Mean ITL (ms):                           XX
Median ITL (ms):                         XX
P99 ITL (ms):                            XX
==================================================
```

## Clean up

### 1. Delete the VM

This command will delete the GCE instance and all its disks.

```bash
gcloud compute instances delete ${VM_NAME?} --zone=${ZONE?} --project=${PROJECT_ID} --quiet --delete-disks=all
```
