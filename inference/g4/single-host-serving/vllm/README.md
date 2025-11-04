# vLLM serving on a GCP VM with G4 GPUs

This recipe shows how to serve and benchmark open source models using [vLLM](https://github.com/vllm-project/vllm) on a single GCP VM with G4 GPUs. vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. For more information on G4 machine types, see the [GCP documentation](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-machine-types).

## Before you begin

### 1. Create a GCP VM with G4 GPUs

First, we will create a Google Cloud Platform (GCP) Virtual Machine (VM) that has the necessary GPU resources.

Make sure you have the following prerequisites:
*   [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) is initialized.
*   You have a project with a GPU quota. See [Request a quota increase](https://cloud.google.com/docs/quota/view-request#requesting_higher_quota).
*   [Enable required APIs](https://console.cloud.google.com/flows/enableapi?apiid=compute.googleapis.com).

The following commands set up environment variables and create a GCE instance. The `MACHINE_TYPE` is set to `g4-standard-48` for a single GPU VM. The boot disk is set to 200GB to accommodate the models and dependencies.

```bash
export VM_NAME="${USER}-g4-test"
export PROJECT_ID="your-project-id"
export ZONE="your-zone"
# g4-standard-48 is for a single GPU VM. For a multi-GPU VM (e.g., 8 GPUs), you can use g4-standard-384.
export MACHINE_TYPE="g4-standard-48"
export IMAGE_PROJECT="debian-cloud"
export IMAGE_FAMILY="debian-12"

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

### 3. Install the NVIDIA GPU driver and other dependencies

These commands install the necessary drivers for the GPU to work, along with other development tools. `build-essential` contains a list of packages that are considered essential for building software, and `cmake` is a tool to manage the build process.

Note: The CUDA toolkit version is specified here for reproducibility. Newer versions may be available and can be used by updating the download link.

```bash
sudo bash

# Install dependencies
apt-get update && apt-get install libevent-core-2.1-7 libevent-2.1-7 libevent-dev zip gcc make wget zip libboost-program-options-dev build-essential devscripts debhelper fakeroot -y && wget https://cmake.org/files/v3.26/cmake-3.26.0-rc1-linux-x86_64.sh && bash cmake-3.26.0-rc1-linux-x86_64.sh --skip-license

# Update linux headers
apt-get -y install linux-headers-$(uname -r)

# Download CUDA toolkit 12.9.1
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run

# Install CUDA
# Accept the EULA (type accept)
# Keep the default selections for CUDA installation (hit the down arrow, and then hit enter on "Install")
sh cuda_12.9.1_575.57.08_linux.run

exit
```
### 4. Set environment variables and check devices

We need to update the `PATH` and `LD_LIBRARY_PATH` environment variables so the system can find the CUDA executables and libraries. `HF_TOKEN` is your Hugging Face token, which is required to download some models.

```bash
# Update the PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export HF_TOKEN=<token>

# Run NVIDIA smi to verify the driver installation and see the available GPUs.
nvidia-smi
```

## Serve a model

### 1. Setup vLLM Environment

Using a conda environment is a best practice to isolate python dependencies and avoid conflicts with system-wide packages.

```bash
# Not required but adding for reproducibility
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc
conda create -n vllm python=3.11.2
source activate vllm
```

### 2. Install vLLM

Here we use `uv`, a fast python package installer. The `--extra-index-url` flag is used to point to the vLLM wheel index, and `--torch-backend=auto` will automatically select the correct torch backend.

Note: The version of `flashinfer`, `vllm` is specified for reproducibility. You can check for and install newer versions as they become available.

```bash
pip install uv
uv pip install vllm==0.10.2 --extra-index-url https://wheels.vllm.ai/0.10.2/ --torch-backend=auto
uv pip install flashinfer-python==0.3.1
uv pip install guidellm==0.3.0
```

## Run Benchmarks for Qwen3-8B-FP4

### 1. Set environment variables

These environment variables are used to enable specific features in vLLM and its backends.
- `ENABLE_NVFP4_SM120=1`: Enables NVIDIA\'s FP4 support on newer GPU architectures.
- `VLLM_ATTENTION_BACKEND=FLASHINFER`: Sets the attention backend to FlashInfer, which is a high-performance implementation. Other available backends include `XFORMERS` etc.

```bash
export ENABLE_NVFP4_SM120=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
```

### 2. Run the server

The `vllm serve` command starts the vLLM server. Here\'s a breakdown of the arguments:
- `nvidia/Qwen3-8B-FP4`: The model to be served from Hugging Face.
- `--served-model-name nvidia/Qwen3-8B-FP4`: The name to use for the model endpoint.
- `--kv-cache-dtype fp8`: Sets the data type for the key-value cache to FP8 to save GPU memory.
- `--port 8000`: The port the server will listen on.
- `--disable-log-requests`: Disables request logging for better performance.
- `--seed 42`: Sets a random seed for reproducibility.
- `--max-model-len 8192`: The maximum sequence length the model can handle.
- `--gpu-memory-utilization 0.95`: The fraction of GPU memory to be used by vLLM.
- `--tensor-parallel-size 1`: The number of GPUs to use for tensor parallelism. Since we are using a single GPU, this is set to 1. vLLM supports combination of multiple parallization strategies which can be enabled with different arguments (--data-parallel-size, --pipeline-parallel-size etc).

```bash
vllm serve nvidia/Qwen3-8B-FP4 --served-model-name nvidia/Qwen3-8B-FP4 --kv-cache-dtype fp8 --port 8000 --disable-log-requests --seed 42 --max-model-len 8192 --gpu-memory-utilization 0.95 --tensor-parallel-size 1
```

### 3. Server Output

When the server is up and running, you should see output similar to the following.

```
(APIServer pid=758221) INFO 11-03 19:48:49 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=758221) INFO:     Started server process [758221]
(APIServer pid=758221) INFO:     Waiting for application startup.
(APIServer pid=758221) INFO:     Application startup complete.
```

### 4. Run the benchmarks

To run the benchmark, you will need to interact with the server. This requires a separate terminal session. You have two options:

1.  **New Terminal**: Open a new terminal window and create a second SSH connection to your VM. You can then run the benchmark command in the new terminal while the server continues to run in the first one.
2.  **Background Process**: Run the server process in the background. To do this, append an ampersand (`&`) to the end of the `vllm serve` command. This will start the server and immediately return control of the terminal to you.

Example of running the server in the background:
```bash
vllm serve nvidia/Qwen3-8B-FP4 --served-model-name nvidia/Qwen3-8B-FP4 --kv-cache-dtype fp8 --port 8000 --disable-log-requests --seed 42 --max-model-len 8192 --gpu-memory-utilization 0.95 --tensor-parallel-size 1 &
```

Once the server is running (either in another terminal or in the background), you can run the benchmark client.

The `vllm bench serve` command is used to benchmark the running vLLM server. Here\'s a breakdown of the arguments:
- `--model nvidia/Qwen3-8B-FP4`: The model to benchmark.
- `--dataset-name random`: The dataset to use for the benchmark. `random` will generate random prompts.
- `--random-input-len 128`: The length of the random input prompts.
- `--random-output-len 2048`: The length of the generated output.
- `--request-rate inf`: The number of requests per second to send. `inf` sends requests as fast as possible.
- `--num-prompts 100`: The total number of prompts to send.
- `--ignore-eos`: A flag to ignore the end-of-sentence token and generate a fixed number of tokens.

```bash
vllm bench serve   --model nvidia/Qwen3-8B-FP4   --dataset-name random   --random-input-len 128  --random-output-len 2048  --request-rate inf  --num-prompts 100   --ignore-eos
```
### 5. Example output

The output shows various performance metrics of the model, such as throughput and latency.

```bash
============ Serving Benchmark Result ============
Successful requests:                     100
Request rate configured (RPS):           100.00
Benchmark duration (s):                  10.00
Total input tokens:                      12800
Total generated tokens:                  204800
Request throughput (req/s):              10.00
Output token throughput (tok/s):         20480.00
Total Token throughput (tok/s):          21760.00
---------------Time to First Token----------------
Mean TTFT (ms):                          100.00
Median TTFT (ms):                        99.00
P99 TTFT (ms):                           150.00
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.00
Median TPOT (ms):                        9.90
P99 TPOT (ms):                           15.00
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.00
Median ITL (ms):                         9.90
P99 ITL (ms):                            15.00
==================================================
```

## Clean up

### 1. Delete the VM

This command will delete the GCE instance and all its disks.

```bash
gcloud compute instances delete ${VM_NAME?} --zone=${ZONE?} --project=${PROJECT_ID} --quiet --delete-disks=all
```