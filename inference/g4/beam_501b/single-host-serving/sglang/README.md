# Serve Beam 1.0 NVFP4 on G4 with SGLang, DFlash2 and HiCache

This recipe serves Reflection's [`reflection/Beam-1.0-501B-A23B-NVFP4`](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4) on one 8-GPU G4 node (`g4-standard-384`, 8× NVIDIA RTX PRO 6000 Blackwell Server Edition) in GKE. It uses the official SGLang image `lmsysorg/sglang:beam`, the [DFlash2 drafter](https://huggingface.co/reflection/Beam-1.0-501B-A23B-DFlash2) for speculative decoding, and HiCache, a pinned host-memory KV cache tier behind the GPU radix cache. A Helm chart deploys the server, and an in-cluster job benchmarks it with the SemiAnalysis AgentX multi-turn agentic coding workload (`semianalysis_cc_traces_weka_062126_256k`).

> [!NOTE]
> **Why the NVFP4 checkpoint:** the BF16 checkpoint [`reflection/Beam-1.0-501B-A23B`](https://huggingface.co/reflection/Beam-1.0-501B-A23B) is about 1.0 TB, and the FP8 checkpoint is about 503 GB. One G4 node has 768 GB of GPU memory. The NVFP4 checkpoint (about 294 GB) leaves room for a large KV cache.

> [!IMPORTANT]
> The [SGLang Beam 1.0 cookbook](https://docs.sglang.io/cookbook/autoregressive/Reflection/Beam-1_0) covers NVFP4 on B300 and GB300 only. On G4 (SM120), Beam's NVFP4 MoE layers run on the `flashinfer_cutlass` runner, which Beam's model code in the image does not allow yet. The recipe applies a one-line patch ([`sm120-beam-moe.patch`](sm120-beam-moe.patch)) at container start to allow it.

<a name="table-of-contents"></a>
## Table of Contents

* [1. Environment](#environment)
* [2. What the Recipe Configures](#configuration)
* [3. Environment Setup](#environment-setup)
  * [3.1. Clone the Repository](#clone-repo)
  * [3.2. Connect to your GKE Cluster](#connect-cluster)
  * [3.3. Create Hugging Face Kubernetes Secret (Optional)](#setup-hf-secret)
  * [3.4. Verify GPU Peer-to-Peer](#verify-p2p)
* [4. Deploy with Helm](#deploy)
* [5. Benchmark with AgentX](#benchmark)
* [6. Results](#results)
* [7. Cleanup](#cleanup)

<a name="environment"></a>
## 1. Environment

[Back to Top](#table-of-contents)

* **GKE**: version `1.35.8-gke.1439001`, Container-Optimized OS, GKE default GPU driver (`580.173.02`, which supports the image's CUDA 13.0).
* **Node**: one `g4-standard-384` (8× RTX PRO 6000 Blackwell Server Edition, 96 GB each; 384 vCPUs; about 1,380 GiB allocatable memory). GPUs communicate over PCIe with direct peer-to-peer.
* **Node boot disk**: use at least 200 GB (`--disk-size` when you create the node pool), or Local SSD-backed ephemeral storage (`--ephemeral-storage-local-ssd`). The serving image takes about 31 GiB on the node once unpacked, and about 41 GiB while it is pulled. The test node's 100 GB boot disk fit the image only when no other large image was cached on it; with one cached, the pull filled the disk and the kubelet evicted the pod.
* **Serving image**: `lmsysorg/sglang:beam@sha256:0ea912aadc3d7827858de9aa5a312c4d51c254e7dbe670dbbe8181895b92071d` (SGLang commit `f7473673dd`, CUDA 13.0.3). Pinned by digest because `sm120-beam-moe.patch` targets this build.
* **Model**: `reflection/Beam-1.0-501B-A23B-NVFP4` at revision `3bd1dccf13c8f7aa3d6732c5aaa48ba107a2c086` (293.7 GB), with drafter `reflection/Beam-1.0-501B-A23B-DFlash2` at revision `163b276be744c8ce2b8385ea85de54edb3088211` (3.9 GB). Both download from Hugging Face at server launch into a 500Gi pod-scoped volume mounted at `/ssd`.

<a name="configuration"></a>
## 2. What the Recipe Configures

[Back to Top](#table-of-contents)

| File | Purpose |
| :--- | :--- |
| [`values-hicache-dflash.yaml`](values-hicache-dflash.yaml) | Helm values: image, Hugging Face model IDs, env vars, and resources. |
| [`beam_nvfp4_hicache_dflash.yaml`](../../../../../src/frameworks/g4/sglang-configs/beam_nvfp4_hicache_dflash.yaml) | SGLang server arguments. The chart renders each key as `--<key> <value>`. |
| [`sm120-beam-moe.patch`](sm120-beam-moe.patch) | Allows the `flashinfer_cutlass` MoE runner for NVFP4 Beam experts on SM120. Applied at container start. |
| [`bench_client.yaml`](bench_client.yaml) | In-cluster AgentX benchmark job. |
| [`sglang-deployment` chart](../../../../../src/helm-charts/g4/inference-templates/sglang-deployment) | Deployment, Service, and patch ConfigMap. |

Key settings:

* **Download on launch**: at startup, SGLang downloads the checkpoint and the drafter from Hugging Face at the revisions pinned with `--revision` and `--speculative-draft-model-revision`, into `HF_HOME=/ssd`. `/ssd` is a 500Gi pod-scoped volume from the `dynamic-rwo` StorageClass (`volumes.ephemeralScratchSize` and `volumes.ephemeralStorageClass`). The volume is deleted with the pod, so every new pod downloads the weights again; container restarts within the same pod reuse them.
* **Parallelism**: TP8 with EP8; NVFP4 MoE experts run on `flashinfer_cutlass`. Attention uses `trtllm_mha` with page size 64, as in the cookbook.
* **Speculative decoding**: DFlash2 with a verify window of 8 tokens (`--speculative-dflash-block-size 8`), as in the cookbook and the drafter's model card. DFlash does not support mixed prefill/decode batches, so `--prefill-decode-interval 1` runs one decode round after every prefill batch to keep long prefills from starving decode.
* **HiCache**: `--enable-hierarchical-cache` with `--hicache-ratio 2.0` (host pool per rank = 2× the device KV pool), `--hicache-write-policy write_back`, `--hicache-mem-layout page_first`, and `--hicache-io-backend kernel`. Multi-turn prefixes evicted from GPU memory are reloaded over PCIe instead of being prefilled again. `--enable-cache-report` returns `cached_tokens`, so the benchmark can report the prefix-cache hit rate.
* **Memory**: the pod requests 1 TiB of host memory, because the pinned HiCache pool is charged to the pod. There is no memory limit: a limit would OOM-kill the pod while the pool is pinned at start-up.
* **Communication and CPU placement**: `NCCL_P2P_LEVEL=SYS` uses direct GPU peer-to-peer for the TP all-reduces. `SGLANG_SET_CPU_AFFINITY=1` pins each rank to its own 24-core slice, which also keeps each rank's host pool on its own socket.
* **API**: the model is served as `beam` on port 30080, with the `beam` reasoning and tool-call parsers.

<a name="environment-setup"></a>
## 3. Environment Setup

[Back to Top](#table-of-contents)

<a name="clone-repo"></a>
### 3.1. Clone the Repository

```bash
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=$(pwd)
export RECIPE_ROOT=$REPO_ROOT/inference/g4/beam_501b/single-host-serving/sglang
```

<a name="connect-cluster"></a>
### 3.2. Connect to your GKE Cluster

```bash
export PROJECT_ID=<PROJECT_ID>
export CLUSTER_REGION=<CLUSTER_REGION>
export CLUSTER_NAME=<CLUSTER_NAME>

gcloud config set project $PROJECT_ID
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

<a name="setup-hf-secret"></a>
### 3.3. Create Hugging Face Kubernetes Secret (Optional)

Both `reflection/Beam-1.0-501B-A23B-NVFP4` and `reflection/Beam-1.0-501B-A23B-DFlash2` are public, ungated repositories. Creating the `hf-secret` secret is optional, but recommended so the ~298 GB download at container launch is not throttled by unauthenticated Hugging Face rate limits:

```bash
export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>

kubectl create secret generic hf-secret \
  --from-literal=hf_api_token=${HF_TOKEN} \
  --dry-run=client -o yaml | kubectl apply -f -
```

<a name="verify-p2p"></a>
### 3.4. Verify GPU Peer-to-Peer

The recipe relies on direct GPU peer-to-peer (`NCCL_P2P_LEVEL=SYS`). Before you deploy, verify P2P bandwidth and data integrity between all GPU pairs on the node, including pairs that cross CPU sockets. A node with faulty P2P can produce incorrect results instead of errors. To keep the server on verified nodes, add `--set targetNodes={<NODE_NAME>}` to the `helm install` command.

<a name="deploy"></a>
## 4. Deploy with Helm

[Back to Top](#table-of-contents)

1. Install the chart:

    ```bash
    cd $RECIPE_ROOT
    helm install -f values-hicache-dflash.yaml \
      --set-file serving_config=$REPO_ROOT/src/frameworks/g4/sglang-configs/beam_nvfp4_hicache_dflash.yaml \
      --set-file patch.content=$RECIPE_ROOT/sm120-beam-moe.patch \
      beam-hicache \
      $REPO_ROOT/src/helm-charts/g4/inference-templates/sglang-deployment
    ```

    This creates a Deployment and a Service named `reflection-beam-nvfp4-g4-hicache`. To review the manifests first, run the same command with `helm template` instead of `helm install`.

2. Wait for the server to become ready. On the first launch, the pod pulls the image, downloads the model and drafter checkpoints (~298 GB) to `/ssd`, compiles JIT kernels, loads weights, captures CUDA graphs, and pins the host KV pool. The startup probe allows up to 90 minutes.

    ```bash
    kubectl rollout status deployment/reflection-beam-nvfp4-g4-hicache --timeout=90m
    kubectl logs -f deployment/reflection-beam-nvfp4-g4-hicache
    ```

3. Send a test request:

    ```bash
    kubectl port-forward svc/reflection-beam-nvfp4-g4-hicache 30080:30080 &
    curl -s http://localhost:30080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model": "beam", "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 256}'
    ```

<a name="benchmark"></a>
## 5. Benchmark with AgentX

[Back to Top](#table-of-contents)

The benchmark job runs AgentX (`agentx-harness` 1.0.7) at concurrency 16, 32, 64, and 128, for 1,800 seconds each. It downloads the tokenizer from Hugging Face, flushes the server's cache before each concurrency level, and saves the server arguments to `server_info.json` with the results.

```bash
kubectl apply -f $RECIPE_ROOT/bench_client.yaml
kubectl logs -f job/beam-agentx-bench
```

When the log prints `DONE`, copy the results within two hours:

```bash
BENCH_POD=$(kubectl get pods -l job-name=beam-agentx-bench -o jsonpath='{.items[0].metadata.name}')
kubectl cp $BENCH_POD:/results ./results/agentx-hicache-dflash
```

<a name="results"></a>
## 6. Results

[Back to Top](#table-of-contents)

> [!WARNING]
> Results for this recipe's configuration (official image, checkpoint and drafter) are pending. The reference numbers below were measured with a pre-release build of the same SGLang branch and a pre-release NVFP4 checkpoint. Its `config.json` matches the official checkpoint's, and its drafter config matches DFlash2's. Flags were the same except the served model name and the explicit attention-backend and page-size settings, which matched the values the server selected.

Pre-release reference: AgentX `semianalysis_cc_traces_weka_062126_256k`, 1,800 s per concurrency level. Requests average about 87k input tokens.

| Concurrency | Est. KV demand | TTFT p50 | ITL p50 | Prefix-cache hit rate |
| :---: | :---: | :---: | :---: | :---: |
| 64 | ~5.6M tokens | 3,232.85 ms | 83.98 ms | — |
| 128 | ~11.1M tokens | 422,370 ms | 396.74 ms | 49.9% |

At concurrency 128, the working set no longer fits in the GPU and host cache tiers. Prefixes are evicted from both and must be prefilled again, so TTFT rises to about 7 minutes.

<a name="cleanup"></a>
## 7. Cleanup

[Back to Top](#table-of-contents)

```bash
kubectl delete job beam-agentx-bench --ignore-not-found=true
helm uninstall beam-hicache
kubectl delete secret hf-secret --ignore-not-found=true
```

Uninstalling the release deletes the pod and its download volume, which removes the downloaded weights. Compiled JIT kernels stay on the node in `/var/lib/sglang-cache` (`volumes.hostCachePath`) and are reused by later pods on the same node.
