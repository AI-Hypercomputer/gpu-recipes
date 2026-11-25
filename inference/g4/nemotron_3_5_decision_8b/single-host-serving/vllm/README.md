# Serve and benchmark NVIDIA Nemotron 3.5 Decision 8B on G4 with vLLM

This recipe serves NVIDIA Nemotron 3.5 Decision 8B (`NemotronForDecision`, 8.49B dense bidirectional Transformer) on GKE G4 (`nvidia-rtx-pro-6000`, 96 GB, SM 120) with vLLM.

Nemotron 3.5 Decision scores typed questions (`choice`, `noul`, `score`) over a state in a single bidirectional forward pass through `/v1/systemone` and `/v1/decisions`. Because the model allocates no KV cache (`is_causal: false`, `use_cache: false`) and fits on a single RTX PRO 6000 (`9.03 GiB` in FP8), deploy stateless 1-GPU replicas behind a Kubernetes Service or raw-JSON pass-through Gateway instead of KV-aware routing.

> [!NOTE]
> On GKE nodes running NVIDIA driver `580.x` (CUDA 13.0) with a CUDA 13.2+ vLLM container whose FlashAttention-2 binary lacks native `sm_120` cubins, prepend `/usr/local/cuda/compat` to `LD_LIBRARY_PATH` (default in `vllm-deployment.yaml`) or pass `--attention-backend TRITON_ATTN`.

> [!IMPORTANT]
> Configure any HTTP proxy or gateway in front of `/v1/systemone` for raw JSON pass-through. Sorting keys alphabetically inside `choice.criteria` can alter model outputs.

<a name="table-of-contents"></a>
## Table of Contents

* [1. Environment](#environment)
* [2. What the Recipe Configures](#configuration)
* [3. Environment Setup](#environment-setup)
  * [3.1. Clone the Repository](#clone-repo)
  * [3.2. Connect to your GKE Cluster](#connect-cluster)
  * [3.3. Create Hugging Face Secret (Optional)](#setup-hf-secret)
* [4. Deploy on GKE](#deploy)
* [5. Validate and Benchmark `/v1/systemone`](#benchmark)
* [6. Results](#results)
* [7. Cleanup](#cleanup)

<a name="environment"></a>
## 1. Environment

[Back to Top](#table-of-contents)

* GKE with Container-Optimized OS and NVIDIA GPU driver `580.173.02` (`latest`).
* G4 node pool (`g4-standard-48` with 1x NVIDIA RTX PRO 6000 Blackwell Server Edition, or `g4-standard-384` with 8x RTX PRO 6000 for multi-replica horizontal scaling).
* Node boot disk: provision at least `--disk-size=200` for single-replica nodes (image layers plus one `~16 GiB` checkpoint). When scaling to 8 pod-scoped `emptyDir` replicas on a single `g4-standard-384` node (`~127 GiB` cached weights plus unpacked image), provision at least `--disk-size=300` (or `--ephemeral-storage-local-ssd`) or switch `model-cache` in `vllm-deployment.yaml` to a node-shared `hostPath`.
* vLLM container with `NemotronForDecision` support (`vllm >= 0.34`, CUDA 13.2). Replace `<VLLM_IMAGE>` and `<MODEL_URI>` in [`vllm-deployment.yaml`](vllm-deployment.yaml) with the published GA container digest and model reference.

<a name="configuration"></a>
## 2. What the Recipe Configures

[Back to Top](#table-of-contents)

| File | Purpose |
| :--- | :--- |
| [`vllm-deployment.yaml`](vllm-deployment.yaml) | Single-GPU GKE Deployment and ClusterIP Service serving `/v1/systemone` and `/v1/decisions` on port `8000`. |
| [`bench_systemone.py`](bench_systemone.py) | Stdlib-only `/v1/systemone` concurrency sweep and boundary probe script. |

Key serving flags in `vllm-deployment.yaml`:

* `--hf-overrides '{"ar_mode": true, "is_causal": false}'` enables bidirectional decision mode.
* `--quantization fp8_per_tensor` with `--dtype bfloat16` cuts weight memory from `15.94 GiB` to `9.03 GiB` and improves saturated throughput on SM 120 by `+72.5%` with `<= 0.0065` score drift.
* `--max-model-len 131072 --max-num-batched-tokens 131072` prevents HTTP 400 errors on prompts longer than 8,192 tokens because bidirectional attention disables chunked prefill.
* `LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64` with `VLLM_ENABLE_CUDA_COMPATIBILITY=1` enables CUDA 13.2 forward compatibility on GKE `580.x` drivers for default `FLASH_ATTN` (`+13.8%` faster than `TRITON_ATTN` at 16K context). To avoid modifying `LD_LIBRARY_PATH`, remove those two env vars and add `--attention-backend TRITON_ATTN`.
* Horizontal scaling uses `spec.replicas` (up to 8 replicas per `g4-standard-384` node) with zero inter-GPU communication.

<a name="environment-setup"></a>
## 3. Environment Setup

[Back to Top](#table-of-contents)

<a name="clone-repo"></a>
### 3.1. Clone the Repository

```bash
git clone https://github.com/AI-Hypercomputer/gpu-recipes.git
cd gpu-recipes
export RECIPE_ROOT="$(pwd)/inference/g4/nemotron_3_5_decision_8b/single-host-serving/vllm"
```

<a name="connect-cluster"></a>
### 3.2. Connect to your GKE Cluster

```bash
export PROJECT_ID="<PROJECT_ID>"
export CLUSTER_REGION="<CLUSTER_REGION>"
export CLUSTER_NAME="<CLUSTER_NAME>"

gcloud config set project "${PROJECT_ID}"
gcloud container clusters get-credentials "${CLUSTER_NAME}" --region "${CLUSTER_REGION}"
```

<a name="setup-hf-secret"></a>
### 3.3. Create Hugging Face Secret (Optional)

If your model URI pulls from Hugging Face, create `hf-secret` to avoid anonymous download rate limits:

```bash
export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"

kubectl create secret generic hf-secret \
  --from-literal=hf_api_token="${HF_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

<a name="deploy"></a>
## 4. Deploy on GKE

[Back to Top](#table-of-contents)

1. Set `<VLLM_IMAGE>` and `<MODEL_URI>` in `vllm-deployment.yaml`, then apply the manifest:

```bash
cd "${RECIPE_ROOT}"
kubectl apply -f vllm-deployment.yaml
kubectl rollout status deployment/nemotron-3-5-decision-vllm --timeout=900s
```

2. Send a warmup and smoke request via port-forward (the first request after `/health` triggers Triton JIT compilation at `~600 ms`; subsequent requests run at `~30 ms`):

```bash
kubectl port-forward svc/nemotron-3-5-decision-vllm 8000:8000 &
PF_PID=$!
sleep 2

curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-decide",
    "state": "Customer says: I was charged twice for my order and want a refund.",
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
          "billing": "payments and refunds",
          "shipping": "delivery problems",
          "tech": "technical issues"
        }
      },
      "refund": {
        "type": "noul",
        "instructions": "Is the customer asking for a refund?"
      },
      "severity": {
        "type": "score",
        "instructions": "How severe is the problem for the customer?",
        "criteria": ["Minor annoyance", "Needs attention", "Critical"]
      }
    }
  }'

kill "${PF_PID}"
```

3. To scale across all 8 GPUs on a `g4-standard-384` node:

```bash
kubectl scale deployment/nemotron-3-5-decision-vllm --replicas=8
```

<a name="benchmark"></a>
## 5. Validate and Benchmark `/v1/systemone`

[Back to Top](#table-of-contents)

Run [`bench_systemone.py`](bench_systemone.py) from an in-cluster CPU client pod on a non-GPU system node to eliminate wide-area network RTT (if your cluster has GPU nodes only, add a `nvidia.com/gpu:NoSchedule` toleration or run via `kubectl port-forward`):

```bash
kubectl run nemotron-bench-client \
  --image=python:3.11-slim \
  --restart=Never \
  --requests='cpu=4,memory=4Gi' \
  -- sleep infinity
kubectl wait --for=condition=Ready pod/nemotron-bench-client --timeout=120s
kubectl cp "${RECIPE_ROOT}/bench_systemone.py" nemotron-bench-client:/tmp/bench_systemone.py

kubectl exec nemotron-bench-client -- python3 /tmp/bench_systemone.py \
  --base-url http://nemotron-3-5-decision-vllm:8000 \
  --model nemotron-decide \
  --questions-per-req 3 \
  --state-tokens 200 \
  --concurrencies 1,4,8,16,32,64 \
  --min-requests 120 \
  --warmup 5 \
  --boundaries
```

<a name="results"></a>
## 6. Results

[Back to Top](#table-of-contents)

### Single-GPU precision and attention comparison (5 interleaved rounds, 0 errors)

| Workload | Config | Reqs / err | `req/s` median (range) | `questions/s` median (range) | `p50` median | Delta vs `bf16_fa2` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Q=3, c=1` (`507 tok`) | `fp8_fa2` | `300 / 0` | `33.18` (`32.58-33.78`) | `99.53` (`97.74-101.33`) | `29.96 ms` | `+34.1%` |
| `Q=3, c=1` (`507 tok`) | `fp8_triton` | `300 / 0` | `33.29` (`32.98-33.38`) | `99.86` (`98.95-100.14`) | `29.87 ms` | `+34.5%` |
| `Q=3, c=1` (`507 tok`) | `bf16_fa2` | `300 / 0` | `24.75` (`24.64-24.80`) | `74.24` (`73.92-74.40`) | `40.20 ms` | Baseline |
| `Q=3, c=32` (`509 tok`) | `fp8_fa2` | `1,600 / 0` | `80.16` (`79.53-80.23`) | `240.49` (`238.60-240.69`) | `394.21 ms` | `+72.9%` |
| `Q=3, c=32` (`509 tok`) | `fp8_triton` | `1,600 / 0` | `79.05` (`78.84-79.59`) | `237.16` (`236.52-238.78`) | `399.51 ms` | `+70.5%` |
| `Q=3, c=32` (`509 tok`) | `bf16_fa2` | `1,600 / 0` | `46.36` (`46.32-46.45`) | `139.07` (`138.96-139.36`) | `685.65 ms` | Baseline |
| `Q=3, c=64` (`510 tok`) | `fp8_fa2` | `3,200 / 0` | `80.36` (`80.12-80.68`) | `241.09` (`240.37-242.04`) | `789.29 ms` | `+72.5%` |
| `Q=3, c=64` (`510 tok`) | `fp8_triton` | `3,200 / 0` | `79.82` (`79.57-79.97`) | `239.47` (`238.70-239.92`) | `795.39 ms` | `+71.3%` |
| `Q=3, c=64` (`510 tok`) | `bf16_fa2` | `3,200 / 0` | `46.59` (`46.43-46.86`) | `139.76` (`139.30-140.58`) | `1368.95 ms` | Baseline |
| `Q=1, 16K, c=4` | `fp8_fa2` | `80 / 0` | `1.24` (`1.24-1.25`) | `1.24` (`1.24-1.25`) | `3206.48 ms` | `+33.3%` |
| `Q=1, 16K, c=4` | `fp8_triton` | `80 / 0` | `1.09` (`1.09-1.10`) | `1.09` (`1.09-1.10`) | `3661.84 ms` | `+17.2%` |
| `Q=1, 16K, c=4` | `bf16_fa2` | `80 / 0` | `0.93` (`0.93-0.94`) | `0.93` (`0.93-0.94`) | `4284.49 ms` | Baseline |

### Hardware limiter and multi-replica scaling

* At `Q=3, c=64`, a 16 s `nvidia-smi dmon` probe confirms `fp8_fa2` runs at `100%` SM utilization and `597-602 W` against the `600.00 W` board TDP cap (`2,212-2,242 MHz` SM clock, `26-27%` memory controller utilization, `~611 TFLOP/s` effective dense FP8 compute).
* Long-context prompts run without OOM up to `114,459` input tokens on 96 GB VRAM (`54,689 MiB` peak); validate task-specific accuracy above `32K` tokens because midpoint needle recall degrades past `32K` under YARN scaling (`0.9724` at `32K` versus `0.4378` at `64K`).
* Two independent 1-GPU replicas (`1x fp8_fa2` + `1x fp8_triton`) running concurrently on one `g4-standard-384` node (`c=64` each, 0 errors) achieve `478.50 questions/s` (`99.6%` of their isolated single-GPU sum, `99.2%` of `2x fp8_fa2`).

<a name="cleanup"></a>
## 7. Cleanup

[Back to Top](#table-of-contents)

```bash
kubectl delete -f "${RECIPE_ROOT}/vllm-deployment.yaml" --ignore-not-found
kubectl delete pod nemotron-bench-client --ignore-not-found
kubectl delete secret hf-secret --ignore-not-found
```
