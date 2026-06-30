# Gemma-4 GEMMA-4-26B-IT BF16 MM4 Recipe (A4X)

This recipe configures and serves the **Gemma-4 GEMMA-4-26B-IT** model in **BF16** precision for the **MM4** workload on a single A4X node (4x GB200 GPUs) using vLLM V1.

## How to run

To deploy this model, follow the generic vLLM serving instructions in the main vLLM recipe:
[Single Host Model Serving with vLLM on A4X](../../../../../vllm/README.md)

When running the `helm install` command (Section 4.1.1 in the main guide), make sure to use the `values.yaml` and `serving-args.yaml` from this directory:

```bash
helm install -f values.yaml \
  --set-file workload_launcher=$REPO_ROOT/src/launchers/vllm-launcher.sh \
  --set-file serving_config=serving-args.yaml \
  --set queue=${KUEUE_NAME} \
  --set "volumes.gcsMounts[0].bucketName=${GCS_BUCKET}" \
  --set workload.image=${ARTIFACT_REGISTRY}/${VLLM_IMAGE}:vllm${VLLM_VERSION}-ngcvllm${NGC_VLLM_VERSION} \
  $USER-serving-gemma4-gemma-4-26b-it-bf16-mm4 \
  $REPO_ROOT/src/helm-charts/a4x/inference-templates/deployment
```

## Benchmarking

To benchmark the deployed model, run the following command inside the serving container (after the deployment is ready):

```bash
kubectl exec -it deployment/$USER-serving-gemma4-gemma-4-26b-it-bf16-mm4 -c serving -- \
  vllm bench serve \
  --model google/gemma-4-26B-A4B-it \
  --dataset-name random-mm \
  --ignore-eos \
  --num-prompts 512 \
  --random-input-len 1024 \
  --random-output-len 500 \
  --random-mm-bucket-config '{(512, 512, 1): 1.0}' \
  --random-mm-limit-mm-per-prompt '{"image": 4}' \
  --random-mm-base-items-per-request 4 \
  --max-concurrency 256 \
  --port 8000 \
  --backend openai-chat \
  --endpoint /v1/chat/completions
```
