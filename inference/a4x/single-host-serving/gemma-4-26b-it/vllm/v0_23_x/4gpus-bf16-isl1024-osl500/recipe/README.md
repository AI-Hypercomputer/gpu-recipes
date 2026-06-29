# Gemma-4 GEMMA-4-26B-IT BF16 TEXT Recipe (A4X)

This recipe configures and serves the **Gemma-4 GEMMA-4-26B-IT** model in **BF16** precision for the **TEXT** workload on a single A4X node (4x GB200 GPUs) using vLLM V1.

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
  $USER-serving-gemma4-gemma-4-26b-it-bf16-text \
  $REPO_ROOT/src/helm-charts/a4x/inference-templates/deployment
```
