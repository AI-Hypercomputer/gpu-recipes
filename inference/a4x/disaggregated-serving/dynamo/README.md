# Disaggregated Multi-Node Dynamo Recipe for A4x

This recipe runs a disaggregated multi-node Dynamo deployment on A4x.

## Setup

1.  **Set Environment Variables**

    ```bash
    export REPO_ROOT=$(git rev-parse --show-toplevel)
    export RELEASE_VERSION="24.05"
    export USER=$(whoami)
    ```

2.  **Run the Recipe**

    ```bash
    helm install -f values.yaml \
      --set-file workload_launcher=$REPO_ROOT/src/launchers/dynamo-vllm-launcher.sh \
      --set-file serving_config=$REPO_ROOT/src/frameworks/a4x/dynamo-configs/llama-3.3-70b-multi-node.yaml \
      --set workload.framework=vllm \
      --set workload.model.name=meta-llama/Llama-3.3-70B-Instruct \
      --set workload.image=nvcr.io/nvidia/ai-dynamo/vllm-runtime:${RELEASE_VERSION} \
      --set workload.gpus=16 \
      $USER-dynamo-multi-node-serving-a4x \
      $REPO_ROOT/src/helm-charts/a4x/inference-templates/dynamo-deployment
    ```

