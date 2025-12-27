# Disaggregated Multi-Node Dynamo Recipe for A4x

This recipe runs a disaggregated multi-node Dynamo deployment on A4X.

## Setup

1.  **Set Environment Variables**

    ```bash
    export USER=$(whoami)
    ```

2.  **Run the Recipe**

  ```bash
  cd $RECIPE_ROOT
  helm install -f values.yaml \
  --set-file workload_launcher=$REPO_ROOT/src/launchers/dynamo-sglang-launcher.sh \
  --set-file prefill_serving_config=$REPO_ROOT/src/frameworks/a4x/dynamo-configs/deepseekr1-fp8-multi-node-prefill.yaml \
  --set-file decode_serving_config=$REPO_ROOT/src/frameworks/a4x/dynamo-configs/deepseekr1-fp8-multi-node-decode.yaml \
  $USER-dynamo-a4x-multi-node \
  $REPO_ROOT/src/helm-charts/a4x/inference-templates/dynamo-deployment
  ```

