# Run NCCL Tests on A3 Ultra GKE node pools

This recipe outlines the steps for running [NVIDIA NCCL tests](https://github.com/NVIDIA/nccl-tests) on Google Kubernetes Engine (GKE) clusters with A3 Ultra node pools.

## Prerequisites

Before running a NCCL test, ensure your environment is configured as follows:

- A GKE cluster with an A3 Ultra node pool.
- A Google Cloud Storage (GCS) bucket to store results.
- A client workstation with the following pre-installed:
  - Google Cloud SDK
  - Helm
  - kubectl

To prepare the required environment, see
[GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-ultra.md).

## Run the recipe

From your client workstation, complete the following steps:

### Configure environment settings

Set the environment variables to match your environment:

 ```bash
 export PROJECT_ID=<PROJECT_ID>
 export CLUSTER_NAME=<CLUSTER_NAME>
 export CLUSTER_LOCATION=<CLUSTER_LOCATION>
 export GCS_BUCKET=<GCS_BUCKET>
 export KUEUE_NAME=<KUEUE_NAME>
 ```

 Replace the following values:

- `<PROJECT_ID>`: your Google Cloud project ID.
- `<CLUSTER_REGION>`: the region where your cluster is located.
- `<CLUSTER_NAME>`: the name of your GKE cluster.
- `<GCS_BUCKET>`: the name of your Cloud Storage bucket. Don't include the `gs://` prefix.
- `<KUEUE_NAME>`: the name of the Kueue local queue configured for TAS.  The default queue created by the cluster toolkit is `a3-ultra`. Make sure to verify the name of the local queue in your cluster.

Set the default project:

 ```
 gcloud config set project $PROJECT_ID
 ```

### Get the recipe

From your client, clone the `gpu-recipes` repository and set a reference to the repo folder.

```
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT="$REPO_ROOT/training/a3ultra/nccl-tests"
cd $RECIPE_ROOT
```

### Get cluster credentials

```
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

### Configure and run a NCCL test

The NCCL tests are run using a Helm chart that configures and submits a [Kubernetes Index Job](https://kubernetes.io/blog/2021/04/19/introducing-indexed-jobs/).

You can configure an NCCL test using several chart parameters. For a complete list, see the [Chart Configuration](#chart-configuration). To install the chart with default settings, execute the following command. This will run the `all_gather` NCCL test on two nodes using the default values specified in the table.

```
helm  install -f $RECIPE_ROOT/values.yaml \
    --set queue=${KUEUE_NAME} \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set-file workload_launcher=$REPO_ROOT/src/launchers/nccl-test.sh \
    --set workload.gpus=16 \
    $USER-nccl-test \
    $REPO_ROOT/src/helm-charts/a3ultra/job
```

You can modify any of the chart parameters using the Helm `--set` option. For example, to run the `all_reduce` NCCL test for 500 iterations on 4 nodes use the following command:

```
helm  install -f $RECIPE_ROOTvalues.yaml \
    --set queue=${KUEUE_NAME} \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
    --set-file workload_launcher=$REPO_ROOT/src/launchers/nccl-test.sh \
    --set workload.gpus=32 \
    --set workload.arguments[0]="--benchmark all_reduce --run_iters 200" \
    $USER-nccl-test \
    $REPO_ROOT/src/helm-charts/a3ultra/job
```

### Chart Configuration


The following table lists the configurable parameters and their default values.


| Parameter | Description | Required |Default |
| --------- | ----------- | ---------| ------- |
| `network.subnetworks[]`|A list of Kubernetes Network resource names for the host and GPU-to-GPU networks configured in your cluster. This parameter sets the correct network annotations for clusters that use a different naming schema for Network resources than those provisioned with the Cluster Toolkit blueprints. |No|N/A|
|`queue`| The name of the Kueue local queue. If specified, the NCCL test job will be submitted to this queue.| No |N/A|
|`workload.gpus`| The number of GPUs to run the test on | No | 16|
|`volumes.gcsMounts[0].bucketName`| The name of the bucket where the logs will be saved. The logs will be save to the `nccl-test-results` folder within the bucket| Yes| N/A|
|`workload.arguments[0]`| You can use this parameter to override the default NCCL test settings. . For a detailed explanation of supported overrides, refer to the note below this table.| No | See the note below|


The `workload.arguments[0]` parameter can be used to override the default settings for the NCCL test. The parameter's format is a space-separated list of NCCL test settings. The following settings are supported:


- `--benchmark BENCHMARK`, where the `BENCHMARK` can be one of the following: `all_gather`, `all_reduce`, `alltoall`, `broadcast`, `gather`, `reduce`, `reduce_scatter`, `hypercube`, `scatter`, `sendrecv`. The default is `all_gather`.
- `--mask MASK`, where the `MASK` sets the value of the NCCL_TESTS_SPLIT environment variable. The default is `0x0`.
- `--begin_msg_size BEGIN_MESSAGE_SIZE`, where `BEGIN_MESSAGE_SIZE` is the minimum message size to start the test with. The default is `1K`.
- `--end_msg_size END_MESSAGE_SIZE`, where `END_MESSAGE_SIZE` is the maximum message size to end the test with. The default is `16G`.
- `--factor FACTOR`, where `FACTOR` is the multiplication factor between message sizes. The default is 2.
- `--warmup_iters WARMUP_ITERS`, where `WARMUP_ITERS` is the number of warmup iterations. The default is `50`.
- `--run_iters RUN_ITERS`, where `RUN_ITERS` is the number of iterations to. The default is `100`.

For example, to run the `alltoall` test with messages ranging from 10K to 32G, performing 200 iterations and 100 warmup iterations, set the chart's `workload.arguments[0]` parameter as follows:

```
helm install -f $RECIPE_ROOT/values.yaml \
...
--set workload.arguments[0]="--benchmark alltoall --begin_msg_size 10K --end_msg_size 32G --run_iters 200 --warmup_iters 100"
...
```





