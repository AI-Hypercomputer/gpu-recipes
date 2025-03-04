# nccl-tests

The **nccl-tests** chart enables you to run [NVIDIA NCCL tests](https://github.com/NVIDIA/nccl-tests) on Google Kubernetes Engine (GKE) clusters with A3 Ultra node pools. This chart runs a NCCL test as a Kubernetes Job, allowing you to configure and execute the test on a specified number of A3 Ultra nodes.

## Prerequisites

Before running a NCCL test, ensure your environment is configured as follows:

- A GKE cluster with an A3 Ultra node pool.
- A Google Cloud Storage (GCS) bucket to store results. *Important: The Kubernetes default service account of your cluster must have [Storage Object User](https://cloud.google.com/iam/docs/understanding-roles#storage.objectUser) permissions on the bucket*.
- A client workstation with the following pre-installed:
  - Google Cloud SDK
  - Helm
  - kubectl

To prepare the required environment, see the [GKE with A3 Ultra node pools setup guide](../../../docs/configuring-environment-gke-a3-ultra.md).

## Configure environment settings

From your client, complete the following steps:

1. Set the environment variables to match your environment:

    ```bash
    PROJECT_ID=<PROJECT_ID>
    CLUSTER_NAME=<CLUSTER_NAME>
    CLUSTER_LOCATION=<CLUSTER_LOCATION>
    GCS_BUCKET=<GCS_BUCKET>
    LOGS_FOLDER=<LOGS_FOLDER>
    ```

    Replace the following values:
      - `<PROJECT_ID>`: your Google Cloud project ID
      - `<CLUSTER_LOCATION>`: the location of your cluster
      - `<CLUSTER_NAME>`: the name of your GKE cluster
      - `<GCS_BUCKET>`: the name of your Cloud Storage bucket. Do not include the `gs://` prefix
      - `<LOGS_FOLDER>`: the name of the folder within the <GCS_BUCKET> bucket for storing results

1. Set the default project:

    ```
    gcloud config set project $PROJECT_ID
    ```

1. Get cluster credentials

    ```
    gcloud container clusters get-credentials $CLUSTER_NAME --location $CLUSTER_LOCATION
    ```

## Get the chart

From your client, clone the `gpu-recipes` repository and set a reference to the repo folder.

```
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
```

## Run a NCCL test

You can configure an NCCL test using several chart parameters. For a complete list, see the [Chart Configuration](#chart-configuration) section. To install the chart with default settings, execute the following command. This will run the `all_gather` NCCL test using the default values specified in the table.

```
RELEASE_NAME=<HELM_RELEASE_NAME>

helm install \
--set clusterName=$CLUSTER_NAME \
--set testSettings.gcsBucket=$GCS_BUCKET \
--set testSettings.logsFolder=$LOGS_FOLDER \
$RELEASE_NAME \
$REPO_ROOT/src/helm-charts/a3ultra/nccl-tests
```

Replace the following values:
- `<RELEASE_NAME>` - the name of the chart release

*Important: This simplified chart configuration works with clusters provisioned using the [Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/overview) blueprints (see the referenced setup guides). If your cluster doesn't follow the default naming conventions for Kubernetes network resources used by the blueprints, you must use the `subnetwork` parameter instead of the `clusterName parameter`.  See the [Chart Configuration](#chart-configuration) section for details.*


You can modify any of the chart parameters using the Helm `--set` option. For example, to run the `all_reduce` NCCL test for 500 iterations on 4 nodes use the following command:

```
helm install \
--set clusterName=$CLUSTER_NAME \
--set testSettings.gcsBucket=$GCS_BUCKET \
--set testSettings.logsFolder=$LOGS_FOLDER \
--set testSettings.benchmark=all_reduce \
--set testSettings.nodes=2 \
--set testSettings.runIterations=500 \
$RELEASE_NAME \
$REPO_ROOT/src/helm-charts/a3ultra/nccl-tests
```

## Chart Configuration


The following table lists the configurable parameters of the `nccl-tests` chart and their default values.


| Parameter | Description | Required |Default |
| --------- | ----------- | ---------| ------- |
| `clusterName` | The name of the cluster. This value is used to automatically generate networking annotations for NCCL test jobs on clusters provisioned using the default settings in the Cluster Toolkit blueprints. For clusters using different network naming conventions, refer to the `network.subnetworks[] parameter`. | Yes, if the `network.subnetworks[]` is not se | N/A |
| `network.subnetworks[]`|A list of Kubernetes Network resource names for the host and GPU-to-GPU networks configured in your cluster. This parameter sets the correct network annotations for clusters that use a different naming schema for Network resources than those provisioned with the Cluster Toolkit blueprints. |Yes, if the `clusterName` parameter is not set|N/A|
|`queue`| The name of the Kueue local queue. If specified, the NCCL test job will be submitted to this queue.| No |N/A|
|`testSettings.nodes`|The number of nodes on which to run the test.|No|2|
|`testSettings.benchmark`|The NCCL test to run: `all_gather`, `all_reduce`, `alltoall`, `broadcast`, `gather`, `reduce`, `reduce_scatter`, `hypercube`, `scatter`, `sendrecv`.|No|`all_gather`|
|`testSettings.mask`|This parameter sets the value of the NCCL_TESTS_SPLIT environment variable.|No|0x0|
|`testSettings.beginMessageSize`|The minimum message size to start with.|No|1K|
|`testSettings.endMessageSize`|The maximum message size to end with.|No|16G|
|`testSettings.factor`|The multiplication factor between message sizes.|No|2|
|`testSettings.warmupIterations`|The number of warmup iterations.|No|50|
|`testSettings.runIterations`|The number of iterations.|No|100|
|`testSettings.gcsBucket`|The name of the GCS bucket for storing results.|Yes|N/A|
|`testSettings.logsFolder`|The name of the folder within the GCS bucket for storing results.|Yes|N/A|
|`gpuPlatformSettings.useHostPlugin`|A boolean value. If `true`, the test uses NCCL libraries pre-installed on the GKE node. If `false`, the test installs and uses NCCL libraries from the `ncclPluginImage` container image.|No|true|
|`gpuPlatformSettings.ncclPluginImage`|The NCCL plugin container image to use.|No|`us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic:v1.0.3`|
|`tasSettings.useLegacyTAS`|A boolean value. If `true`, the chart configures the `gke.io/topology-aware-auto-scheduling` scheduling gate to use the legacy [Topology Aware Scheduler](https://github.com/GoogleCloudPlatform/container-engine-accelerators/tree/master/gke-topology-scheduler). If `false` and the `queue` parameter is set, the chart adds Kueue topology annotations.|No|false|
|`tasSettings.topologyRequest`|If the `queue` parameter is set this value is used to configure Kueue topology annotations.|No|`kueue.x-k8s.io/podset-preferred-topology: "kubernetes.io/hostname"`|


