# Pretrain Llama-3.1-405B workloads on A3 Ultra GKE Node pools using the Google Cloud Resiliency library.

This recipe outlines the steps for running a Llama-3.1-405B pretraining workload on
[A3 Ultra GKE Node pools](https://cloud.google.com/kubernetes-engine) by using the
[NVIDIA NeMo framework](https://github.com/NVIDIA/nemo). The recipe uses the Google Cloud Resiliency library to maximize Goodput of a training job.


## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine),
- Job configuration and deployment - a Helm chart is used to configure and deploy the
  [Kubernetes JobSet](https://github.com/kubernetes-sigs/jobset).
  This JobsSet resource encapsulates a NVIDIA NeMo 2.0 Megatron GPT pretraining workload
  with built-in checkpointing optimizations provided by the Google Cloud Resiliency library.
- Job resilience - The Google Cloud Resiliency library supervises the cluster and the workloads running on it, performing mitigation actions in case of failures.

## Test environment

This recipe has been tested on the following environment configuration:

- A GKE cluster with the following configuration:
  - GKE version: 1.32.2-gke.1475000 or later,
  - A GPU node pool with 54  [a3-ultragpu-8g](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) machines using a [compact placement policy](https://cloud.google.com/kubernetes-engine/docs/how-to/compact-placement),
  - A CPU node pool with 2 e2-standard-8 machines,
  - Workload Identity Federation for GKE enabled cluster,
  - Cloud Storage FUSE CSI driver enabled,
  - DCGM metrics enabled,
  - Kueue and JobSet APIs are installed.
- A Google Cloud Storage bucket with [hierarchical namespace](https://cloud.google.com/storage/docs/hns-overview) is used for checkpoints and logs. The bucket must be in the same region as the GKE cluster
- A client workstation with the following pre-installed:
   - Google Cloud SDK,
   - Helm,
   - kubectl.

To prepare the required environment, see
[GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-ultra.md).

### Add a CPU node pool to the GKE cluster

The Cluster Toolkit blueprint used in the [GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-ultra.md) doesn't create a dedicated CPU node pool required by this recipe to host Google PyTorch Resilience library components. To create the node pool execute the following command after the cluster is ready.

```
gcloud container node-pools create supervisor-cpu-pool \
--cluster <CLUSTER_NAME> \
--location <CLUSTER_REGION> \
--machine-type e2-standard-8 \
--num-nodes 2 \
--node-locations <ZONE>
```

Replace the following:
- `CLUSTER_NAME`: the name of your GKE cluster
- `CLUSTER_REGION`: the region where you cluster is located
- `ZONE`: the zone where your cluster's A3 Ultra node pool is located. It is recommended that the CPU node pool is in the same zone as the A3 Ultra node pool

### Enable Topology Aware Scheduling for Kueue

While the Cluster Toolkit blueprint in the [GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-ultra.md) installs and configures Kueue, it does not enable Topology Aware Scheduling (TAS).

To enable TAS, run the following command:

```bash
kubectl -n kueue-system patch deployment kueue-controller-manager \
  --type json \
  -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--feature-gates=TopologyAwareScheduling=true"}]'
```

### Setup All-or-nothing with ready Pods


Training jobs require all pods to run simultaneously. By default, the [all-or-nothing scheduling in Kueue]((https://kueue.sigs.k8s.io/docs/tasks/manage/setup_wait_for_pods_ready/), which is optimal for these jobs, is not configured.

To configure all-or-nothing scheduling execute the following commands:

1. Patch the `kueue-manager-config`
    ```bash
    kubectl patch configmap kueue-manager-config \
      -n kueue-system \
      --type merge \
      --patch-file kueue-merge-patch.yaml
    ```

2.  Restart the controller Pod

    ```bash
    kubectl rollout restart deployment/kueue-controller-manager -n kueue-system
    ```

3.  Check the new configuration

    You can check the new configuration by running: `bash kubectl get configmap
    kueue-manager-config -n kueue-system -o yaml`

    In the output, you should see the `waitForPodsReady` section updated in the
    `controller_manager_config.yaml` data field, looking like this:

    ```yaml
    ...
        waitForPodsReady:
          enable: true
          timeout: 1m
          recoveryTimeout: 1m
    ...
    ```


## Training dataset

The recipe uses a mock pretraining dataset provided by the NeMo framework.


## Run the recipe

It's recommended to use Cloud Shell as your client to complete the steps.
Cloud Shell comes pre-installed with the necessary utilities, including
`kubectl`, `the Google Cloud SDK`, and `Helm`.

### Launch Cloud Shell

In the Google Cloud console, start a [Cloud Shell Instance](https://console.cloud.google.com/?cloudshell=true).

### Configure environment settings

From your client, complete the following steps:

1. Set the environment variables to match your environment:

  ```bash
  export PROJECT_ID=<PROJECT_ID>
  export CLUSTER_REGION=<CLUSTER_REGION>
  export CLUSTER_NAME=<CLUSTER_NAME>
  export GCS_BUCKET=<GCS_BUCKET>
  export KUEUE_NAME=<KUEUE_NAME>
  ```
  Replace the following values:

  - `<PROJECT_ID>`: your Google Cloud project ID
  - `<CLUSTER_REGION>`: the region where your cluster is located
  - `<CLUSTER_NAME>`: the name of your GKE cluster
  - `<GCS_BUCKET>`: the name of your Cloud Storage bucket with hierarchical namespace enabled. Don't include the `gs://` prefix
  - `<KUEUE_NAME>`: the name of the Kueue local queue configured for TAS.  The default queue created by the cluster toolkit is `a3-ultra`. Make sure to verify the name of the local queue in your cluster.

2. Set the default project:

  ```bash
  gcloud config set project $PROJECT_ID
  ```

### Get the recipe

From your client, clone the `gpu-recipes` repository and set a reference to the recipe folder.

```
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/training/a3ultra/llama3-1-405b/nemo-pretraining-gke-resiliency
```

### Get cluster credentials

From your client, get the credentials for your cluster.

```
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

### Deploy supervisor components from the Google Cloud Resiliency library.


In this section, you'll deploy the supervisor components from the Google Cloud Resiliency library into your GKE cluster. The supervisor comprises two primary components: the Supervisor itself and Host Monitors.

The Supervisor is deployed as a set of three [Kubernetes Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/): the Actuator, the Sensor, and the Controller. They're deployed to the CPU node pool of your cluster.

The Host Monitor is deployed as a [Kubernetes DaemonSet](https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/). A Host Monitor Pod is deployed to each A3 Ultra node in the A3 Ultra node pool of your cluster.


#### Create Supervisor Service Account and Role

Run the following command to create a GKE service account and role for the Supervisor.

```
cd $RECIPE_ROOT
kubectl apply -f ksa-setup.yaml
```

To verify that the account was created successfully execute the following command:

```
kubectl get clusterroles | grep supervisor
```

If the output isn't empty, the cluster role was successfully created.


#### Deploy the Supervisor and the Host Monitor

```
cd $RECIPE_ROOT
helm install -f values-supervisor.yaml \
  $USER-supervisor \
  $REPO_ROOT/src/helm-charts/resiliency/supervisor-chart
```

To verify that the Supervisor has been deployed successfully, execute the following command:

```
kubectl get deployments
```

You should see an output similar to the following:

```
NAME                                      READY   UP-TO-DATE   AVAILABLE   AGE
user-supervisor-actuator-deployment     1/1     1            0           4m36s
user-supervisor-controller-deployment   1/1     1            0           4m36s
user-supervisor-sensor-deployment       1/1     1            0           4m36s
```

The Supervisor is ready when all Deployments are in the Ready state.

To verify that the Host Monitors have been deployed successfully, execute the following command:

```
kubectl get daemonsets
```

You should see an output similar to the following

```
NAME                               DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR                   AGE
user-supervisor-host-daemonset   16        16        16      16           16          cloud.google.com/gke-gpu=true   8m1s
```

The Host Monitor is ready when the number of `READY` pods is equal to the number of nodes in your cluster's A3 Ultra node pool.


### Create Persistent Volumes and Persistent Volume Claims

The JobSet, orchestrating a training job, accesses a GCS checkpoints and logs bucket through the [the Cloud Storage FUSE CSI driver](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver) configured using a Kubernetes Persistent Volume (PV) and Persistent Volume Claim (PVC). You must generate a PV and a PVC for the bucket using the [gcs-fuse helper Helm chart](../../../../src/helm-charts/storage/gcs-fuse). The chart configures the FUSE driver settings following best practices for optimizing access to a bucket that manages checkpoints.

```
helm install -f values-gcs.yaml \
--set gcsVolumes[0].bucketName=${GCS_BUCKET} \
$USER-gcs-pv-pvc \
$REPO_ROOT/src/helm-charts/storage/gcs-fuse
```

### Configure and submit a training job

By default the training job will run 500 steps and generate a checkpoint every 20 steps.
To execute the job with the default settings, run the following command from your client:

```bash
helm install -f $RECIPE_ROOT/values.yaml \
  --set-file workload_launcher=$REPO_ROOT/src/launchers/grl-nemo-20-launcher-a3u.sh \
  --set-file workload_config=$RECIPE_ROOT/train.py \
  --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
  --set queue=$KUEUE_NAME \
  $USER-rt \
  $REPO_ROOT/src/helm-charts/a3ultra/jobset
```


#### Monitor the training job

To check the status of pods in the indexed job, run the following command from your client:

```
kubectl get pods | grep $USER-rt
```

To get and follow the logs for one of the pods, run the following command from your client:

```
kubectl logs -f "<pod_name>"
```

#### Goodput Analysis for the job

This section explains how to perform the goodput analysis of the training job. You can use the [resiliency-metrics](../../../../src/utils/resiliency_metrics/) tool
to help analyze the training job and provide the Goodput Percentage of the training. The tool calculates the following metrics:

- *Total Events*: the total events during the training session
- *Job Started Count*: the number of times the job have started
- *Checkpoints Loaded*: the number of times checkpoints have been loaded
- *Checkpoints Saved*: the number of times a checkpoint have been saved
- *Total Runtime (hours)*: the total number of hours of the job
- *Min Loaded Step*: the minimum step used to load a checkpoint
- *Max Saved Step*: the maximum step on which a checkpoint was saved
- *Step Difference*: the progress made in number of steps
- *Effective Computation Time (hours)*: the amount of time effectively running the training job
- *Goodput Percentage*: the percentage of the total time that the job was making progress

To run the analysis:

1. Create virtual environment and install the required libraries:
```bash
cd $REPO_ROOT/src/utils/resiliency_metrics
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```
2. Run the [`calculator.py`](../../../../src/utils/resiliency_metrics/calculator.py)
   script, specifying the <JOBSET_NAME>

```bash
python3 calculator.py --job-name <JOBSET_NAME> \
  --export mymetrics.json \
  --gcloud-logging-lookback-days 1 \
  --verbose \
  --reference-step-time=45
```

To get the <JOBSET_NAME> you can run:
```
kubectl get jobsets
```


### Remove the training job

To delete the training job execute the following command:

```bash
helm uninstall $USER-rt
```

#### Uninstall the Supervisor and the Host Monitors

To remove the Supervisor and the Host Monitors, execute the following command:

```bash
helm uninstall $USER-supervisor
```

