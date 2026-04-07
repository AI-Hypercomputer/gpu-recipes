<!-- mdformat global-off -->
# Pretrain Llama 3 8B workloads on A3U Slurm Cluster with Nvidia Megatron-Bridge

This recipe outlines the steps for running a Llama 3 8B pretraining workload on [Google Cloud A3U Slurm clusters](https://docs.cloud.google.com/ai-hypercomputer/docs/create/create-slurm-cluster) by using [NVIDIA Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge).

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Slurm Workload Manager](https://slurm.schedmd.com/)
- Deployment - [Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/overview)

## Test environment

This recipe has been optimized for and tested with the following configuration:

- A3U Slurm Cluster (2 nodes, 16 GPUs)
- Machine Type: `a3-ultragpu-8g`
- Lustre Filesystem

Please follow the instructions in the [Cluster Toolkit A3U Example README](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/examples/machine-learning/a3-ultragpu-8g) to provision an A3 ultra Slurm cluster.

Version of the blueprint to create the slurm cluster is `v1.82.0`

## Docker container image

This recipe uses the following container images:

- `nvcr.io/nvidia/nemo:25.11`

## Run the recipe


### Configure environment settings

Set the environment variables to match your environment:

 ```bash
 export PROJECT_ID=<PROJECT_ID>
 export CLUSTER_REGION=<CLUSTER_REGION>
 export CLUSTER_NAME=<CLUSTER_NAME>
 gcloud compute ssh $CLUSTER_NAME --project <project-name> --zone $CLUSTER_REGION -- -o Hostname=nic0.$CLUSTER_NAME.$CLUSTER_REGION.c.$PROJECT_ID$.internal.gcpnode.com

 ```

Replace the following values:

 - `<PROJECT_ID>`: your Google Cloud project ID.
 - `<CLUSTER_REGION>`: the region where your cluster is located.
 - `<CLUSTER_NAME>`: the name of your SLURM cluster.

Set the default project:

 ```bash
 gcloud config set project $PROJECT_ID
 ```

From your cluster login node, complete the following steps:

### Get the recipe

Clone the `gpu-recipes` repository and set a reference to the recipe folder.

```
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/training/a3u/llama3-8b/megatron-bridge-pretraining-slurm/2node-FP8CS-GBS128/recipe
cd $RECIPE_ROOT
```

### Submit a pretraining job


```
# set your HF_TOKEN inside launch_script.sh
export HF_TOKEN="YOUR_HF_TOKEN" # Replace with your Hugging Face token.

cd ..
sbatch ./recipe/sbatch_script.sh
```

### Monitor the job

To check the status of pods in your job, run the following command:

```
squeue --me
```


To get the logs for the job, run the following command:

```
tail -f slurm_{jobID}.out
```

### Uninstall the job

```bash
scancel -u $USER
```
