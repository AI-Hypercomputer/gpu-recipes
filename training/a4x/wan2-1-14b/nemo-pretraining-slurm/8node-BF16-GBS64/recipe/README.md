<!-- mdformat global-off -->
# Pretrain wan2.1-14b workloads on a4x Slurm with Nvidia NeMo Framework

This recipe outlines the steps for running a wan2.1-14b pretraining
workload on [a4x Slurm](https://docs.cloud.google.com/ai-hypercomputer/docs/create/create-slurm-cluster) by using the
[NVIDIA NeMo framework](https://github.com/NVIDIA/nemo).

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [SLURM](https://docs.cloud.google.com/ai-hypercomputer/docs/create/create-slurm-cluster)

## Test environment

This recipe has been optimized for and tested with the following configuration:

- SLURM cluster
Please follow [instructions](https://docs.cloud.google.com/ai-hypercomputer/docs/create/create-slurm-cluster)
to create your a4x SLURM cluster.

## Training dataset

This recipe uses a mock pretraining dataset provided by the NeMo framework.

## Docker container image

This recipe uses the following docker images:

- `nvcr.io/nvidia/nemo:25.11`

## Run the recipe

From your client workstation, complete the following steps:

### Configure environment settings

Set the environment variables to match your environment:

 ```bash
 export PROJECT_ID=<PROJECT_ID>
 export CLUSTER_REGION=<CLUSTER_REGION>
 export CLUSTER_NAME=<CLUSTER_NAME>
 gcloud compute ssh $a --project supercomputer-testing --zone $CLUSTER_REGION -- -o Hostname=nic0.$a.$CLUSTER_NAME.c.$PROJECT_ID$.internal.gcpnode.com

 ```

Replace the following values:

 - `<PROJECT_ID>`: your Google Cloud project ID.
 - `<CLUSTER_REGION>`: the region where your cluster is located.
 - `<CLUSTER_NAME>`: the name of your SLURM cluster.

Set the default project:

 ```bash
 gcloud config set project $PROJECT_ID
 ```

### Get the recipe

Clone the `gpu-recipes` repository and set a reference to the recipe folder.

```
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/training/a4x/wan2-1-14b/nemo-pretraining-slurm/8node-BF16-GBS64/recipe
cd $RECIPE_ROOT
```

### Configure and submit a pretraining job

#### Using 8 node (32 gpus) precision
To execute the job with the default settings, run the following command from
your client:

```bash
sbatch wan_14b_benchmark.sh
```

### Monitor the job

To check the status of pods in your job, run the following command:

```
squeue --me
```


To get the logs for the job, run the following command:

```
tail -f wan-14b-benchmark_{jobID}.out
```

### Uninstall the job

```bash
scancel -u $USER
```