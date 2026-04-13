helm install $USER-a4x-max-llama3-1-405b-64gpus . -f values.yaml \
  --set-file workload_launcher=launcher.sh \
  --set-file workload_config=custom_setup_experiment.py \
  --set workload.image=nvcr.io/nvidia/nemo:26.02 \
  --set volumes.gcsMounts[0].bucketName=ubench-logs \
  --set volumes.gcsMounts[0].mountPath=/job-logs \
  --set workload.envs[0].value=/job-logs/$USER-a4x-max-llama3-1-405b-64gpus \
  --set queue=tas-lq \
  --set workload.hfToken=$HF_TOKEN
