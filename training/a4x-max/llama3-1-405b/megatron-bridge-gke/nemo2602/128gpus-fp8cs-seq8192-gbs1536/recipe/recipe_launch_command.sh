helm install $USER-a4x-max-llama3-1-405b-128gpus . -f values.yaml \
  --set-file workload_launcher=launcher.sh \
  --set-file workload_config=custom_setup_experiment.py \
  --set workload.image=nvcr.io/nvidia/nemo:26.02 \
  --set queue=tas-lq \
  --set workload.hfToken=$HF_TOKEN

