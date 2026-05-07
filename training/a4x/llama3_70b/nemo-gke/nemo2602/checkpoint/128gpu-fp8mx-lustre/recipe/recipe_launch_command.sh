
# run command
NAME=asq-llama70b-ckpt-128gpu-gcs
helm install "${NAME}" . -f values.yaml --set-file workload_launcher=launcher.sh --set-file workload_config=custom_setup_experiment.py --set-file run_script_config=run_script.py --set workload.image=nvcr.io/nvidia/nemo:26.02.01 --set workload.envs[0].value=/job-logs/${NAME}


