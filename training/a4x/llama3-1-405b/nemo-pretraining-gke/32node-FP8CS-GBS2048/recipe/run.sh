#!/bin/bash

helm install mutianzhu-llama16 . -f values.yaml \
  --set-file workload_launcher=launcher.sh \
  --set-file workload_config=llama3-1-405b-fp8cs-gbs2048-gpus128.py \
  --set 'volumes.gcsMounts[0].bucketName=ubench-logs' \
  --set 'volumes.gcsMounts[0].mountPath=/job-logs' \
  --set 'workload.envs[0].value=/job-logs/mutianzhu-ubench-405b1' \
  --set queue=tas-lq
