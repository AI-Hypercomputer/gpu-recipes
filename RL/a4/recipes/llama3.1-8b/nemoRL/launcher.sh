#!/bin/bash
REPLICA_COUNT=4

helm install ray-cluster ../llama3.1-8b \
  --set values.additionalWorkerGroups.worker-grp-0.replicas=$REPLICA_COUNT