#!/bin/bash

export HF_TOKEN=${HF_TOKEN}
export PYTHONUNBUFFERED=1

pip3 install \
  transformers==4.46.3 \
  datasets \
  accelerate \
  peft \
  bitsandbytes \
  pillow \
  tensorboard

export LD_LIBRARY_PATH="$NCCL_PLUGIN_PATH"
ldconfig "$LD_LIBRARY_PATH"
echo "Added $LD_LIBRARY_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""

export NODE_RANK=$JOB_COMPLETION_INDEX
export HYDRA_FULL_ERROR=1
export NVIDIA_VISIBLE_DEVICES=0

echo "Launching Torch distributed as node rank $NODE_RANK out of $NNODES nodes"

mkdir /app
cat > /app/train.sh << 'EOF'
#!/bin/bash

python "$PYTHON_MAIN"
EOF

export TOKENIZERS_PARALLELISM=false
export NVTE_UB_SOCKET_IFNAME="eth1"
export NUM_TRAIN_EPOCHS=1
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=2
chmod +x /app/train.sh

accelerate launch --no_python /app/train.sh
