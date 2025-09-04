# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tunes a PaliGemma model on a given dataset."""

from datasets import load_dataset
import torch
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
import os

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

""" Parse env vars """

num_train_epochs = int(os.getenv("NUM_TRAIN_EPOCHS", "3"))
per_device_train_batch_size = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "4"))
gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
learning_rate = float(os.getenv("LEARNING_RATE", "2e-5"))
warmup_steps = int(os.getenv("WARMUP_STEPS", "2"))
weight_decay = float(os.getenv("WEIGHT_DECAY", "1e-6"))
adam_beta2 = float(os.getenv("ADAM_BETA2", "0.999"))
logging_steps = int(os.getenv("LOGGING_STEPS", "100"))
save_steps = int(os.getenv("SAVE_STEPS", "1000"))
preset_max_steps = int(os.getenv("MAX_STEPS", "1000"))
save_total_limit = int(os.getenv("SAVE_TOTAL_LIMIT", "1"))
optim = os.getenv("OPTIM", "adamw_torch")

dataloader_num_workers = int(os.getenv("DATALOADER_NUM_WORKERS", "1"))
dataset_name = os.getenv("DATASET_NAME", "merve/vqav2-small")
data_source = os.getenv("DATA_SOURCE", "lssd")
local_dataset_path = os.getenv("LOCAL_DATASET_PATH", "/ssd/waymo-open-dataset")

model_id = os.getenv("MODEL_ID", "google/paligemma2-3b-pt-224")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Starting..... on {device}")

""" Load dataset and model. """
processor = PaliGemmaProcessor.from_pretrained(model_id)

print(f"Loading {dataset_name} dataset from {data_source}")
ds = None
is_waymo_dataset = "waymo" in dataset_name

# assume load waymo dataset only from GCS
if is_waymo_dataset:
    if data_source == "lssd":
        print(f"Loading Waymo dataset from local {local_dataset_path}")
        ds = load_dataset(
            "parquet", data_dir=local_dataset_path, split="train", # streaming=True
        )
    else:  # GCS
        ds = load_dataset("parquet", data_dir="/data", split="train", streaming=True)
else:
    if data_source == "lssd":
        print(f"Loading dataset from huggingface datasets")
        ds = load_dataset(dataset_name, split="validation")
    else:  # GCS
        ds = load_dataset("parquet", data_dir="/data", split="validation")
print(f"Dataset loaded successfully from {data_source}")


def collate_fn(examples):
    texts = ["<image>answer en " + example["question"] for example in examples]
    labels = [example["multiple_choice_answer"] for example in examples]
    images = [example["image"] for example in examples]

    tokens = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="longest",
    )

    tokens = tokens.to(torch.bfloat16)
    return tokens


model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
print(f"Model loaded on {model.device}")

""" Get max steps"""

num_samples_in_dataset = 999 * 798 if is_waymo_dataset else 21500
global_batch_size = (
    per_device_train_batch_size
    * gradient_accumulation_steps
    * int(os.environ.get("WORLD_SIZE", "1"))
)
# Hack: set less max_steps for validation purpose
max_steps = (
    (num_samples_in_dataset * num_train_epochs) // global_batch_size
    if preset_max_steps < 0
    else preset_max_steps
)

# Training Arguments from parsed arguments
training_args = TrainingArguments(
    num_train_epochs=num_train_epochs,
    remove_unused_columns=False,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    adam_beta2=adam_beta2,
    logging_steps=logging_steps,
    optim=optim,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    output_dir="out_paligemma",
    bf16=True,
    gradient_checkpointing=True,  # enable reduce memory consumption
    report_to=["tensorboard"],
    dataloader_pin_memory=True,
    max_steps=max_steps,
    dispatch_batches=False,  # if is_waymo_dataset else True,
    dataloader_num_workers=dataloader_num_workers,
)

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
trainer = Trainer(
    model=model,
    train_dataset=ds,
    data_collator=collate_fn,
    args=training_args,
)
print(
    f'Rank-{os.environ.get("RANK", "0")} out of total '
    f'{os.environ.get("WORLD_SIZE", "None")} ranks starts training....'
)

from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_flops=True
) as prof:
    train_result = trainer.train()

print(f"Rank-{os.environ.get("RANK", "0")} Training completed: {train_result=}")

# After training is complete, access the state
total_flops_calculated = trainer.state.total_flos
train_runtime = train_result.metrics["train_runtime"]
num_devices = int(os.environ.get("WORLD_SIZE", "8"))
print(
    f"Local {os.getenv("LOCAL_RANK")}: Total FLOPs calculated by Trainer: {trainer.state.total_flos}"
)

print(f"===== Calculating TFLOPS/device/second.")
if os.getenv("LOCAL_RANK", "0") == "0":
    rank = os.getenv("LOCAL_RANK")
    if total_flops_calculated is not None and total_flops_calculated > 0:
        print(
            f"Local {rank=}: Total FLOPs calculated by Trainer: {total_flops_calculated}"
        )
        # The number is large, so you might want to format it (e.g., TFLOPs)
        print(f"Local {rank=}: Total TFLOPs: {total_flops_calculated / 1e12:.4f}")
        tflops_per_device_per_sec = (
            total_flops_calculated / 1e12 / train_runtime
        ) / num_devices
        print(
            f"Local {rank=}: Train runtime {train_runtime}, total devices {num_devices}, "
            f"TFLOPS/device/second {tflops_per_device_per_sec:.4f}"
        )
    else:
        print(
            f"Local {rank=}: FLOPs calculation was not available or returned zero."
        )

prof_log_path = os.path.join(
    os.environ.get("EXPERIMENT_ROOT_DIR"),
    os.environ.get('JOB_IDENTIFIER'),
    f"trace-{os.environ["LOCAL_RANK"]}.json"
)
if os.environ["LOCAL_RANK"] == "0":
    print(prof.key_averages().table(sort_by="flops", row_limit=10))
    prof.export_chrome_trace(prof_log_path)
