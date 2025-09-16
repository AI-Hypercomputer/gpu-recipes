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


# Reference: https://huggingface.co/blog/paligemma2.
"""Fine-tunes a PaliGemma model on a given dataset."""

import os
import datasets
import torch
import transformers

load_dataset = datasets.load_dataset
TrainingArguments = transformers.TrainingArguments
Trainer = transformers.Trainer
PaliGemmaProcessor = transformers.PaliGemmaProcessor
PaliGemmaForConditionalGeneration = transformers.PaliGemmaForConditionalGeneration

num_train_epochs = int(os.getenv("NUM_TRAIN_EPOCHS", "3"))
per_device_train_batch_size = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "96"))
gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
learning_rate = float(os.getenv("LEARNING_RATE", "2e-5"))
warmup_steps = int(os.getenv("WARMUP_STEPS", "2"))
weight_decay = float(os.getenv("WEIGHT_DECAY", "1e-6"))
adam_beta2 = float(os.getenv("ADAM_BETA2", "0.999"))
logging_steps = int(os.getenv("LOGGING_STEPS", "100"))
save_steps = int(os.getenv("SAVE_STEPS", "1000"))
save_total_limit = int(os.getenv("SAVE_TOTAL_LIMIT", "1"))

dataset_path = os.getenv("DATASET_PATH", "merve/vqav2-small")
dataloader_num_workers = int(os.getenv("DATALOADER_NUM_WORKERS", "16"))

model_id = os.getenv("MODEL_ID", "google/paligemma2-3b-pt-224")


print(f"[INFO] Loading {dataset_path} dataset")
ds = load_dataset(dataset_path, split="validation")
ds = ds.train_test_split(test_size=0.01)["train"]

processor = PaliGemmaProcessor.from_pretrained(model_id)


def collate_fn(examples):
    """Collate function for the dataset."""
    texts = ["<image>answer en " + example["question"] for example in examples]
    labels = [example["multiple_choice_answer"] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
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
    optim="adamw_torch",
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    output_dir="out_paligemma",
    bf16=True,
    report_to=["tensorboard"],
    dataloader_pin_memory=False,
    dataloader_num_workers=dataloader_num_workers,
)

trainer = Trainer(
    model=model, train_dataset=ds, data_collator=collate_fn, args=training_args
)

trainer.train()
