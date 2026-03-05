"""Nemo2 pretraining recipe for Llama 3.1 70B model."""

import os
import time

from megatron.core.transformer.enums import AttnBackend
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.collections.llm.peft import LoRA
from nemo.collections.llm.recipes import llama31_70b
from nemo.lightning.pytorch.callbacks import NsysCallback
from nemo.utils.loggers.dllogger import DLLogger
import nemo_run as run
from scripts.performance.helpers import set_primary_perf_configs


HF_MODEL_URI = "meta-llama/Llama-3.1-70B"


def recipe(
    profile_enabled: bool = False,
    profile_start_step: int = 0,
    profile_end_step: int = 0,
    profile_ranks: str = "0",
) -> run.Partial:
  """Returns a Nemo2 finetuning recipe for Llama 3.1 70B model.

  Args:
      profile_enabled: Whether to enable Nsys profiling.
      profile_start_step: The step to start profiling.
      profile_end_step: The step to end profiling.
      profile_ranks: The ranks to profile, comma separated.

  Returns:
      A Nemo2 finetuning recipe.
  """
  # Start from the Nemo standard recipe.
  finetune = llama31_70b.finetune_recipe(seq_length=4096)

  num_nodes = 1
  num_gpus_per_node = 8
  mbs = 1
  gbs = 32
  max_steps = 10
  tp_size = 1
  pp_size = 4
  cp_size = 1
  vp_size = 20
  ep_size = 1
  enable_cuda_graphs = False
  compute_dtype = "bf16"

  finetune = set_primary_perf_configs(
      finetune,
      "lora",
      num_nodes=num_nodes,
      num_gpus_per_node=num_gpus_per_node,
      mbs=mbs,
      gbs=gbs,
      max_steps=max_steps,
      tp_size=tp_size,
      pp_size=pp_size,
      cp_size=cp_size,
      vp_size=vp_size,
      ep_size=ep_size,
      enable_cuda_graphs=enable_cuda_graphs,
      compute_dtype=compute_dtype,
  )

  finetune.data.tokenizer = run.Config(
      AutoTokenizer,
      pretrained_model_name=HF_MODEL_URI,
      use_fast=True,
  )

  finetune.trainer.plugins.precision = "bf16-mixed"
  finetune.trainer.plugins.grad_reduce_in_fp32 = True
  finetune.peft = LoRA(dim=16, alpha=32)

  finetune.optim.config.use_precision_aware_optimizer = False

  # Disable validation batches.
  finetune.trainer.limit_val_batches = 0.0
  finetune.trainer.val_check_interval = 10

  # Disable checkpointing.
  finetune.log.ckpt = None
  finetune.trainer.enable_checkpointing = False

  # Log every step.
  finetune.trainer.log_every_n_steps = 1

  finetune.model.config.attention_backend = AttnBackend.flash

  # Add the Nsys profiling callback if enabled.
  if profile_enabled:
    finetune.trainer.callbacks.append(
        run.Config(
            NsysCallback,
            start_step=profile_start_step,
            end_step=profile_end_step,
            ranks=[int(x) for x in profile_ranks.split(",")],
            gen_shape=False,
        )
    )

  # Enable DLLogger
  dllogger_config = run.Config(
      DLLogger,
      verbose=True,
      stdout=True,
      json_file="dllogger.json",
  )
  finetune.log.extra_loggers = [dllogger_config]

  return finetune


if __name__ == "__main__":
  os.environ["NCCL_P2P_LEVEL"] = "SYS"

  if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    llm.import_ckpt(model=llm.LlamaModel(llm.Llama31Config70B(
        cross_entropy_loss_fusion=False)),
                    source=f"hf://{HF_MODEL_URI}")

  run.cli.main(llm.finetune, default_factory=recipe)
