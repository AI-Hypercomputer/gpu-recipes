"""Nemo2 pretraining recipe for Llama 3.1 70B model."""

from nemo.collections import llm
from nemo.collections.llm.recipes import llama31_70b
from nemo.lightning.pytorch.callbacks import NsysCallback
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.utils.loggers.dllogger import DLLogger
import nemo_run as run
from scripts.performance.helpers import (
    set_primary_perf_configs,
)


def recipe(
    profile_enabled: bool = False,
    profile_start_step: int = 0,
    profile_end_step: int = 0,
    profile_ranks: str = "0",
) -> run.Partial:
  """Returns a Nemo2 training recipe for Llama 3.1 70B model.

  Args:
      profile_enabled: Whether to enable Nsys profiling.
      profile_start_step: The step to start profiling.
      profile_end_step: The step to end profiling.
      profile_ranks: The ranks to profile, comma separated.

  Returns:
      A Nemo2 training recipe.
  """
  # Start from the Nemo standard recipe.
  pretrain = llama31_70b.pretrain_recipe(performance_mode=True)

  num_nodes = 32
  num_gpus_per_node = 4
  mbs = 1
  gbs = 2048
  max_steps = 100
  tp_size = 1
  pp_size = 1
  cp_size = 1
  vp_size = 1
  ep_size = 1
  enable_cuda_graphs = False
  compute_dtype = "fp8"
  fp8_recipe = "cs"
  nccl_communicator_config_path = None
  use_mcore_fsdp = True
  use_fsdp_double_buffer = False
  use_user_buffer_registration = False
  use_sharp = False
  keep_fsdp_fp8_transpose_cache = False

  pretrain = set_primary_perf_configs(
      pretrain,
      "pre_train",
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
      fp8_recipe=fp8_recipe,
      nccl_communicator_config_path=nccl_communicator_config_path,
      use_mcore_fsdp=use_mcore_fsdp,
      use_fsdp_double_buffer=use_fsdp_double_buffer,
      use_user_buffer_registration=use_user_buffer_registration,
      use_sharp=use_sharp,
      keep_fsdp_fp8_transpose_cache=keep_fsdp_fp8_transpose_cache,
  )

  # Disable validation batches.
  pretrain.trainer.limit_val_batches = 0.0
  pretrain.trainer.val_check_interval = 100

  # Disable checkpointing.
  pretrain.log.ckpt = None
  pretrain.trainer.enable_checkpointing = False

  # Log every step.
  pretrain.trainer.log_every_n_steps = 1

  # Add the Nsys profiling callback if enabled.
  if profile_enabled:
    pretrain.trainer.callbacks.append(
        run.Config(
            NsysCallback,
            start_step=profile_start_step,
            end_step=profile_end_step,
            ranks=[int(x) for x in profile_ranks.split(",")],
            gen_shape=False,
        )
    )

  # Add the FLOPs measurement callback.
  pretrain.trainer.callbacks.append(
      run.Config(
          FLOPsMeasurementCallback,
          model_name="llama31-70b",
          model_config=pretrain.model.config,
          data_config=pretrain.data,
      )
  )

  # Enable DLLogger
  dllogger_config = run.Config(
      DLLogger,
      verbose=True,
      stdout=True,
      json_file="dllogger.json",
  )
  pretrain.log.extra_loggers = [dllogger_config]

  return pretrain


if __name__ == "__main__":
  run.cli.main(llm.pretrain, default_factory=recipe)
