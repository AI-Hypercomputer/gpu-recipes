"""Nemo2 pretraining recipe for Nemotron4 340B model."""

from nemo.collections import llm
from nemo.collections.llm.recipes import nemotron4_340b
from nemo.lightning.pytorch.callbacks import NsysCallback
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.utils.loggers.dllogger import DLLogger
import nemo_run as run
from scripts.performance.helpers import (
    set_primary_perf_configs,
)
from scripts.performance.utils import get_comm_overlap_callback_idx


def recipe(
    profile_enabled: bool = False,
    profile_start_step: int = 0,
    profile_end_step: int = 0,
    profile_ranks: str = "0",
) -> run.Partial:
  """Returns a Nemo2 training recipe for Nemotron4 340B model.

  Args:
      profile_enabled: Whether to enable Nsys profiling.
      profile_start_step: The step to start profiling.
      profile_end_step: The step to end profiling.
      profile_ranks: The ranks to profile, comma separated.

  Returns:
      A Nemo2 training recipe.
  """
  # Start from the Nemo standard recipe.
  pretrain = nemotron4_340b.pretrain_recipe(performance_mode=True)

  num_nodes = 32
  num_gpus_per_node = 4
  mbs = 1
  gbs = 256
  max_steps = 100
  tp_size = 4
  pp_size = 8
  cp_size = 1
  vp_size = 1
  ep_size = 1
  activation_offload_layers = 0
  enable_cuda_graphs = True
  compute_dtype = "bf16"
  fp8_recipe = None
  nccl_communicator_config_path = None
  use_mcore_fsdp = False
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
      activation_offload_layers=activation_offload_layers,
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

  comm_overlap_callback_idx = get_comm_overlap_callback_idx(
      pretrain.trainer.callbacks
  )
  pretrain.trainer.callbacks[
      comm_overlap_callback_idx
  ].tp_comm_bootstrap_backend = "nccl"
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
          model_name="nemotron4-340b",
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
