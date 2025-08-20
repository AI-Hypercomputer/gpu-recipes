# See the License for the specific language governing permissions and
# limitations under the License.

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


import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, cast
import fiddle as fdl
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.recipes.llama31_405b import model as llama_model
from nemo.lightning.pytorch.callbacks import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO
from nemo.utils.callbacks.dist_ckpt_io import DistributedCheckpointIO
from nvidia_resiliency_ext.ptl_resiliency import FaultToleranceCallback
from nvidia_resiliency_ext.ptl_resiliency.fault_tolerance_callback import SimulatedFaultParams
from resiliency.callbacks import model_checkpoint
from resiliency.callbacks.logging import StepLoggingCallback, TPSLoggingCallback
from resiliency.connectors import checkpoint_connector
from resiliency.plugins.combined_functionality import CombinedCheckpointIO
from resiliency.plugins.min_ckpt_overhead import MinCkptOverheadCheckpointIO
from resiliency.plugins.persistent_ckpt_proc import PersistentCheckpointProcessIO
from resiliency.plugins.replication_utils import ReplicatedOptimizerMegatronStrategy
from resiliency.utils import get_resiliency_logger
import torch


resiliency_logger = get_resiliency_logger(__name__)


def get_parser():
  parser = argparse.ArgumentParser(
      description="Llama3 1.405B Pretraining script using NeMo 2.0."
  )
  parser.add_argument(
      "--enable-optimized-async-ckpt",
      action="store_true",
      help="Enable optimized async checkpointing.",
      default=False,
  )
  parser.add_argument(
      "--enable-fault-tolerance",
      action="store_true",
      help="Enable nvrx fault tolerance.",
      default=False,
  )
  parser.add_argument(
      "--sim-fault-desc",
      type=str,
      help=(
          "Description of a fault to be simulated, format is:"
          " <fault_type>,<base_delay>."
      ),
      required=False,
      default="",
  )
  parser.add_argument(
      "--max-steps",
      type=int,
      default=1000,
      help="Max steps to run.",
  )
  parser.add_argument(
      "--max-runtime",
      type=int,
      default=86400,
      help="Max runtime in seconds.",
  )
  parser.add_argument(
      "--num-nodes",
      type=int,
      default=1,
      help="How many nodes to use.",
  )
  parser.add_argument(
      "--num-gpus",
      type=int,
      default=8,
      help="Specify the number of GPUs per node.",
  )
  parser.add_argument(
      "--log-dir",
      type=str,
      help="Output log dir.",
      required=False,
      default="/log/",
  )
  parser.add_argument(
      "--log-level",
      type=str,
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
      help="Set the logging level for the script.",
  )
  parser.add_argument(
      "--ckpt-dir",
      type=str,
      help="Output checkpoint dir.",
      required=False,
      default="/checkpoints/",
  )
  parser.add_argument(
      "--job-name",
      type=str,
      help="Job name of the current run.",
      required=False,
      default="test_job",
  )
  parser.add_argument(
      "--local-ckpt-interval",
      type=int,
      default=-1,
      help="Checkpoint saving to local storage interval in steps.",
  )
  parser.add_argument(
      "--persistent-ckpt-interval",
      type=int,
      default=-1,
      help="Checkpoint saving to persistent storage interval in steps.",
  )
  parser.add_argument(
      "--local-ckpt-dir",
      type=str,
      help="Local checkpoint dir.",
      required=False,
      default=None,
  )
  parser.add_argument(
      "--persistent-ckpt-dir",
      type=str,
      help="Checkpoint dir.",
      required=False,
      default=None,
  )
  parser.add_argument(
      "--ckpt-threads-per-rank",
      type=int,
      default=2,
      help="Number of threads to use for writing checkpoint files per rank.",
  )
  parser.add_argument(
      "--enable-dist-ckpt",
      action="store_true",
      help="Enable distributed checkpointing.",
      default=False,
  )
  parser.add_argument(
      "--enable-comm-overlap",
      action="store_true",
      help="Enable communication overlap.",
      default=False,
  )
  parser.add_argument(
      "--enable-gc",
      action="store_true",
      help="Enable garbage collection.",
      default=False,
  )
  parser.add_argument(
      "--enable-async-ckpt",
      action="store_true",
      help="Enable async checkpointing.",
      default=False,
  )
  parser.add_argument(
      "--global-bs",
      type=int,
      default=None,
      help="Global batch size.",
  )
  parser.add_argument(
      "--tokenizer-path",
      type=str,
      default="tokenizer.model",
      help="Path to the tokenizer model file.",
  )

  return parser


def get_ft_callback(log_file_dir, sim_fault_desc=None):
  simulated_fault = None
  if sim_fault_desc:
    fault_type, base_delay = sim_fault_desc.split(",")
    fault_type = fault_type.strip()
    base_delay = float(base_delay.strip())
    simulated_fault = SimulatedFaultParams(
        fault_type=fault_type,
        base_delay=base_delay,
    )
  ft_callback = FaultToleranceCallback(
      autoresume=False,
      calculate_timeouts=True,
      exp_dir=log_file_dir,
      simulated_fault_params=simulated_fault,
  )
  return ft_callback


def main():
  args = get_parser().parse_args()

  logging.basicConfig(level=args.log_level)

  model_config = llama_model()

  mbs = 1
  gbs = (
      (args.global_bs // (mbs * args.num_gpus * args.num_nodes))
      * (mbs * args.num_gpus * args.num_nodes)
      if args.global_bs
      else (mbs * args.num_gpus * args.num_nodes)
  )
  assert gbs > 0

  mix_model = fdl.build(model_config)
  if args.local_ckpt_dir:
    # Supports replicated distributed optimizer for local ckpt
    assert args.num_nodes % 18 == 0
    strategy = ReplicatedOptimizerMegatronStrategy(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=18,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=7,
        context_parallel_size=1,
        sequence_parallel=True,
        ckpt_async_save=args.enable_async_ckpt
        or args.enable_optimized_async_ckpt,
        ckpt_parallel_load=True,
        ddp=DistributedDataParallelConfig(
            num_distributed_optimizer_instances=args.num_nodes // 18
        ),
    )
  else:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=18,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=7,
        context_parallel_size=1,
        sequence_parallel=True,
        gradient_as_bucket_view=True,
        ckpt_async_save=args.enable_async_ckpt
        or args.enable_optimized_async_ckpt,
        ckpt_parallel_load=True,
        ddp=DistributedDataParallelConfig(),
    )

  log_dir = Path(args.log_dir) / args.job_name
  tb_dir = Path(log_dir) / "tb"
  callbacks = []
  callbacks.append(StepLoggingCallback(tb_dir))
  callbacks.append(
      TPSLoggingCallback(
          gbs=gbs,
          seq_length=model_config.config.seq_length,
          tb_dir=tb_dir,
      )
  )

  local_ckpt_dir = None
  persistent_ckpt_dir = None
  if args.local_ckpt_dir is not None:
    local_ckpt_dir = Path(args.local_ckpt_dir) / args.job_name / "checkpoint"
    callbacks.append(
        model_checkpoint.ModelCheckpoint(
            dirpath=local_ckpt_dir,
            save_last=False,
            monitor="step",
            # Disable deleting ckpt from training code,
            # as it take a long time to delete ckpts and block training progress.
            # ckpt_cleaner will delete old ckpts.
            save_top_k=-1,
            mode="max",
            save_weights_only=False,
            every_n_train_steps=args.local_ckpt_interval,
            save_on_train_epoch_end=True,
            save_optim_on_train_end=True,
            always_save_context=False,
            filename="{step}",
            enable_version_counter=False,
            use_in_cluster_local_ckpts=True,
            is_persistent_storage=False,
            enable_high_scale_ckpt=False,
            preprocess_files=False if args.persistent_ckpt_dir else True,
            priority=0,
            delete_unfinished_ckpt_on_start=True,
        )
    )

  if args.persistent_ckpt_dir is not None:
    persistent_ckpt_dir = (
        Path(args.persistent_ckpt_dir) / args.job_name / "checkpoint"
    )
    callbacks.append(
        model_checkpoint.ModelCheckpoint(
            dirpath=persistent_ckpt_dir,
            save_last=False,
            monitor="step",
            # Disable deleting ckpt from training code,
            # as it take a long time to delete ckpts and block training progress.
            save_top_k=-1,
            mode="max",
            save_weights_only=False,
            every_n_train_steps=args.persistent_ckpt_interval,
            save_on_train_epoch_end=True,
            save_optim_on_train_end=True,
            always_save_context=False,
            filename="{step}",
            enable_version_counter=False,
            use_in_cluster_local_ckpts=False,
            is_persistent_storage=True,
            enable_high_scale_ckpt=False,
            preprocess_files=True,
            priority=1,
            delete_unfinished_ckpt_on_start=True,
        )
    )

  # Need to sort callbacks so that priority is respected
  callbacks.sort(key=lambda cb: getattr(cb, "priority", 100), reverse=True)

  if args.enable_fault_tolerance:
    callbacks.append(get_ft_callback(log_dir, args.sim_fault_desc))
  if args.enable_gc:
    callbacks.append(
        GarbageCollectionCallback(gc_interval_train=100, gc_interval_val=100)
    )

  plugins = [nl.MegatronMixedPrecision(precision="bf16-mixed")]
  checkpoint_io = DistributedCheckpointIO(
      save_ckpt_format="torch_dist",
      load_directly_on_device=True,
      async_save=args.enable_async_ckpt or args.enable_optimized_async_ckpt,
      torch_dist_multiproc=args.ckpt_threads_per_rank,
      assume_constant_structure=True,
      parallel_save=True,
      parallel_save_within_dp=False,
      parallel_load=True,
  )
  if args.enable_optimized_async_ckpt:
    if args.local_ckpt_dir is not None:
      checkpoint_io = CombinedCheckpointIO(
          save_ckpt_format="torch_dist",
          load_directly_on_device=True,
          async_save=args.enable_async_ckpt or args.enable_optimized_async_ckpt,
          torch_dist_multiproc=args.ckpt_threads_per_rank,
          assume_constant_structure=True,
          persistent_parallel_save=True,
          persistent_parallel_save_within_dp=False,
          persistent_parallel_load=True,
          local_parallel_save=False,
          local_parallel_save_within_dp=False,
          local_parallel_load=False,
          use_ckpt_load_replication=True,
          local_ckpt_dir=args.local_ckpt_dir,
      )
    else:
      checkpoint_io = MinCkptOverheadCheckpointIO(
          save_ckpt_format="torch_dist",
          load_directly_on_device=True,
          async_save=args.enable_async_ckpt or args.enable_optimized_async_ckpt,
          torch_dist_multiproc=args.ckpt_threads_per_rank,
          assume_constant_structure=True,
          parallel_save=True,
          parallel_save_within_dp=False,
          parallel_load=True,
      )
    checkpoint_io = PersistentCheckpointProcessIO(checkpoint_io)

  elif args.enable_async_ckpt:
    checkpoint_io = AsyncFinalizableCheckpointIO(checkpoint_io)

  if args.enable_dist_ckpt:
    plugins.append(checkpoint_io)

  trainer = nl.Trainer(
      accelerator="gpu",
      accumulate_grad_batches=1,
      callbacks=callbacks,
      devices=8,
      limit_test_batches=None,
      limit_val_batches=None,
      log_every_n_steps=None,
      max_steps=args.max_steps,
      max_time={"seconds": args.max_runtime},
      num_nodes=args.num_nodes,
      plugins=plugins,
      strategy=strategy,
      use_distributed_sampler=False,
      val_check_interval=2000,
      enable_progress_bar=False,
  )
  trainer._checkpoint_connector = checkpoint_connector.CheckpointConnector(
      trainer=trainer,
      local_ckpt_dir=local_ckpt_dir,
      persistent_ckpt_dir=persistent_ckpt_dir,
      use_ckpt_load_replication=True,
  )
  data = MockDataModule(
      seq_length=model_config.config.seq_length,
      global_batch_size=gbs,
      num_train_samples=10_000_000_000,
      pin_memory=False,
      micro_batch_size=1,
      tokenizer=SentencePieceTokenizer(model_path=args.tokenizer_path),
  )
  opt_config = OptimizerConfig(
      optimizer="adam",
      lr=1e-2,
      weight_decay=0.1,
      adam_beta1=0.9,
      adam_beta2=0.95,
      adam_eps=1e-8,
      clip_grad=1.0,
      log_num_zeros_in_grad=False,
      timers=None,
      bf16=True,
      use_distributed_optimizer=True,
  )
  optim = MegatronOptimizerModule(config=opt_config)

  llm.train(
      model=mix_model,
      data=data,
      trainer=trainer,
      log=nl.NeMoLogger(
          log_dir=log_dir,
          use_datetime_version=False,
          update_logger_directory=True,
          wandb=None,
      ),
      resume=None,
      optim=optim,
      tokenizer="data",
  )


if __name__ == "__main__":
  main()
