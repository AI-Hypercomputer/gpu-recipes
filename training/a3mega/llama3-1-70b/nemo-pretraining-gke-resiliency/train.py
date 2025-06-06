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
from nemo.collections.llm.recipes.llama31_70b import model as llama_model
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
)
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
from resiliency.plugins._ckpt_utils import find_latest_checkpoint_path
from resiliency.plugins.min_ckpt_overhead import MinCkptOverheadCheckpointIO
from resiliency.plugins.persistent_ckpt_proc import PersistentCheckpointProcessIO
from resiliency.utils import get_resiliency_logger
import torch


resiliency_logger = get_resiliency_logger(__name__)


def get_parser():
  parser = argparse.ArgumentParser(
      description="Mixtral Pretraining script using NeMo 2.0."
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
      "--topk-ckpt",
      type=int,
      default=10,
      help="Number of top checkpoints to keep.",
  )
  parser.add_argument(
      "--checkpoint-interval",
      type=int,
      default=1000,
      help="Number of steps to save a checkpoint.",
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
  # nemo_logging.is_global_rank_zero = lambda: True

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
  strategy = nl.MegatronStrategy(
      tensor_model_parallel_size=4,
      pipeline_model_parallel_size=4,
      pipeline_dtype=torch.bfloat16,
      virtual_pipeline_model_parallel_size=4,
      context_parallel_size=2,
      sequence_parallel=True,
      gradient_as_bucket_view=True,
      ckpt_async_save=args.enable_async_ckpt
      or args.enable_optimized_async_ckpt,
      ckpt_parallel_load=True,
      ddp=DistributedDataParallelConfig(),
  )

  ckpt_dir = Path(args.ckpt_dir) / args.job_name / "checkpoint"
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
  checkpoint_callback = model_checkpoint.ModelCheckpoint(
      dirpath=ckpt_dir,
      save_last=False,
      monitor="step",
      save_top_k=args.topk_ckpt,
      mode="max",
      save_weights_only=False,
      every_n_train_steps=args.checkpoint_interval,
      save_on_train_epoch_end=True,
      save_optim_on_train_end=True,
      always_save_context=False,
      filename="{step}",
      enable_version_counter=False,
      use_in_cluster_local_ckpts=None,
      enable_high_scale_ckpt=False,
  )
  callbacks.append(checkpoint_callback)
  if args.enable_fault_tolerance:
    callbacks.append(get_ft_callback(log_dir, args.sim_fault_desc))
  if args.enable_gc:
    callbacks.append(
        GarbageCollectionCallback(gc_interval_train=100, gc_interval_val=100)
    )
  if args.enable_comm_overlap:
    callbacks.append(
        MegatronCommOverlapCallback(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
            defer_embedding_wgrad_compute=True,
            wgrad_deferral_limit=50,
            overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to an issue with checkpointing
            align_param_gather=True,
        )
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
  else:
    if args.enable_async_ckpt:
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
      persistent_ckpt_dir=ckpt_dir,
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
          ckpt=checkpoint_callback,
      ),
      resume=None,
      optim=optim,
      tokenizer="data",
  )


if __name__ == "__main__":
  main()
