"""Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from collections import namedtuple
from dataclasses import dataclass

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
import nemo
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.llm import GemmaConfig2B, GemmaModel
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model.llama import Llama3Config, LlamaModel
from nemo.lightning.io.pl import MegatronCheckpointIO
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
import torch

if nemo.__version__.startswith("2.1.0"):
  import lightning.pytorch as pl
else:
  import pytorch_lightning as pl
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO
from nemo.utils.callbacks.dist_ckpt_io import DistributedCheckpointIO
from nemo.utils.exp_manager import TimingCallback  # , DeltaTimingCallback
import logging
from nemo.utils import logging as nemo_logging
from torch.distributed import TCPStore
import datetime
import signal
import os, sys
import logging  # Configure logging
import torch.distributed as dist
from pathlib import Path
from typing import Optional
from collections import defaultdict
from nemo.lightning import io
from nemo.collections.llm.gpt.model.llama import Llama31Config405B, Llama31Config70B, Llama31Config8B
from nemo.lightning.pytorch.callbacks import GarbageCollectionCallback
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
)
from resiliency.callbacks import comm_overlap, model_checkpoint
from resiliency.callbacks.profile import ProfileCheckpointCallback
from resiliency.callbacks.logging import StepLoggingCallback, TPSLoggingCallback
from nemo.utils.mcore_logger import add_handlers_to_mcore_logger
from resiliency.utils import test_all_reduce, get_resiliency_logger, SingleLetterFormatter
from resiliency.model import get_model_config
from nvidia_resiliency_ext.ptl_resiliency import FaultToleranceCallback
from nvidia_resiliency_ext.ptl_resiliency.fault_tolerance_callback import SimulatedFaultParams
import resiliency.high_scale_ckpt_utils as high_scale_ckpt_utils
from resiliency.plugins._ckpt_utils import get_is_checkpoint_file_handler, find_latest_checkpoint_path
from resiliency.connectors import checkpoint_connector
from resiliency.plugins.replication_utils import ReplicatedOptimizerMegatronStrategy

resiliency_logger = get_resiliency_logger(__name__)


@dataclass(kw_only=True)
class AutoResume(nl.AutoResume):

  use_ckpt_load_replication: bool = False

  def _find_trainer_ckpt_path(self) -> Optional[Path]:
    from nemo.utils.exp_manager import NotFoundError, _filter_out_unfinished_checkpoints
    from nemo.utils.app_state import AppState
    import re

    app_state = AppState()
    log_dir = app_state.log_dir

    checkpoint = None

    checkpoint_dir = (
        Path(self.resume_from_directory)
        if self.resume_from_directory
        else Path(Path(log_dir) / "checkpoints")
    )
    resume_ckpt_path = find_latest_checkpoint_path(
        checkpoint_dir=checkpoint_dir,
        synchronize=self.use_ckpt_load_replication,
    )

    return resume_ckpt_path


def get_trainer(args, callbacks, world_size, parallel_config, trace_dir):
  if args.num_optimizer_replicas > 1:
    # Supports replicated distributed optimizer
    strategy = ReplicatedOptimizerMegatronStrategy(
        tensor_model_parallel_size=parallel_config.tp,
        pipeline_model_parallel_size=parallel_config.pp,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=parallel_config.vp,
        context_parallel_size=parallel_config.cp,
        sequence_parallel=parallel_config.tp > 1,
        ckpt_async_save=args.enable_async_ckpt
        or args.enable_optimized_async_ckpt,
        ckpt_parallel_load=True,
        ddp=DistributedDataParallelConfig(
            num_distributed_optimizer_instances=args.num_optimizer_replicas
        ),
        progress_interval=1,
    )
  else:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=parallel_config.tp,
        pipeline_model_parallel_size=parallel_config.pp,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=parallel_config.vp,
        context_parallel_size=parallel_config.cp,
        sequence_parallel=parallel_config.tp > 1,
        ckpt_async_save=args.enable_async_ckpt
        or args.enable_optimized_async_ckpt,
        ckpt_parallel_load=True,
        ddp=DistributedDataParallelConfig(),
        progress_interval=1,
    )

  if args.enable_optimized_async_ckpt:
    if args.enable_in_cluster_local_ckpt:
      from resiliency.plugins.in_cluster_local_ckpt import InClusterLocalCheckpointIO

      checkpoint_io = InClusterLocalCheckpointIO(
          save_ckpt_format="torch_dist",
          load_directly_on_device=True,
          async_save=args.enable_async_ckpt or args.enable_optimized_async_ckpt,
          torch_dist_multiproc=args.ckpt_threads_per_rank,
          assume_constant_structure=True,
          parallel_save=False,
          parallel_save_within_dp=False,
          parallel_load=False,
          use_ckpt_load_replication=True
          if args.enable_ckpt_load_replication
          else False,
      )

    else:
      from resiliency.plugins.min_ckpt_overhead import MinCkptOverheadCheckpointIO

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

    from resiliency.plugins.persistent_ckpt_proc import PersistentCheckpointProcessIO

    checkpoint_io = PersistentCheckpointProcessIO(
        checkpoint_io,
        profile_dir=trace_dir
        if args.profile_checkpoint_interval is not None
        else None,
    )

  else:
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
    if args.enable_async_ckpt:
      checkpoint_io = AsyncFinalizableCheckpointIO(checkpoint_io)

  plugins = [nl.MegatronMixedPrecision(precision="bf16-mixed")]
  if args.enable_dist_ckpt:
    plugins.append(checkpoint_io)
  trainer = nl.Trainer(
      accelerator="gpu",
      devices=args.num_gpus,
      num_nodes=args.num_nodes,
      max_steps=args.max_steps,
      max_time={"seconds": args.max_runtime},
      callbacks=callbacks,
      log_every_n_steps=None,
      val_check_interval=None,
      limit_val_batches=None,
      plugins=plugins,
      strategy=strategy,
      enable_progress_bar=False,
  )
  trainer._checkpoint_connector = checkpoint_connector.CheckpointConnector(
      trainer=trainer,
      enable_high_scale_ckpt=args.enable_high_scale_ckpt,
      use_ckpt_load_replication=args.enable_ckpt_load_replication,
  )
  return trainer


def get_parser():
  parser = argparse.ArgumentParser(
      description="Llama3 Pretraining script using NeMo 2.0."
  )

  parser.add_argument(
      "--tokenizer-path",
      type=str,
      default="tokenizer.model",
      help="Path to the tokenizer model file.",
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
  parser.add_argument("--max-runtime", type=int, default=900)  # in seconds
  parser.add_argument(
      "--ckpt-threads-per-rank",
      type=int,
      default=2,
      help="Number of threads to use for writing checkpoint files per rank.",
  )
  parser.add_argument(
      "--max-steps",
      type=int,
      default=1_000_000,
      help="Number of steps to run the training for.",
  )
  parser.add_argument(
      "--checkpoint-interval",
      type=int,
      default=80,
      help="Checkpoint saving interval in steps.",
  )
  parser.add_argument(
      "--profile-checkpoint-interval",
      type=int,
      default=None,
      help="Checkpoint profiling interval in steps.",
  )
  parser.add_argument(
      "--val-check-interval",
      type=int,
      default=40,
      help="Validation check interval in steps.",
  )
  parser.add_argument(
      "--limit-val-batches",
      type=int,
      default=10,
      help="How many batches to use for validation.",
  )
  parser.add_argument(
      "--global-bs",
      type=int,
      default=None,
      help="Global batch size.",
  )
  parser.add_argument(
      "--topk-ckpt",
      type=int,
      default=10,
      help="Number of top checkpoints to keep.",
  )
  parser.add_argument(
      "--log-dir",
      type=str,
      help="Output log dir.",
      required=False,
      default="/log/",
  )
  parser.add_argument(
      "--log-to-remote-storage",
      action="store_true",
      help=(
          "Enable logging to remote storage log dir, otherwise it will log to"
          " /tmp/ folder."
      ),
      default=False,
  )
  parser.add_argument(
      "--job-name",
      type=str,
      help="Job name of the current run.",
      required=False,
      default="test_job",
  )
  parser.add_argument(
      "--model",
      type=str,
      choices=["36M", "2B", "8B", "70B", "70Bt", "405B", "405Bt"],
      help="Model size to use for training.",
      required=False,
      default="36M",
  )
  parser.add_argument(
      "--num-optimizer-replicas",
      type=int,
      help=(
          "Number of times optimizer is replicated. When using loading"
          " replication, this should be > 1."
      ),
      required=False,
      default="1",
  )

  parser.add_argument(
      "--enable-async-ckpt",
      action="store_true",
      help="Enable async checkpointing.",
      default=False,
  )
  parser.add_argument(
      "--enable-optimized-async-ckpt",
      action="store_true",
      help="Enable optimized async checkpointing.",
      default=False,
  )
  parser.add_argument(
      "--enable-in-cluster-local-ckpt",
      action="store_true",
      help=(
          "Enable in-cluster local checkpointing. Must be used with"
          " `--enable-optimized-async-ckpt`."
      ),
      default=False,
  )
  parser.add_argument(
      "--enable-ckpt-load-replication",
      action="store_true",
      help=(
          "Enable checkpoint load replication. Must be used with"
          " `--enable-in-cluster-local-ckpt` and `--num-optimizer-replicas>=2`."
      ),
      default=False,
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
      "--enable-high-scale-ckpt",
      action="store_true",
      help=(
          "Enable High Scale Checkpointing. Must be used with"
          " `--enable-in-cluster-local-ckpt`."
      ),
      default=False,
  )
  parser.add_argument(
      "--enable-fault-tolerance",
      action="store_true",
      help="Enable nvrx fault tolerance.",
      default=False,
  )
  parser.add_argument(
      "--enable-tensorboard",
      action="store_true",
      help="Enable tensorboard logging.",
      default=False,
  )
  parser.add_argument(
      "--trace-name",
      type=str,
      help="Name of the trace file.",
      required=False,
      default=None,
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
      "--log-level",
      type=str,
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
      help="Set the logging level for the script.",
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
  resiliency_logger.info("First Line of main func.")

  # Get command line arguments
  args = get_parser().parse_args()

  # Torchrun env vars
  rank = int(os.environ["RANK"])
  local_rank = int(os.environ["LOCAL_RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  # Set global logging level
  logging.basicConfig(level=args.log_level)

  nemo_logging.is_global_rank_zero = lambda: True

  pl_logger = logging.getLogger("lightning.pytorch")
  megatron_logger = logging.getLogger("megatron")
  add_handlers_to_mcore_logger()

  # Define a new log format
  formatter = SingleLetterFormatter(
      "[Lightning %(levelname)s %(asctime)s %(filename)s:%(lineno)d]"
      " %(message)s",
      "%Y-%m-%d %H:%M:%S",
  )

  while pl_logger.handlers:
    pl_logger.removeHandler(pl_logger.handlers[0])

  # Create a console handler with the new format
  console_handler = logging.StreamHandler()
  console_handler.setFormatter(formatter)

  # Add the new handler to the logger
  pl_logger.addHandler(console_handler)

  ckpt_dir = Path(args.log_dir) / args.job_name / "checkpoint"
  # stream logging files to gcsfuse impacting gcsfuse performane for ckpting
  # loging to /tmp folder by default
  log_file_dir = f"/tmp/{args.job_name}/log"
  if args.log_to_remote_storage:
    log_file_dir = Path(args.log_dir) / args.job_name / "log"
  trace_dir = Path(args.log_dir) / args.job_name / "profile"

  torch.cuda.set_device(local_rank)
  dist.init_process_group(backend="nccl")

  if args.enable_ckpt_load_replication and args.num_optimizer_replicas <= 1:
    raise ValueError(
        "Checkpoint load replication requires num_optimizer_replicas > 1."
    )

  if args.enable_high_scale_ckpt:
    if get_is_checkpoint_file_handler(args.enable_high_scale_ckpt):
      assert (
          args.enable_in_cluster_local_ckpt
      ), "in cluster local ckpt should be enabled to use high scale ckpt."
      resiliency_logger.info("Init high scale ckpt...")
      ckpt_dir = Path(high_scale_ckpt_utils.CHECKPOINT_FOLDER_PATH)
      high_scale_ckpt_utils.init_high_scale_ckpt(
          ckpt_dir, args.job_name, blocking=False
      )

    # barrier for high scale ckpt ops
    dist.barrier()

  tracer = None
  if args.trace_name is not None and rank == 0:
    from viztracer import VizTracer

    tracer = VizTracer(
        tracer_entries=500_000_000, max_stack_depth=50, log_torch=True
    )
    tracer.start()

  if not test_all_reduce(rank, local_rank, world_size):
    sys.exit("Simple all reduce test is not passed.")
  resiliency_logger.info("All Reduce test passed.")

  mbs = 1
  gbs = mbs * args.num_gpus * args.num_nodes
  if args.global_bs is not None:
    gbs = (args.global_bs // (mbs * args.num_gpus * args.num_nodes)) * (
        mbs * args.num_gpus * args.num_nodes
    )
    assert gbs > 0

  if args.num_optimizer_replicas > 1:
    if args.model in ["36M", "70B", "405B"]:
      args.model += "ReplicatedOpt"

  model_config = get_model_config(args.model)

  if args.num_optimizer_replicas > 1:
    assert (
        model_config.parallel_config.cp == 1
    ), "Megatron does not support CP with replicated optimizer."

  model = model_config.create_model()

  data = MockDataModule(
      seq_length=model_config.model_config.seq_length,
      global_batch_size=gbs,
      num_train_samples=10_000_000_000,
      pin_memory=False,
      micro_batch_size=mbs,
      tokenizer=SentencePieceTokenizer(model_path=args.tokenizer_path),
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
      use_in_cluster_local_ckpts=args.enable_in_cluster_local_ckpt,
      enable_high_scale_ckpt=args.enable_high_scale_ckpt,
  )

  if args.enable_tensorboard:
    tb_dir = f"gs://{os.getenv('GCS_FUSE_BUCKET')}/nemo-experiments/{args.job_name}/tb"
  else:
    tb_dir = None

  callbacks = [
      checkpoint_callback,
      StepLoggingCallback(tb_dir),
      TPSLoggingCallback(
          gbs=gbs,
          seq_length=model_config.model_config.seq_length,
          tb_dir=tb_dir,
      ),
  ]
  for cb in callbacks:
    assert isinstance(cb, pl.Callback), f"{type(cb)}"
  if args.enable_comm_overlap:
    callbacks.append(model_config._create_comm_overlap_callback())
  if args.enable_gc:
    callbacks.append(
        GarbageCollectionCallback(gc_interval_train=100, gc_interval_val=100)
    )
  if args.profile_checkpoint_interval is not None:
    assert args.profile_checkpoint_interval % args.checkpoint_interval == 0
    callbacks.append(
        ProfileCheckpointCallback(trace_dir, args.profile_checkpoint_interval)
    )
  if args.enable_fault_tolerance:
    callbacks.append(get_ft_callback(log_file_dir, args.sim_fault_desc))
  trainer = get_trainer(
      args, callbacks, world_size, model_config.parallel_config, trace_dir
  )

  nemo_logger = nl.NeMoLogger(
      log_dir=log_file_dir,
      use_datetime_version=False,
      update_logger_directory=True,
      wandb=None,
      ckpt=checkpoint_callback,
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
      # reload check with distributed true/false will fail
  )
  optim = MegatronOptimizerModule(config=opt_config)
  # trainer.save_checkpoint('./test/ckpt')

  llm.train(
      model=model,
      data=data,
      trainer=trainer,
      log=nemo_logger,
      resume=AutoResume(
          resume_from_directory=ckpt_dir,
          resume_if_exists=True,
          resume_ignore_no_checkpoint=True,
          use_ckpt_load_replication=args.enable_ckpt_load_replication,
      ),
      optim=optim,
      tokenizer="data",
  )
  dist.barrier()
  dist.destroy_process_group()
  if args.trace_name is not None and rank == 0:
    tracer.stop()
    tracer.save(
        output_file=f"{trace_dir}/{args.trace_name}_rank{rank}_trace.json"
    )


if __name__ == "__main__":
  main()
