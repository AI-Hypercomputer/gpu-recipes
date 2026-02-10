"""Nemo2 pretraining recipe for Llama 3.1 405B model."""

# hack for relative imports

from os.path import basename, splitext
import random
import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
from nemo.collections.llm.recipes import llama31_405b
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin, PerfEnvPlugin
from nemo.utils.loggers.dllogger import DLLogger
import nemo_run as run
from scripts.performance.argument_parser import parse_cli_args
from scripts.performance.helpers import args_sanity_check
from scripts.performance.helpers import get_user_configs
from scripts.performance.helpers import set_exp_logging_configs
from scripts.performance.helpers import set_primary_perf_configs
from scripts.performance.utils import get_comm_overlap_callback_idx, hf_tokenizer


def recipe(
    args,
    num_nodes,
    mbs,
    gbs,
    tp_size,
    pp_size,
    cp_size,
    vp_size,
    ep_size,
    enable_cuda_graphs,
    use_mcore_fsdp,
    recompute_layers,
    activation_offload_layers,
) -> run.Partial:
  """Returns a Nemo2 training recipe for Llama 3.1 405B model."""
  # Start from the Nemo standard recipe.
  pretrain = llama31_405b.pretrain_recipe(performance_mode=True)

  pretrain = set_primary_perf_configs(
      pretrain,
      "pre_train",
      num_nodes=num_nodes,
      num_gpus_per_node=4,
      mbs=mbs,
      gbs=gbs,
      max_steps=args.max_steps,
      tp_size=tp_size,
      pp_size=pp_size,
      cp_size=cp_size,
      vp_size=vp_size,
      ep_size=ep_size,
      enable_cuda_graphs=enable_cuda_graphs,
      activation_offload_layers=activation_offload_layers,
      compute_dtype=args.compute_dtype,
      fp8_recipe=args.fp8_recipe,
      nccl_communicator_config_path=args.nccl_communicator_config_path,
      use_mcore_fsdp=use_mcore_fsdp,
      recompute_layers=recompute_layers,
      use_fsdp_double_buffer=args.use_fsdp_double_buffer,
      use_user_buffer_registration=args.use_user_buffer_registration,
      use_sharp=args.use_sharp,
      keep_fsdp_fp8_transpose_cache=args.keep_fsdp_fp8_transpose_cache,
  )
  comm_overlap_callback_idx = get_comm_overlap_callback_idx(
      pretrain.trainer.callbacks
  )
  pretrain.trainer.callbacks[
      comm_overlap_callback_idx
  ].tp_comm_bootstrap_backend = "nccl"

  pretrain = set_exp_logging_configs(
      pretrain,
      "pre_train",
      "llm",
      "llama3",
      args.tensorboard,
      args.wandb,
      args.wandb_prj_name,
      args.wandb_job_name,
  )

  if args.use_hf_tokenizer:
    pretrain.data.tokenizer = hf_tokenizer("meta-llama/Llama-3.1-405B")
  else:
    pretrain.data.tokenizer = run.Config(
        get_nmt_tokenizer,
        library="null",
        model_name="NullTokenizer",
        vocab_size=128256,
    )
  pretrain.model.tokenizer = pretrain.data.tokenizer

  comm_overlap_callback_idx = get_comm_overlap_callback_idx(
      pretrain.trainer.callbacks
  )
  assert (
      comm_overlap_callback_idx is not None
  ), "MegatronCommOverlapCallback missing. Required for performance."

  tp_comm_overlap_cfg = fdl.cast(
      run.Config,
      fdl_dc.convert_dataclasses_to_configs(
          userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
      ),
  )
  pretrain.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = (
      tp_comm_overlap_cfg
  )

  if use_mcore_fsdp:
    pretrain.trainer.strategy.num_distributed_optimizer_instances = (
        num_nodes * 4
    ) // 64

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
  args = parse_cli_args().parse_args()
  args_sanity_check(args)

  kwargs = get_user_configs(
      args.gpu.lower(), "pre_train", "llama31", "405b", args
  )
  (
      num_nodes,
      mbs,
      gbs,
      tp_size,
      pp_size,
      cp_size,
      vp_size,
      ep_size,
      _,
      enable_cuda_graphs,
      use_mcore_fsdp,
      recompute_layers,
      activation_offload_layers,
  ) = kwargs[:13]

  recipe = recipe(
      args,
      num_nodes,
      mbs,
      gbs,
      tp_size,
      pp_size,
      cp_size,
      vp_size,
      ep_size,
      enable_cuda_graphs,
      use_mcore_fsdp,
      recompute_layers,
      activation_offload_layers,
  )

  exp_config = (
      f"{num_nodes}nodes_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_{mbs}mbs_{gbs}gbs-{random.randint(0, 100000)}"
  )
  exp_name = (
      f"{splitext(basename(__file__))[0]}_{args.compute_dtype}_{exp_config}"
  )

  if use_mcore_fsdp:
    # Needed to enable CuDNN LN for FSDP overlap
    env_vars = {"NVTE_NORM_FWD_USE_CUDNN": "1", "NVTE_NORM_BWD_USE_CUDNN": "1"}
  else:
    env_vars = {}

  executor = run.LocalExecutor()

  plugins = [
      PerfEnvPlugin(
          enable_vboost=False,
          nccl_pp_comm_chunksize=2097152 if pp_size > 1 else None,
          gpu_sm100_or_newer=True,
      )
  ]
  if args.enable_nsys:
    plugins.append(
        NsysPlugin(start_step=10, end_step=13, ranks=list(range(0, 1)))
    )
  if args.enable_memory_profile:
    assert args.memory_profile_out_path is not None
    plugins.append(MemoryProfilePlugin(dir=args.memory_profile_out_path))

  with run.Experiment(exp_name) as exp:
    exp.add(
        recipe,
        executor=executor,
        name=exp_name,
        plugins=plugins,
    )

    exp.run(sequential=True, direct=True, detach=False)
