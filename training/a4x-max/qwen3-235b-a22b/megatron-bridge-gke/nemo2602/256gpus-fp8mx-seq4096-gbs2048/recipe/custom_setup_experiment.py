import glob
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import nemo_run as run
from nemo_run.config import get_nemorun_home


try:
    from argument_parser import parse_cli_args
    from utils.evaluate import calc_convergence_and_performance
    from utils.executors import dgxc_executor, slurm_executor
    from utils.utils import get_exp_name_config, select_config_variant_interactive
except (ImportError, ModuleNotFoundError):
    from .argument_parser import parse_cli_args
    from .utils.evaluate import calc_convergence_and_performance
    from .utils.executors import dgxc_executor, slurm_executor
    from .utils.utils import get_exp_name_config, select_config_variant_interactive

try:
    import wandb

    HAVE_WANDB = True
except (ImportError, ModuleNotFoundError):
    HAVE_WANDB = False

try:
    from perf_plugins import NsysPlugin, PerfEnvPlugin, PyTorchProfilerPlugin
    from resiliency_plugins import FaultTolerancePlugin
except (ImportError, ModuleNotFoundError):
    from .perf_plugins import NsysPlugin, PerfEnvPlugin, PyTorchProfilerPlugin
    from .resiliency_plugins import FaultTolerancePlugin


SCRIPT_DIR = Path(__file__).parent.resolve()
ENTRYPOINT_PEFORMANCE = "run_script.py"
ENTRYPOINT_RECIPE = "run_recipe.py"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)




def main(
    use_recipes: bool,
    model_family_name: str,
    model_recipe_name: str,
    task: str,
    compute_dtype: str,
    gpu: str,
    hf_token: str,
    detach: bool,
    dryrun: bool,
    enable_vboost: bool,
    enable_nsys: bool,
    pytorch_profiler: bool,
    moe_a2a_overlap: bool,
    tp_size: Optional[int],
    pp_size: Optional[int],
    cp_size: Optional[int],
    ep_size: Optional[int],
    wandb_key: str,
    wandb_project_name: str,
    wandb_experiment_name: str,
    wandb_entity_name: str,
    profiling_start_step: int,
    profiling_stop_step: int,
    record_memory_history: bool,
    profiling_gpu_metrics: bool,
    profiling_ranks: Optional[List[int]],
    nsys_trace: Optional[List[str]],
    nsys_extra_args: Optional[List[str]],
    nemo_home: str,
    account: str,
    partition: str,
    log_dir: str,
    gpus_per_node: int,
    time_limit: str,
    container_image: str,
    custom_mounts: List[str],
    custom_env_vars: Dict[str, str],
    custom_srun_args: List[str],
    custom_bash_cmds: List[List[str]],
    nccl_ub: bool,
    pretrained_checkpoint: Optional[str],
    num_gpus: int,
    is_long_convergence_run: bool,
    additional_slurm_params: Optional[Dict[str, Any]],
    golden_values_path: str,
    convergence_params: Dict[str, Any],
    performance_params: Dict[str, Any],
    memory_params: Dict[str, Any],
    max_retries: int,
    dgxc_base_url: str,
    dgxc_cluster: str,
    dgxc_kube_apiserver_url: str,
    dgxc_app_id: str,
    dgxc_app_secret: str,
    dgxc_project_name: str,
    dgxc_pvc_claim_name: str,
    dgxc_pvc_mount_path: str,
    config_variant: str = "v1",
):
    """Sets up the experiment and runs it."""
    if (
        model_family_name in ["qwen3"]
        and model_recipe_name
        in [
            "qwen3_30b_a3b",
            "qwen3_235b_a22b",
        ]
        and task == "pretrain"
    ):
        assert hf_token is not None, "HF token is required for Qwen3 tokenizer. NullTokenizer to be used soon."

    if wandb_key is not None:
        assert wandb_project_name is not None and wandb_experiment_name is not None, (
            "both wandb_project_name and wandb_experiment_name are required for logging with WandB"
        )

    if use_recipes:
        script_name = ENTRYPOINT_RECIPE
        exp_name = (
            wandb_experiment_name
            if wandb_experiment_name is not None
            else f"{model_recipe_name}_{task}_{num_gpus}gpu_{gpu}"
        )

    else:
        script_name = ENTRYPOINT_PEFORMANCE
        exp_config = get_exp_name_config(
            args, model_family_name, model_recipe_name, gpu, compute_dtype, task, config_variant
        )
        exp_name = (
            wandb_experiment_name
            if wandb_experiment_name is not None
            else f"{task}_{model_recipe_name}_{compute_dtype}_{exp_config}"
        )

    if pretrained_checkpoint is not None:
        custom_mounts.append(f"{pretrained_checkpoint}:{pretrained_checkpoint}")

    import os
    rank = os.environ.get('RANK', '0')
    exp_name += f'_worker{rank}'

    run_script_path = SCRIPT_DIR / script_name
    logger.info(f"Run script path: {run_script_path}")
    if not run_script_path.is_file():
        logger.error(f"Specified run script not found: {run_script_path}")
        sys.exit(1)

    custom_mounts.extend(
        [
            f"{run_script_path}:{run_script_path}",
            f"{SCRIPT_DIR}:{SCRIPT_DIR}",
        ]
    )

    if nccl_ub:
        custom_env_vars.update({"NCCL_NVLS_ENABLE": "1", "NCCL_CTA_POLICY": "1"})

    executor = run.LocalExecutor()

    plugins = []

    if not use_recipes:
        plugins.append(
            PerfEnvPlugin(
                enable_vboost=enable_vboost,
                moe_a2a_overlap=moe_a2a_overlap,
                tp_size=tp_size,
                pp_size=pp_size,
                cp_size=cp_size,
                ep_size=ep_size,
                model_family_name=model_family_name,
                model_recipe_name=model_recipe_name,
                gpu=gpu,
                compute_dtype=compute_dtype,
                train_task=task,
                config_variant=config_variant,
            )
        )

    if enable_nsys:
        plugins.append(
            NsysPlugin(
                profile_step_start=profiling_start_step,
                profile_step_end=profiling_stop_step,
                nsys_gpu_metrics=profiling_gpu_metrics,
                profile_ranks=profiling_ranks,
                nsys_trace=nsys_trace,
                nsys_extra_args=nsys_extra_args,
            )
        )
    if pytorch_profiler:
        plugins.append(
            PyTorchProfilerPlugin(
                profile_step_start=profiling_start_step,
                profile_step_end=profiling_stop_step,
                profile_ranks=profiling_ranks,
                record_memory_history=record_memory_history,
            )
        )

    nemorun_script = run.Script(
        path=str(run_script_path),
        entrypoint="python",
        env={"PYTHONPATH": f"{SCRIPT_DIR}:$PYTHONPATH"},
        args=list(sys.argv[1:]),
    )

    logger.info("Will launch the following command with Nemo-Run: %s", " ".join(nemorun_script.to_command()))

    run.run(
        nemorun_script,
        executor=executor,
        plugins=plugins,
        dryrun=dryrun,
        detach=detach,
        name=exp_name,
    )


if __name__ == "__main__":
    parser = parse_cli_args()
    args, unknown_args = parser.parse_known_args()

    assert not (args.enable_nsys and args.pytorch_profiler), (
        "Both NSys and PyTorch profiler cannot be enabled at the same time"
    )

    # probably better to use parser.parse_args() and make unknowns an error,
    # but for now we'll just issue a warning.
    if unknown_args:
        logger.warning(f"Ignoring unrecognized arguments: {' '.join(unknown_args)}")

    # Handle --list_config_variants: show available variants and interactively select
    config_variant = args.config_variant
    if args.list_config_variants:
        config_variant = select_config_variant_interactive(
            model_family_name=args.model_family_name,
            model_recipe_name=args.model_recipe_name,
            gpu=args.gpu,
            compute_dtype=args.compute_dtype,
            task=args.task,
        )

    main(
        use_recipes=args.use_recipes,
        model_family_name=args.model_family_name,
        model_recipe_name=args.model_recipe_name,
        task=args.task,
        compute_dtype=args.compute_dtype,
        gpu=args.gpu,
        hf_token=args.hf_token,
        detach=args.detach,
        dryrun=args.dryrun,
        enable_vboost=args.enable_vboost,
        enable_nsys=args.enable_nsys,
        pytorch_profiler=args.pytorch_profiler,
        moe_a2a_overlap=args.moe_a2a_overlap,
        tp_size=args.tensor_model_parallel_size,
        pp_size=args.pipeline_model_parallel_size,
        cp_size=args.context_parallel_size,
        ep_size=args.expert_model_parallel_size,
        wandb_key=args.wandb_key,
        wandb_project_name=args.wandb_project_name,
        wandb_experiment_name=args.wandb_experiment_name,
        wandb_entity_name=args.wandb_entity_name,
        profiling_start_step=args.profiling_start_step,
        profiling_stop_step=args.profiling_stop_step,
        record_memory_history=args.record_memory_history,
        profiling_gpu_metrics=args.profiling_gpu_metrics,
        profiling_ranks=args.profiling_ranks,
        nsys_trace=args.nsys_trace,
        nsys_extra_args=args.nsys_extra_args,
        nemo_home=args.nemo_home,
        account=args.account,
        partition=args.partition,
        log_dir=args.log_dir,
        gpus_per_node=args.gpus_per_node,
        time_limit=args.time_limit,
        container_image=args.container_image,
        custom_mounts=args.custom_mounts,
        custom_env_vars=args.custom_env_vars,
        custom_srun_args=args.custom_srun_args,
        custom_bash_cmds=args.custom_bash_cmds,
        nccl_ub=args.nccl_ub,
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_gpus=args.num_gpus,
        is_long_convergence_run=args.is_long_convergence_run,
        additional_slurm_params=args.additional_slurm_params,
        golden_values_path=args.golden_values_path,
        convergence_params={
            "correlation_threshold": args.correlation_threshold,
            "high_loss_tolerance": args.high_loss_tolerance,
            "medium_loss_tolerance": args.medium_loss_tolerance,
            "low_loss_tolerance": args.low_loss_tolerance,
            "final_loss_tolerance": args.final_loss_tolerance,
            "max_outlier_ratio": args.max_outlier_ratio,
            "outlier_threshold": args.outlier_threshold,
            "skip_first_percent_loss": args.skip_first_percent_loss,
        },
        performance_params={
            "timing_threshold": args.timing_threshold,
            "skip_first_percent_time": args.skip_first_percent_time,
        },
        memory_params={
            "memory_threshold": args.memory_threshold,
        },
        max_retries=args.max_retries,
        dgxc_base_url=args.dgxc_base_url,
        dgxc_cluster=args.dgxc_cluster,
        dgxc_kube_apiserver_url=args.dgxc_kube_apiserver_url,
        dgxc_app_id=args.dgxc_app_id,
        dgxc_app_secret=args.dgxc_app_secret,
        dgxc_project_name=args.dgxc_project_name,
        dgxc_pvc_claim_name=args.dgxc_pvc_claim_name,
        dgxc_pvc_mount_path=args.dgxc_pvc_mount_path,
        config_variant=config_variant,
    )