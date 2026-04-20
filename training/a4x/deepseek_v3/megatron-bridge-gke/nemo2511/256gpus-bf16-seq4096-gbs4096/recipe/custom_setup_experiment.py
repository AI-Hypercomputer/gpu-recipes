import glob
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import nemo_run as run
from nemo_run.config import get_nemorun_home


try:
  from argument_parser import parse_cli_args
  from utils.evaluate import calc_convergence_and_performance
  from utils.executors import dgxc_executor, slurm_executor
except (ImportError, ModuleNotFoundError):
  from .argument_parser import parse_cli_args
  from .utils.evaluate import calc_convergence_and_performance
  from .utils.executors import dgxc_executor, slurm_executor

try:
  import wandb

  HAVE_WANDB = True
except (ImportError, ModuleNotFoundError):
  HAVE_WANDB = False

try:
  from perf_plugins import NsysPlugin, PerfEnvPlugin
  from resiliency_plugins import FaultTolerancePlugin
except (ImportError, ModuleNotFoundError):
  from .perf_plugins import NsysPlugin, PerfEnvPlugin
  from .resiliency_plugins import FaultTolerancePlugin

import logging


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
    moe_a2a_overlap: bool,
    tp_size: Optional[int],
    pp_size: Optional[int],
    cp_size: Optional[int],
    wandb_key: str,
    wandb_project_name: str,
    wandb_experiment_name: str,
    wandb_entity_name: str,
    profiling_start_step: int,
    profiling_stop_step: int,
    profiling_gpu_metrics: bool,
    profiling_ranks: Optional[List[int]],
    nemo_home: str,
    account: str,
    partition: str,
    log_dir: str,
    gpus_per_node: int,
    time_limit: str,
    container_image: str,
    custom_mounts: List[str],
    custom_env_vars: List[str],
    custom_srun_args: List[str],
    pretrained_checkpoint: Optional[str],
    num_gpus: int,
    is_long_convergence_run: bool,
    additional_slurm_params: Optional[Dict[str, Any]],
    golden_values_path: str,
    convergence_params: Dict[str, Any],
    performance_params: Dict[str, Any],
    max_retries: int,
    dgxc_base_url: str,
    dgxc_cluster: str,
    dgxc_kube_apiserver_url: str,
    dgxc_app_id: str,
    dgxc_app_secret: str,
    dgxc_project_name: str,
    dgxc_pvc_claim_name: str,
    dgxc_pvc_mount_path: str,
):
    rank = os.environ['RANK']

    exp_name = f"{model_recipe_name}_{model_family_name}"
    exp_name += f'_worker{rank}'
    if use_recipes:
        script_name = ENTRYPOINT_RECIPE

    else:
        script_name = ENTRYPOINT_PEFORMANCE

    run_script_path = SCRIPT_DIR / script_name
    logger.info(f"Run script path: {run_script_path}")
    if not run_script_path.is_file():
        logger.error(f"Specified run script not found: {run_script_path}")
        sys.exit(1)

    nemorun_script = run.Script(
        path=str(run_script_path),
        entrypoint="python",
        env={"PYTHONPATH": f"{SCRIPT_DIR}:$PYTHONPATH"},
        args=list(sys.argv[1:]),
    )

    plugins = []

    if not use_recipes:
        plugins.append(
            PerfEnvPlugin(
                enable_vboost=enable_vboost,
                moe_a2a_overlap=moe_a2a_overlap,
                tp_size=tp_size,
                pp_size=pp_size,
                cp_size=cp_size,
                model_family_name=model_family_name,
                model_recipe_name=model_recipe_name,
                gpu=gpu,
                compute_dtype=compute_dtype,
                train_task=task,
            )
        )

    if enable_nsys:
        plugins.append(
            NsysPlugin(
                profile_step_start=profiling_start_step,
                profile_step_end=profiling_stop_step,
                nsys_gpu_metrics=profiling_gpu_metrics,
                profile_ranks=profiling_ranks,
            )
        )

    executor = run.LocalExecutor()
    run.run(
        nemorun_script,
        executor=executor,
        plugins=plugins,
        dryrun=False,
        detach=False,
        name=exp_name,
    )


if __name__ == "__main__":
    parser = parse_cli_args()
    args, unknown_args = parser.parse_known_args()

    # probably better to use parser.parse_args() and make unknowns an error,
    # but for now we'll just issue a warning.
    if unknown_args:
        logger.warning(f"Ignoring unrecognized arguments: {' '.join(unknown_args)}")

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
        moe_a2a_overlap=args.moe_a2a_overlap,
        tp_size=args.tensor_model_parallel_size,
        pp_size=args.pipeline_model_parallel_size,
        cp_size=args.context_parallel_size,
        wandb_key=args.wandb_key,
        wandb_project_name=args.wandb_project_name,
        wandb_experiment_name=args.wandb_experiment_name,
        wandb_entity_name=args.wandb_entity_name,
        profiling_start_step=args.profiling_start_step,
        profiling_stop_step=args.profiling_stop_step,
        profiling_gpu_metrics=args.profiling_gpu_metrics,
        profiling_ranks=args.profiling_ranks,
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
        max_retries=args.max_retries,
        dgxc_base_url=args.dgxc_base_url,
        dgxc_cluster=args.dgxc_cluster,
        dgxc_kube_apiserver_url=args.dgxc_kube_apiserver_url,
        dgxc_app_id=args.dgxc_app_id,
        dgxc_app_secret=args.dgxc_app_secret,
        dgxc_project_name=args.dgxc_project_name,
        dgxc_pvc_claim_name=args.dgxc_pvc_claim_name,
        dgxc_pvc_mount_path=args.dgxc_pvc_mount_path,
    )
