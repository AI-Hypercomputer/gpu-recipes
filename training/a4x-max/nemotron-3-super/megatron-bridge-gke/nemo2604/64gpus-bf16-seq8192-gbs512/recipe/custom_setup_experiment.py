import glob
import logging
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import nemo_run as run
from nemo_run.config import get_nemorun_home


try:
    from argument_parser import NUM_GPUS_PER_NODE_MAP, parse_cli_args
    from utils.evaluate import calc_convergence_and_performance
    from utils.executors import dgxc_executor, kubeflow_executor, slurm_executor
    from utils.utils import get_exp_name_config, select_config_variant_interactive
except (ImportError, ModuleNotFoundError):
    from .argument_parser import NUM_GPUS_PER_NODE_MAP, parse_cli_args
    from .utils.evaluate import calc_convergence_and_performance
    from .utils.executors import dgxc_executor, kubeflow_executor, slurm_executor
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
logger.setLevel(logging.DEBUG)  # pin level so nemo_run's WARNING root doesn't suppress INFO


def check_training_finished(log_file_paths: List[str], is_long_convergence_run: bool = True) -> bool:
    """Check if training is finished.

    For long convergence runs, returns True when a clean-exit marker is found in the logs.
    For normal runs, returns True when the last logged iteration matches the total number
    of iterations (catches jobs that completed all training steps but hung on teardown
    before the job reached SUCCEEDED status).
    """
    found_exit_marker = False
    max_iter_seen = 0
    total_iters = None

    for log_path in log_file_paths:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                if (
                    "StopIteration" in line
                    or "after training is done" in line
                    or "exiting program at iteration" in line
                    or "AssertionError: no samples left to consume:" in line
                ):
                    found_exit_marker = True

                m = re.search(r"iteration\s+(\d+)/\s*(\d+)", line)
                if m:
                    current, total = int(m.group(1)), int(m.group(2))
                    max_iter_seen = max(max_iter_seen, current)
                    total_iters = total

    if is_long_convergence_run:
        return found_exit_marker

    return total_iters is not None and max_iter_seen >= total_iters


def check_slurm_timeout(log_file_path: str) -> bool:
    """Check if Slurm job timed out."""
    with open(log_file_path, "r") as f:
        log_lines = f.readlines()
    log = "\n".join(log_lines)
    return "DUE TO TIME LIMIT" in log


def is_flaky_failure(log_file_path: str) -> bool:
    """Check if Slurm job failed due to flaky failure."""
    with open(log_file_path, "r") as f:
        log_lines = f.readlines()
    log = "\n".join(log_lines)

    return (
        "The server socket has failed to listen on any local network address." in log
        or "Some NCCL operations have failed or timed out." in log
        or "uncorrectable ECC error encountered" in log
        or "illegal memory access" in log
        or "illegal instruction" in log
        or "torch.distributed.DistNetworkError" in log
        or "Segmentation fault" in log
        or "found NaN in" in log
        or "For debugging consider passing CUDA_LAUNCH_BLOCKING=1" in log
        or "double free or corruption" in log
        or "Call to CUDA function failed." in log
        or "Connection reset by peer" in log
        or "invalid pointer" in log
        or "malloc(): unaligned tcache chunk detected" in log
        or "zmq.error.ZMQError: Address already in use" in log
        or "We couldn't connect to 'https://huggingface.co'" in log
        or "Unpack failed: incomplete input" in log
        or "unspecified launch failure" in log
        or "free(): corrupted unsorted chunks" in log
        or "Segfault encountered" in log
        or "Fatal glibc error" in log
        or "EOFError: No data left in file" in log
    )


def build_performance_config(args) -> Optional[Dict[str, Any]]:
    """Build performance configuration from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dictionary with performance configuration or None if performance is disabled
    """
    config = {}

    performance_params = {
        "timing_threshold": args.timing_threshold,
        "skip_first_percent_time": args.skip_first_percent_time,
        "eval_time_start_step": args.eval_time_start_step,
        "eval_time_end_step": args.eval_time_end_step,
    }

    for key, value in performance_params.items():
        if value is not None:
            config[key] = value

    return config if config else None


def ensure_logs_where_written(log_file_paths: List[str]):
    """Ensure logs were written to disk."""
    if len(log_file_paths) == 0:
        raise FileNotFoundError(
            f"Unexpected number of log files found: {log_file_paths}. Expected at least 1, got {len(log_file_paths)}"
        )


def get_job_dir_and_status_from_run(exp_name: str):
    """Get job directory and status from run."""
    result_dict = run.Experiment.from_title(exp_name).status(return_dict=True)
    _, job_dict = list(result_dict.items())[0]
    job_dir = job_dict["local_dir"]
    job_status = str(job_dict["status"])
    return job_dir, job_status


def maybe_increase_n_attempts_on_flaky_failure(
    n_attempts: int,
    max_retries: int,
    is_finished_experiment: bool,
    is_long_convergence_run: bool,
    log_file_paths: List[str],
):
    """Maybe increase number of attempts."""
    if not is_finished_experiment and not is_long_convergence_run:
        if is_flaky_failure(log_file_paths[-1]):
            n_attempts += 1
        else:
            n_attempts = max_retries  # On non-flaky failures, we don't need to restart the experiment.

    return n_attempts


def main(
    use_recipes: bool,
    model_family_name: str,
    model_recipe_name: str,
    task: str,
    compute_dtype: str,
    gpu: str,
    hf_token: str,
    offline: bool,
    detach: bool,
    dryrun: bool,
    enable_vboost: bool,
    enable_nsys: bool,
    pytorch_profiler: bool,
    moe_a2a_overlap: bool,
    tp_size: Optional[int],
    pp_size: Optional[int],
    cp_size: Optional[int],
    vp_size: Optional[int],
    ep_size: Optional[int],
    etp_size: Optional[int],
    micro_batch_size: Optional[int],
    global_batch_size: Optional[int],
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
    kubeflow_namespace: str,
    kubeflow_workdir_pvc: str,
    kubeflow_workdir_pvc_path: str,
    kubeflow_image_pull_secrets: List[str],
    config_variant: str = "v1",
    gres: Optional[str] = None,
    packager: str = "git",
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
        assert hf_token or offline, (
            "Qwen3 tokenizer requires --hf_token (online) or --offline (with a pre-populated local HF cache). "
            "For --offline, pre-download the tokenizer with `huggingface-cli download` and ensure HF_HOME points "
            "to the cache directory. NullTokenizer to be used soon."
        )

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
        # Create a simple namespace with the args needed by get_exp_name_config
        args_for_config = SimpleNamespace(
            num_gpus=num_gpus,
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
            virtual_pipeline_model_parallel_size=vp_size,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=etp_size,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
        )
        exp_config = get_exp_name_config(
            args_for_config, model_family_name, model_recipe_name, gpu, compute_dtype, task, config_variant
        )
        exp_name = (
            wandb_experiment_name
            if wandb_experiment_name is not None
            else f"{task}_{model_recipe_name}_{compute_dtype}_{exp_config}"
        )
    import os
    rank = os.environ.get('RANK', '0')
    exp_name += f'_worker{rank}'

    if pretrained_checkpoint is not None:
        custom_mounts.append(f"{pretrained_checkpoint}:{pretrained_checkpoint}")

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
        if nsys_trace is None:
            logger.warning("Using `cuda-sw` trace mode for profiling")
            logger.warning("Profiling results might not be accurate due to software tracing limitations.")
            # TODO: Remove this once the associated functional issues are resolved.
            nsys_trace = ["cuda-sw", "nvtx"]
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

    if use_recipes and dgxc_cluster is not None:
        plugins.append(
            FaultTolerancePlugin(
                enable_ft_package=True,
                calc_ft_timeouts=True,
                num_in_job_restarts=10,
                num_job_retries_on_failure=10,
                initial_rank_heartbeat_timeout=1800,
                rank_heartbeat_timeout=300,
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

    gpus_per_node = args.gpus_per_node
    if gpus_per_node is None:
        if args.gpu in NUM_GPUS_PER_NODE_MAP:
            gpus_per_node = NUM_GPUS_PER_NODE_MAP[args.gpu]
        else:
            raise ValueError(
                f"Invalid GPU type: {args.gpu}. Please use one of the following: {NUM_GPUS_PER_NODE_MAP.keys()}"
            )

    assert not (args.enable_nsys and args.pytorch_profiler), (
        "Both NSys and PyTorch profiler cannot be enabled at the same time"
    )

    # probably better to use parser.parse_args() and make unknowns an error,
    # but for now we'll just issue a warning.
    if unknown_args:
        logger.warning(f"Ignoring unrecognized arguments: {' '.join(unknown_args)}")

    env = dict(args.env or [])

    custom_env_vars = args.custom_env_vars
    custom_env_vars.update(env)

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
        offline=args.offline,
        detach=args.detach,
        dryrun=args.dryrun,
        enable_vboost=args.enable_vboost,
        enable_nsys=args.enable_nsys,
        pytorch_profiler=args.pytorch_profiler,
        moe_a2a_overlap=args.moe_a2a_overlap,
        tp_size=args.tensor_model_parallel_size,
        pp_size=args.pipeline_model_parallel_size,
        cp_size=args.context_parallel_size,
        vp_size=args.virtual_pipeline_model_parallel_size,
        ep_size=args.expert_model_parallel_size,
        etp_size=args.expert_tensor_parallel_size,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
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
        gpus_per_node=gpus_per_node,
        time_limit=args.time_limit,
        container_image=args.container_image,
        custom_mounts=args.custom_mounts,
        custom_env_vars=custom_env_vars,
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
            "eval_time_start_step": args.eval_time_start_step,
            "eval_time_end_step": args.eval_time_end_step,
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
        kubeflow_namespace=args.kubeflow_namespace,
        kubeflow_workdir_pvc=args.kubeflow_workdir_pvc,
        kubeflow_workdir_pvc_path=args.kubeflow_workdir_pvc_path,
        kubeflow_image_pull_secrets=args.kubeflow_image_pull_secrets,
        config_variant=config_variant,
        gres=args.gres,
        packager=args.packager,
    )
