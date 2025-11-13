"""Nemo2 pretraining recipe for Llama 3.1 70B model."""

from nemo.collections import llm
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.llm.recipes import llama31_70b
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import AutoResume
from nemo.lightning import ModelCheckpoint
from nemo.lightning.pytorch.callbacks import NsysCallback
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.utils.loggers.dllogger import DLLogger
import nemo_run as run
import os
import re


def recipe(
    profile_enabled: bool = False,
    profile_start_step: int = 0,
    profile_end_step: int = 0,
    profile_ranks: str = "0",
    job_name: str = "llama31-70b",
    enable_dataloading: bool = False,
    enable_ckpt_write: bool = False,
    enable_ckpt_load: bool = False,
    restore_path: str = None,
    dataset_path: str = None,
    token_path: str = None,
    ckpt_write_interval: int = 25,
):
    """Returns a Nemo2 training recipe for Llama 3.1 70B model.

  Args:
      profile_enabled: Whether to enable Nsys profiling.
      profile_start_step: The step to start profiling.
      profile_end_step: The step to end profiling.
      profile_ranks: The ranks to profile, comma separated.
      job_name: The name of the job.
      enable_dataloading: Whether to enable dataloading.
      enable_ckpt_write: Whether to enable checkpoint writing.
      enable_ckpt_load: Whether to enable checkpoint loading.
      ckpt_write_interval: The interval at which to write checkpoints.
      dataset_path: Path to the dataset.
      token_path: Path to the tokenizer.
      restore_path: Path to a specific checkpoint to restore from.

  Returns:
      A Nemo2 training recipe.
  """

    # Start from the Nemo standard recipe.
    pretrain = llama31_70b.pretrain_recipe(performance_mode=False)

    # Set the number of steps to 50 for a quicker benchmark.
    pretrain.trainer.max_steps = 50

    # Disable validation batches.
    pretrain.trainer.limit_val_batches = 0.0
    pretrain.trainer.val_check_interval = 100

    # Add the Nsys profiling callback if enabled.
    if profile_enabled:
        pretrain.trainer.callbacks.append(
            run.Config(
                NsysCallback,
                start_step=profile_start_step,
                end_step=profile_end_step,
                ranks=[int(x) for x in profile_ranks.split(",")],
                gen_shape=False,
            ))

    # Add the FLOPs measurement callback.
    pretrain.trainer.callbacks.append(
        run.Config(
            FLOPsMeasurementCallback,
            model_name="llama31-70b",
            model_config=pretrain.model.config,
            data_config=pretrain.data,
        ))

    if enable_dataloading and dataset_path and token_path:
        # Set the tokenizer.
        tokenizer = run.Config(
            get_nmt_tokenizer,
            library="sentencepiece",
            tokenizer_model=token_path,
        )
        paths = list(get_unique_base_names(dataset_path))
        # Set the training data.
        pretrain.data = run.Config(
            PreTrainingDataModule,
            paths=paths,
            global_batch_size=512,
            micro_batch_size=1,
            num_workers=16,
            pin_memory=False,
            seq_length=8192,
            num_dataset_builder_threads=16,
            index_mapping_dir=None,
            tokenizer=tokenizer,
        )

    pretrain.log.ckpt = None
    pretrain.trainer.enable_checkpointing = False

    # Log every step.
    pretrain.trainer.log_every_n_steps = 1
    # From Nemo 1.0 GA runs https://paste.googleplex.com/5993741590200320#l=77
    if enable_ckpt_write:
        pretrain.trainer.strategy.save_ckpt_format = "torch_dist"
        pretrain.trainer.strategy.ckpt_load_on_device = True
        pretrain.trainer.strategy.ckpt_parallel_save = True
        pretrain.trainer.strategy.ckpt_async_save = True

        pretrain.log.ckpt = run.Config(
            ModelCheckpoint,
            save_top_k=10,
            save_last=False,
            every_n_train_steps=ckpt_write_interval,
            dirpath="/checkpoints/" + job_name,
            monitor="step",
            mode="max",
            filename="megatron_llama--{val_loss:.2f}-{step}-{consumed_samples}",
        )
    else:
        # NeMo recipe has checkpoint enabled by default.
        # https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/llm/recipes/llama31_70b.py#L117
        pretrain.trainer.strategy.ckpt_async_save = False
        pretrain.log.ckpt = run.Config(
            ModelCheckpoint,
            save_last=False,
        )

    if enable_ckpt_load:
        pretrain.trainer.enable_checkpointing = True
        if restore_path == "":
            restore_path = None
        pretrain.resume = run.Config(
            AutoResume,
            resume_if_exists=True,  # enable checkpoint restore.
            resume_ignore_no_checkpoint=True,
            resume_from_path=restore_path,
        )
    elif not enable_ckpt_load and enable_ckpt_write:
        pretrain.resume = run.Config(
            AutoResume,
            resume_if_exists=False,  # disable checkpoint restore.
            resume_ignore_no_checkpoint=False,
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


def get_unique_base_names(directory_path: str) -> set:
    try:
        # A set comprehension processes each name from the directory listing in one line.
        return {
            os.path.join(directory_path,
                         re.sub(r'(\.idx|\.bin)$', '', name.rstrip('/')))
            for name in os.listdir(directory_path)
        }
    except FileNotFoundError:
        print(f"Error: Directory not found at '{directory_path}'")
        return set()


if __name__ == "__main__":
    run.cli.main(llm.pretrain, default_factory=recipe)
