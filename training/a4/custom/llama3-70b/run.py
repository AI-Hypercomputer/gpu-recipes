# To suppress BF16 compile related issue in the CI runs with turing/V100
import torch._dynamo
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

torch._dynamo.config.suppress_errors = True

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    model = MegatronGPTModel(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()