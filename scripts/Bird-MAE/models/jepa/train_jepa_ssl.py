import os 
import hydra 
import torch
import lightning as L
import sys

from omegaconf import OmegaConf, DictConfig
import pyrootutils
from pathlib import Path 

from datamodule import BirdSetDataModule_JEPA
from util.pylogger import get_pylogger
from util.log_hparams import log_hyperparameters
from build_model import instantiate_callbacks, build_model

log = get_pylogger(__name__)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base": None,
    "config_path": str(root / "configs"),
    "config_name": "pretrain.yaml"
}

@hydra.main(**_HYDRA_PARAMS)
def train(cfg: DictConfig):

    #log.info("Using config: %s", OmegaConf.to_yaml(cfg))
    log.info(f"Dataset directory:  <{os.path.abspath(cfg.paths.dataset_dir)}>")
    log.info(f"Log directory:  <{os.path.abspath(cfg.paths.log_dir)}>")
    log.info(f"Root directory:  <{os.path.abspath(cfg.paths.root_dir)}>")
    log.info(f"Work directory:  <{os.path.abspath(cfg.paths.work_dir)}>")
    log.info(f"Output directory:  <{os.path.abspath(cfg.paths.output_dir)}>")
    #log.info(f"Model directory:  <{os.path.abspath(cfg.callbacks.model_checkpoint.dirpath)}>")

    log.info("Seed everything with cfg.")
    L.seed_everything(cfg.seed)

    log.info("Setup datamodule")


    datamodule = BirdSetDataModule_JEPA(
            dataset_configs=cfg.data.dataset,
            loader_configs=cfg.data.loaders,
            transform_configs=cfg.data.transform,
            sampling_rate=cfg.module.network.sampling_rate
        )

    if sys.gettrace():
         log.info("Debugging mode, no logger")
         logger = None
    else:
        log.info("Setup logger")
        logger = hydra.utils.instantiate(cfg.logger)

    log.info("Setup callbacks")
    callbacks = instantiate_callbacks(cfg["callbacks"])
                                      

    log.info("Setup trainer")
    trainer = L.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    log.info("Setup model")
    model = build_model(cfg.module)

    object_dict = {
        "cfg": cfg, 
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer
    }
    if logger: 
        log.info("Logging hyperparameters")
        log_hyperparameters(object_dict)

    log.info("Start training")
    ckpt_path = cfg.get("ckpt_path", None)
    print(f"Loading from checkpoint: {ckpt_path}")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    train()