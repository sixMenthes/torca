"""Entry point for EMA self-distillation domain adaptation.

    python train_selfdistill.py                                  # Bird-MAE arm
    python train_selfdistill.py module/network=mim_distillation_beats   # BEATs arm
    python train_selfdistill.py +trainer.fast_dev_run=true         # smoke test

Mirrors train_fsqnet.py. The one structural difference is that this task is
self-supervised: the datamodule serves unlabelled two-view waveform pairs, so there
is no label_map, no class weights, and no test stage — the readout is a separate
linear probe over the adapted features (probe.py).
"""

import os
import sys
from pathlib import Path

import hydra
import lightning as L
from omegaconf import DictConfig

from build_model import build_model, instantiate_callbacks
from selfdistill_datamodule import SelfDistillDataModule
from util.log_hparams import log_hyperparameters
from util.pylogger import get_pylogger

log = get_pylogger(__name__)

# find project root: walk up from this file until we hit a dir containing ".git"
root = Path(__file__).resolve()
while not (root / ".git").exists():
    if root.parent == root:
        raise FileNotFoundError("could not find project root (no .git found)")
    root = root.parent

sys.path.insert(0, str(root))
os.environ["PROJECT_ROOT"] = str(root)

_HYDRA_PARAMS = {
    "version_base": None,
    "config_path": str(root / "configs"),
    "config_name": "selfdistill.yaml",
}


@hydra.main(**_HYDRA_PARAMS)
def train(cfg: DictConfig):
    log.info(f"Seed everything with {cfg.seed}")
    L.seed_everything(cfg.seed)

    log.info("Setup datamodule")
    datamodule = SelfDistillDataModule(
        dataset_configs=cfg.data.dataset,
        loader_configs=cfg.data.loaders,
        transform_configs=cfg.data.transform,
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
    # label_map is required by build_model's signature but unused for this arm.
    model = build_model(cfg.module, label_map={})

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("Logging hyperparameters")
        log_hyperparameters(object_dict)

    if cfg.train:
        log.info("Start training")
        trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
