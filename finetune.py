import os 
import hydra 
import torch
import lightning as L
import sys
import torch.nn as nn 
from omegaconf import OmegaConf, DictConfig
from pathlib import Path 
from torca_datamodule import LabelDataModule

from util.pylogger import get_pylogger
from util.log_hparams import log_hyperparameters
from build_model import instantiate_callbacks, build_model

log = get_pylogger(__name__)

from pathlib import Path
import sys

# find project root: walk up from this file until we hit a dir containing ".git"
root = Path(__file__).resolve()
while not (root / ".git").exists():
    if root.parent == root:          # reached filesystem root
        raise FileNotFoundError("could not find project root (no .git found)")
    root = root.parent

  # pythonpath=True
sys.path.insert(0, str(root))

# pyrootutils also sets this; keep it if any of your code reads os.environ["PROJECT_ROOT"]
import os
os.environ["PROJECT_ROOT"] = str(root)

_HYDRA_PARAMS = {
    "version_base": None,
    "config_path": str(root / "configs"),
    "config_name": "torca.yaml"
}

@hydra.main(**_HYDRA_PARAMS)
def finetune(cfg: DictConfig):
    log.info(f"Seed everything with {cfg.seed}")
    L.seed_everything(cfg.seed)
    #torch.set_num_threads(12)
    #print(OmegaConf.to_yaml(cfg))
    

    if "dclde" in cfg.data.dataset.name.lower():
        datamodule = LabelDataModule(
            dataset_configs=cfg.data.dataset,
            loader_configs=cfg.data.loaders,
            transform_configs=cfg.data.transform
        )
        label_map = datamodule.label_map

    if sys.gettrace():
         log.info("Debugging mode, no logger")
         logger = None
    else:
        log.info("Setup logger")
        logger = hydra.utils.instantiate(cfg.logger)

    log.info("Setup callbacks")
    callbacks = instantiate_callbacks(cfg["callbacks"])
                                      
    log.info("Setup trainer")
    trainer = L.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger, profiler="simple")

    log.info("Setup model")
    model = build_model(cfg.module, label_map)

    pretrained_weights_path = cfg.module.network.get("pretrained_weights_path", None)

    if pretrained_weights_path: 
        log.info(f"Load pretrained weights from {pretrained_weights_path}")
        model.load_pretrained_weights(pretrained_weights_path, cfg.data.dataset.name)

    if cfg.module.network.get("freeze_backbone", False): # move this to the models!
        log.info("Freezing backbone weights, only training head")
        if cfg.module.network.name == "ConvNext":
             for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
        elif cfg.module.network.name == "VIT_ppnet" or cfg.module.network.name.endswith("ppnet"):
            for name, param in model.named_parameters():
                if 'ppnet' not in name:
                    param.requires_grad = False
        else:
            if cfg.module.network.get("global_pool", "") == "attentive":
                for name, param in model.named_parameters():
                    if 'head' not in name and 'attentive_probe' not in name:
                        param.requires_grad = False
            else:
                for name, param in model.named_parameters():
                    if 'head' not in name:
                        param.requires_grad = False

        
        if cfg.module.network.get("head", None) == "MLP":
            in_features = model.head.in_features
            out_features = model.head.out_features

            head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, out_features)
            )

            model.head = head

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

    if cfg.train: 
        log.info("Start training")
        trainer.fit(model=model, datamodule=datamodule) 
                    #,ckpt_path="/home/lrauch/projects/birdMAE/logs/finetune/runs/audioset_balanced/VIT/2024-10-14_170415/model_checkpoints/last.ckpt")

    if cfg.test:
        log.info("Start testing")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="last")

        
if __name__ == "__main__":
    finetune()


    

