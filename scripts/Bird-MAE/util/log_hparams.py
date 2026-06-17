from pytorch_lightning.utilities import rank_zero_only
from . import pylogger

log = pylogger.get_pylogger(__name__)

@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Saves additionally:
    - Number of model parameters

    Args:
        object_dict (dict): Dict object with all parameters.
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["module"] = cfg["module"]

    # save number of model parameters
    hparams["module/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["module/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["module/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["paths"] = cfg.get("paths")

    hparams["task_name"] = cfg.get("task_name")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)