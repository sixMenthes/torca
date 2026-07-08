import hydra
from omegaconf import DictConfig
from models.vit.vit import VIT_ppnet, VIT
from util import pylogger

log = pylogger.get_pylogger(__name__)


def instantiate_callbacks(cfg_callbacks: DictConfig):
    callbacks = []

    if not cfg_callbacks:
        log.warning("No callbacks found")

    for _, cb_config in cfg_callbacks.items():
        if isinstance(cb_config, DictConfig) and "_target_" in cb_config:
            log.info(f"Instantiating callback <{cb_config._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_config))
    return callbacks


def build_model(cfg_module: DictConfig, label_map: dict):

    if cfg_module.network.name == "VIT_ppnet":
        module = VIT_ppnet(
            img_size_x=cfg_module.network.img_size_x,
            img_size_y=cfg_module.network.img_size_y,
            patch_size=cfg_module.network.patch_size,
            in_chans=cfg_module.network.in_chans,
            embed_dim=cfg_module.network.embed_dim,
            global_pool=cfg_module.network.global_pool,
            drop_path=cfg_module.network.drop_path,
            norm_layer=cfg_module.network.norm_layer,
            mlp_ratio=cfg_module.network.mlp_ratio,
            qkv_bias=cfg_module.network.qkv_bias,
            eps=cfg_module.network.eps,
            num_heads=cfg_module.network.num_heads,
            depth=cfg_module.network.depth,
            pretrained_weights_path=cfg_module.network.pretrained_weights_path,
            target_length=cfg_module.network.target_length,
            num_classes=cfg_module.network.num_classes,
            optimizer=cfg_module.optimizer,
            scheduler=cfg_module.scheduler,
            loss=cfg_module.loss,
            metric_cfg=cfg_module.metric,
            mask2d=cfg_module.network.mask2d,
            mask_t_prob=cfg_module.network.mask_t_prob,
            mask_f_prob=cfg_module.network.mask_f_prob,
            ema_update_rate=cfg_module.network.ema_update_rate,
            ppnet_cfg=cfg_module.network.ppnet,
            mask_inference=cfg_module.network.mask_inference,
            label_map=label_map,
            pcen_cfg=cfg_module.network.get("pcen", None),
        )

    elif cfg_module.network.name == "VIT":
        module = VIT(
            img_size_x=cfg_module.network.img_size_x,
            img_size_y=cfg_module.network.img_size_y,
            patch_size=cfg_module.network.patch_size,
            in_chans=cfg_module.network.in_chans,
            embed_dim=cfg_module.network.embed_dim,
            global_pool=cfg_module.network.global_pool,
            drop_path=cfg_module.network.drop_path,
            norm_layer=cfg_module.network.norm_layer,
            mlp_ratio=cfg_module.network.mlp_ratio,
            qkv_bias=cfg_module.network.qkv_bias,
            eps=cfg_module.network.eps,
            num_heads=cfg_module.network.num_heads,
            depth=cfg_module.network.depth,
            pretrained_weights_path=cfg_module.network.pretrained_weights_path,
            target_length=cfg_module.network.target_length,
            num_classes=cfg_module.network.num_classes,
            optimizer=cfg_module.optimizer,
            scheduler=cfg_module.scheduler,
            loss=cfg_module.loss,
            metric_cfg=cfg_module.metric,
            mask2d=cfg_module.network.mask2d,
            mask_t_prob=cfg_module.network.mask_t_prob,
            mask_f_prob=cfg_module.network.mask_f_prob,
            ema_update_rate=cfg_module.network.ema_update_rate,
            mask_inference=cfg_module.network.mask_inference,
            label_map=label_map,
        )

    else:
        raise ValueError(f"Model {cfg_module.network.name} not found")

    return module
