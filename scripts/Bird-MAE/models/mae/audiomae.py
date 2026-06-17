import hydra 
import torch
import lightning as L
import torch.nn as nn

from .encoder import MAE_Encoder
from .decoder import MAE_Decoder

from timm.optim.optim_factory import param_groups_weight_decay
from util.pos_embed import get_2d_sincos_pos_embed_flexible
from transformers import get_cosine_schedule_with_warmup

class AudioMAE(L.LightningModule):
    def __init__(self, 
                 norm_layer,
                 norm_pix_loss,
                 mask_ratio,
                 cfg_encoder,
                 cfg_decoder,
                 optimizer,
                 scheduler,
                 loss
    ):
        super().__init__()
        self.save_hyperparameters()

        self.norm_pix_loss = norm_pix_loss
        self.mask_ratio = mask_ratio        
        self.optimizer_cfg = optimizer.target
        self.scheduler_cfg = scheduler 
        self.loss = hydra.utils.instantiate(loss)
        self.train_batch_size = optimizer.extras.train_batch_size
        self.layer_decay = optimizer.extras.layer_decay
        self.target_length = cfg_encoder.img_size_x

        self.encoder = MAE_Encoder(
            img_size_x=cfg_encoder.img_size_x,
            img_size_y=cfg_encoder.img_size_y,
            patch_size=cfg_encoder.patch_size,
            in_chans=cfg_encoder.in_chans,
            embed_dim=cfg_encoder.embed_dim,
            depth=cfg_encoder.depth,
            num_heads=cfg_encoder.num_heads,
            mlp_ratio=cfg_encoder.mlp_ratio,
            norm_layer=norm_layer,
            pos_trainable=cfg_encoder.pos_trainable,
        )

        self.decoder = MAE_Decoder(
            embed_dim=cfg_encoder.embed_dim,
            decoder_embed_dim=cfg_decoder.embed_dim,
            num_patches=self.encoder.patch_embed.num_patches,
            decoder_depth=cfg_decoder.depth,
            decoder_num_heads=cfg_decoder.num_heads,
            mlp_ratio=cfg_decoder.mlp_ratio,
            norm_layer=norm_layer,
            patch_size=cfg_decoder.patch_size,
            in_chans=cfg_encoder.in_chans,
            decoder_mode=cfg_decoder.mode,
            pos_trainable=cfg_decoder.pos_trainable,
            no_shift=cfg_decoder.no_shift,
            target_length=self.target_length
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize encoder positional embeddings
        pos_embed = get_2d_sincos_pos_embed_flexible(
            self.encoder.pos_embed.shape[-1], # embedding dim
            self.encoder.patch_embed.patch_hw, # 8,32
            cls_token=True
        )
        self.encoder.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize decoder positional embeddings
        decoder_pos_embed = get_2d_sincos_pos_embed_flexible(
            self.decoder.decoder_pos_embed.shape[-1], # embedding_dim
            self.encoder.patch_embed.patch_hw,  # 8,32
            cls_token=True
        )
        self.decoder.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.encoder.cls_token, std=.02)
        torch.nn.init.normal_(self.decoder.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def patchify(self, imgs):
        p = self.encoder.patch_embed.patch_size[0]
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p

        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        
        return x

    def unpatchify(self, x): # changed? 
        p = self.encoder.patch_embed.patch_size[0]    
        h = self.target_length//p
        w = 128//p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        specs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return specs

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        #loss = self.loss(pred, target)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        #loss = self.loss(pred, target)
        #loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss   

    def forward(self, imgs):
        latent, mask, ids_restore = self.encoder(imgs, self.mask_ratio)
        pred = self.decoder(latent, ids_restore)
        loss_recon = self.forward_loss(imgs, pred, mask)
        return loss_recon, pred, mask

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        #labels = batch["label"]
        loss, pred, mask = self(audio)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        eff_batch_size = self.trainer.accumulate_grad_batches * self.trainer.num_devices * self.train_batch_size * self.trainer.num_nodes
        print("base learning rate on 256 was:", self.optimizer_cfg["lr"])
        self.optimizer_cfg["lr"] = self.optimizer_cfg["lr"] * eff_batch_size / 256
        print("effective learning rate now:", self.optimizer_cfg["lr"], self.layer_decay)
        
        param_groups = param_groups_weight_decay(
            self,
            self.optimizer_cfg["weight_decay"],
            no_weight_decay_list=("bias", "bn", "ln", "gn", "norm")
        )

        optimizer = torch.optim.AdamW(
            param_groups, 
            lr=self.optimizer_cfg["lr"], 
            betas=self.optimizer_cfg["betas"])
    
        if self.scheduler_cfg: 
            num_training_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = 0.09375 # hard coded
            num_warmup_steps = num_training_steps * warmup_ratio

            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  # Update at every step
                "frequency": 1
            }

            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        
        return {"optimizer": optimizer}
