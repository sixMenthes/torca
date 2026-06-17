
from functools import partial

import hydra
import lightning as L
import torch
import torch.nn as nn
from torch.nn import functional as F
import datasets
import copy
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import VisionTransformer,PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed_flexible
from util.lr_decay import param_groups_lrd
from util.patch_embed import PatchEmbed_new

from ..components.attentive_pooling import AttentivePooling
from ..components.cosine_warmup import CosineWarmupScheduler
from ..components.ema import EMA
from ..ppnet.ppnet import PPNet
from torchmetrics import MetricCollection

class VIT(L.LightningModule,VisionTransformer):

    def __init__(self, 
                 img_size_x,
                 img_size_y,
                 patch_size,
                 in_chans,
                 embed_dim,
                 global_pool,
                 norm_layer,
                 mlp_ratio,
                 qkv_bias,
                 eps,
                 drop_path,
                 num_heads,
                 depth,
                 num_classes,
                 optimizer,
                 scheduler,
                 pretrained_weights_path, 
                 target_length,
                 loss,
                 metric_cfg,
                 mask_t_prob,
                 mask_f_prob,
                 mask2d,
                 ema_update_rate,
                 mask_inference
    ):
        
        L.LightningModule.__init__(self)
        
        if mask_inference:
            num_classes = 9735 # XCL overwrite 

        VisionTransformer.__init__(
            self,
            img_size = (img_size_x, img_size_y), 
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            qkv_bias = qkv_bias,
            norm_layer = partial(nn.LayerNorm, eps=eps),
            num_classes = num_classes,
            drop_path_rate=drop_path,
        )
        self.save_hyperparameters()
        self.img_size = (img_size_x, img_size_y)
        self.global_pool = global_pool

        norm_layer = partial(nn.LayerNorm, eps=eps)
        self.fc_norm = norm_layer(embed_dim)
        self.mask_2d = mask2d

        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes 
        self.qkv_bias = qkv_bias 
        self.ema_update_rate = ema_update_rate

        self.loss = hydra.utils.instantiate(loss)
        self.optimizer = None
        self.optimizer_cfg = optimizer.target
        self.train_batch_size = optimizer.extras.train_batch_size
        self.layer_decay = optimizer.extras.layer_decay
        self.decay_type = optimizer.extras.decay_type
        self.scheduler_cfg = scheduler

        if self.global_pool == "attentive":
            #attentive_heads = self.embed_dim // self.num_heads
            #self.attentive_probe = AttentivePooling(self.embed_dim, self.num_heads)
            self.attentive_probe = AttentivePooling(dim=self.embed_dim, num_heads=self.num_heads)

        self.mask_2d = mask2d
        self.mask_t_prob = mask_t_prob
        self.mask_f_prob = mask_f_prob

        self.pretrained_weights_path = pretrained_weights_path
        self.target_length = target_length

        metric = hydra.utils.instantiate(metric_cfg)
        
        additional_metrics = []
        if metric_cfg.get("additional"):
            for _, metric_cfg in metric_cfg.additional.items():
                additional_metrics.append(hydra.utils.instantiate(metric_cfg))
        add_metrics = MetricCollection(additional_metrics)
        self.test_add_metrics = add_metrics.clone()
        self.val_add_metrics = add_metrics.clone()

        self.train_metric = metric.clone()
        self.val_metric = metric.clone()
        self.test_metric = metric.clone()

        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []
        
        self.ema = None
        if self.ema_update_rate: 
            self.ema = EMA(self, decay=ema_update_rate)

        self.class_mask = None
        if mask_inference:
            print("Logit Masking")
            hf_path = "DBD-research-group/BirdSet"
            hf_name = "XCL"
            pretrain_labels = datasets.load_dataset_builder(
                hf_path, hf_name, trust_remote_code=True).info.features["ebird_code"]
            inference_labels = datasets.load_dataset_builder(
                hf_path, mask_inference, trust_remote_code=True).info.features["ebird_code"]
            self.class_mask = [pretrain_labels.names.index(i) for i in inference_labels.names]
        
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # batch, patch, embed
        x = x + self.pos_embed[:, 1:, :] 
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)        

        for blk in self.blocks:
            x = blk(x)
            #x = torch.nan_to_num(x, nan=0.0) 

        if self.global_pool != "average": 
            x = x[:, 1:, :].mean(dim=1)  
            outcome = self.fc_norm(x)
        elif self.global_pool =="attentive":
            outcome = self.attentive_probe(x)
            outcome = self.fc_norm(outcome)
        elif self.global_pool == "cls":
            x = self.norm(x)
            outcome = x[:, 0]
        else:
            raise ValueError(f"Invalid global pool type: {self.global_pool}")
        return outcome
    
    def forward_features_mask(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # batch, patch, embed
        x = x + self.pos_embed[:, 1:, :] # strange

        if self.mask_2d: 
            x, mask, ids_restore = self.random_masking_2d(x)
        else:
            pass

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)        

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome 

    def forward(self, x):
        if self.mask_t_prob > 0.0 or self.mask_f_prob > 0.0: #shape val: 64, 1, 512, 128
            x = self.forward_features_mask(x)
        else:
            x = self.forward_features(x)
        pred = self.head(x)
        return pred 

    def random_masking_2d(self, x):
        N, L, D = x.shape
        T = 64 # AUDIOSET
        F = 8 # AUDIOSET

        # mask T
        x = x.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - self.mask_t_prob))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        #x_masked = torch.gather(x, dim=1, index=index)
        #x_masked = x_masked.reshape(N,len_keep_T*F,D)
        x = torch.gather(x, dim=1, index=index) # N, len_keep_T(T'), F, D

        # mask F
        #x = x.reshape(N, T, F, D)
        x = x.permute(0,2,1,3) # N T' F D => N F T' D
        len_keep_F = int(F * (1 - self.mask_f_prob))
        noise = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_F]
        #index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0,2,1,3) # N F' T' D => N T' F' D 
        #x_masked = x_masked.reshape(N,len_keep*T,D)
        x_masked = x_masked.reshape(N,len_keep_F*len_keep_T,D)
            
        return x_masked, None, None

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]
        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.ema: 
            self.ema.update()

        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        if self.ema: 
            self.ema.apply_shadow()

        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())

        self.val_predictions.append(pred.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.ema:
            self.ema.restore()
    
    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_predictions)
        targets = torch.cat(self.val_targets)
        metric = self.val_metric(preds, targets)
        self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)
        print("val metric:", metric.detach().cpu().item())

        self.val_add_metrics(preds, targets)
        for name, metric in self.val_add_metrics.items():
            self.log(f'valid_{name}', metric, on_epoch=True, prog_bar=True)

        self.val_predictions = []
        self.val_targets = []
    
    def test_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        if self.ema: 
            self.ema.apply_shadow()

        self.mask_t_prob = 0.0
        self.mask_f_prob = 0.0 

        pred = self(audio)

        if self.class_mask: 
            pred = pred[:, self.class_mask]

        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())
        
        self.test_predictions.append(pred.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.ema: 
            self.ema.restore()
    
    def on_test_epoch_end(self):
        preds = torch.cat(self.test_predictions)
        targets = torch.cat(self.test_targets)
        self.test_metric(preds, targets)
        self.log(f'test_{self.test_metric.__class__.__name__}', self.test_metric, on_epoch=True, prog_bar=True)

        self.test_add_metrics(preds, targets)
        for name, metric in self.test_add_metrics.items():
            self.log(f'test_{name}', metric, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):

        if self.layer_decay:
            params = param_groups_lrd(
                model=self,
                weight_decay=self.optimizer_cfg["weight_decay"],
                no_weight_decay_list=self.no_weight_decay(),
                layer_decay=self.layer_decay, #scaling favtor for ech layer 0.75^layer ..--> 0.75^0
                decay_type=self.decay_type
            )

            self.optimizer = hydra.utils.instantiate(
                self.optimizer_cfg, 
                params
            )

        else:
            self.optimizer = hydra.utils.instantiate(
                self.optimizer_cfg, 
                params=self.parameters())

    
        if self.scheduler_cfg: 
            num_training_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = 0.067 # hard coded
            num_warmup_steps = num_training_steps * warmup_ratio


            scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=num_warmup_steps,
                total_steps=num_training_steps
            )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  
                "frequency": 1,
                "name": "lr_cosine"
            }

            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}
        
        return {"optimizer": self.optimizer}      
    
    def load_pretrained_weights(self, pretrained_weights_path, dataset_name): 
        img_size = (self.target_length, 128)
        #img_size = (128, self.target_length) # should be correcter, but not pretrained this way

        if self.target_length == 512: #esc50, hsn, 5 seconds
            #num_patches = 512 # audioset
            if "xc" in self.pretrained_weights_path or "XCL" in self.pretrained_weights_path:
                num_patches = 256 # birdset
            else:
                num_patches = 512 # audioset

            self.patch_embed = PatchEmbed(img_size, 16, 1, self.embed_dim)
            #self.patch_embed = PatchEmbed_org(img_size, 16, 1, self.embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False) #to load pretrained pos embed
            try:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["model"]
            except:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["state_dict"]

            pretrained_state_dict = {}

            for key, value in pre_state_dict.items():
                if key.startswith("decoder."):
                    # Skip any key that starts with "decoder."
                    continue
                elif key.startswith("encoder."):
                    # Remove the "encoder." prefix
                    new_key = key[len("encoder."):]
                else:
                    # Use the original key if no prefix
                    new_key = key
                
                # Add the modified key-value pair to the new state dict
                pretrained_state_dict[new_key] = value

            if not self.class_mask:
                for k in ['head.weight', 'head.bias']:
                    if k in pretrained_state_dict: #and pretrained_state_dict[k].shape != self.state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del pretrained_state_dict[k]
            
            info = self.load_state_dict(pretrained_state_dict, strict=False)

            patch_hw = (img_size[1] // 16, img_size[0] // 16) # 16=patchsize
            #patch_hw = (img_size[0] // 16, img_size[1] // 16) 
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.size(-1), patch_hw, cls_token=True) # not trained, overwrite from sincos
            self.pos_embed.data = torch.from_numpy(pos_embed).float().unsqueeze(0) 

        elif self.target_length == 1024: #audioset, 10 seconds

            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=self.embed_dim, stride=16) # no overlap. stride=img_size=16
           
            if "xc" in self.pretrained_weights_path:
                num_patches = 256 # birdset # does not work right now 
            else:
                num_patches =  num_patches = self.patch_embed.num_patches # audioset
            #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding

            checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
            try:
                pre_state_dict = checkpoint["model"]
            except:
                pre_state_dict = checkpoint["state_dict"]

            pretrained_state_dict = {}

            for key, value in pre_state_dict.items():
                if key.startswith("decoder."):
                    # Skip any key that starts with "decoder."
                    continue
                elif key.startswith("encoder."):
                    # Remove the "encoder." prefix
                    new_key = key[len("encoder."):]
                else:
                    # Use the original key if no prefix
                    new_key = key
                
                # Add the modified key-value pair to the new state dict
                pretrained_state_dict[new_key] = value

            state_dict = self.state_dict()

            for k in ["head.weight", "head.bias"]:
                if k in pretrained_state_dict and pretrained_state_dict[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrained_state_dict[k]

            self.load_state_dict(pretrained_state_dict, strict=False)

            trunc_normal_(self.head.weight, std=2e-5)


class VIT_ppnet(L.LightningModule,VisionTransformer):

    def __init__(self, 
                 img_size_x,
                 img_size_y,
                 patch_size,
                 in_chans,
                 embed_dim,
                 global_pool,
                 norm_layer,
                 mlp_ratio,
                 qkv_bias,
                 eps,
                 drop_path,
                 num_heads,
                 depth,
                 num_classes,
                 optimizer,
                 scheduler,
                 pretrained_weights_path, 
                 target_length,
                 loss,
                 metric_cfg,
                 mask_t_prob,
                 mask_f_prob,
                 mask2d,
                 ema_update_rate,
                 ppnet_cfg,
                 mask_inference
    ):
        
        L.LightningModule.__init__(self)

        if mask_inference:
            num_classes = 411 # XCL overwrite 
            ppnet_cfg.num_classes = num_classes
        
        VisionTransformer.__init__(
            self,
            img_size = (img_size_x, img_size_y), ###test!!
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            qkv_bias = qkv_bias,
            norm_layer = partial(nn.LayerNorm, eps=eps),
            num_classes = num_classes,
            drop_path_rate=drop_path,
        )
        self.save_hyperparameters()

        self.ppnet_cfg = ppnet_cfg
        self.ppnet = PPNet(
            num_prototypes=ppnet_cfg.num_prototypes,
            channels_prototypes=ppnet_cfg.channels_prototypes,
            h_prototypes=ppnet_cfg.h_prototypes,
            w_prototypes=ppnet_cfg.w_prototypes,
            num_classes=ppnet_cfg.num_classes,
            topk_k=ppnet_cfg.topk_k,
            margin=ppnet_cfg.margin,
            init_weights=ppnet_cfg.init_weights,
            add_on_layers_type=ppnet_cfg.add_on_layers_type,
            incorrect_class_connection=ppnet_cfg.incorrect_class_connection,
            correct_class_connection=ppnet_cfg.correct_class_connection,
            bias_last_layer=ppnet_cfg.bias_last_layer,
            non_negative_last_layer=ppnet_cfg.non_negative_last_layer,
            embedded_spectrogram_height=ppnet_cfg.embedded_spectrogram_height,
        )

    #   for p in model.backbone_model.parameters():
    #     p.requires_grad = False
    # for p in model.add_on_layers.parameters():
    #     p.requires_grad = True
    # model.prototype_vectors.requires_grad = True      
        self.img_size = (img_size_x, img_size_y)
        self.global_pool = global_pool

        norm_layer = partial(nn.LayerNorm, eps=eps)
        self.fc_norm = norm_layer(embed_dim)
        self.mask_2d = mask2d

        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes 
        self.qkv_bias = qkv_bias 
        self.ema_update_rate = ema_update_rate

        self.loss = hydra.utils.instantiate(loss)
        self.optimizer = None
        self.optimizer_cfg = optimizer.target
        self.train_batch_size = optimizer.extras.train_batch_size
        self.layer_decay = optimizer.extras.layer_decay
        self.decay_type = optimizer.extras.decay_type
        self.scheduler_cfg = scheduler

        self.mask_2d = mask2d
        self.mask_t_prob = mask_t_prob
        self.mask_f_prob = mask_f_prob

        self.pretrained_weights_path = pretrained_weights_path
        self.target_length = target_length

        metric = hydra.utils.instantiate(metric_cfg)
        
        additional_metrics = []
        if metric_cfg.get("additional"):
            for _, metric_cfg in metric_cfg.additional.items():
                additional_metrics.append(hydra.utils.instantiate(metric_cfg))
        add_metrics = MetricCollection(additional_metrics)
        self.test_add_metrics = add_metrics.clone()
        self.val_add_metrics = add_metrics.clone()

        self.train_metric = metric.clone()
        self.val_metric = metric.clone()
        self.test_metric = metric.clone()

        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []
        
        self.ema = None
        if self.ema_update_rate: 
            self.ema = EMA(self, decay=ema_update_rate)

        
        self.class_mask = None
        if mask_inference:
            print("Logit Masking")
            hf_path = "DBD-research-group/BirdSet"
            hf_name = "XCM"
            pretrain_labels = datasets.load_dataset_builder(
                hf_path, hf_name, trust_remote_code=True).info.features["ebird_code"]
            inference_labels = datasets.load_dataset_builder(
                hf_path, mask_inference, trust_remote_code=True).info.features["ebird_code"]
            self.class_mask = [pretrain_labels.names.index(i) for i in inference_labels.names]
        
        del self.head
        #del self.norm
        del self.fc_norm
        del self.head_drop
        

    def forward_features(self, x):
        B = x.shape[0]
        #x = x.permute(0,1,3,2) # test!!
        x = self.patch_embed(x) # batch, patch, embed
        x = x + self.pos_embed[:, 1:, :] # strange
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)        

        for blk in self.blocks:
            x = blk(x)
            #x = torch.nan_to_num(x, nan=0.0) #????
        x = self.norm(x)

        if self.ppnet_cfg.focal_similarity == True:
            x_cls = x[:, 0, :]
            x_patch = x[:, 1:, :] 
            z_f = x_patch - x_cls.unsqueeze(1) 
            try:
                x = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, 8, 32)
            except:
                x = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, 8, 64) # audioset
        else:
            x = x[:,1:,:].permute(0,2,1).reshape(B, self.embed_dim, 8, 32)

        logits,_ = self.ppnet(x)

        return logits
    

    def forward(self, x):
        logits = self.forward_features(x)
        return logits 


    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]
        logits = self(audio)
        targets = targets.long()
        #preds = logits.sigmoid()
        bce_loss = self.loss(logits, targets.float())
        orthogonality_loss = self.calculate_orthogonality_loss()

        self.log('bce_loss', bce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('orthogonality_loss', orthogonality_loss, on_step=True, on_epoch=True, prog_bar=True)

        loss = bce_loss + orthogonality_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.ema: 
            self.ema.update()

        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        if self.ema: 
            self.ema.apply_shadow()

        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())

        #metric = self.val_metric(pred, targets)
        #pred = torch.softmax(pred, dim=1)
        self.val_predictions.append(pred.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

        #self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.ema:
            self.ema.restore()
    
    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_predictions)
        targets = torch.cat(self.val_targets)
        metric = self.val_metric(preds, targets)
        self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)
        print("val metric:", metric.detach().cpu().item())

        self.val_add_metrics(preds, targets)
        for name, metric in self.val_add_metrics.items():
            self.log(f'valid_{name}', metric, on_epoch=True, prog_bar=True)

        self.val_predictions = []
        self.val_targets = []
    
    def test_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        if self.ema: 
            self.ema.apply_shadow()

        self.mask_t_prob = 0.0
        self.mask_f_prob = 0.0 #fix later!

        pred = self(audio)
        if self.class_mask: 
        # if targets.shape == pred.shape:
        #     targets = targets[:, self.class_mask]
            pred = pred[:, self.class_mask]

        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())
        
        self.test_predictions.append(pred.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.ema: 
            self.ema.restore()
    
    def on_test_epoch_end(self):
        preds = torch.cat(self.test_predictions)
        targets = torch.cat(self.test_targets)
        self.test_metric(preds, targets)
        self.log(f'test_{self.test_metric.__class__.__name__}', self.test_metric, on_epoch=True, prog_bar=True)

        self.test_add_metrics(preds, targets)
        for name, metric in self.test_add_metrics.items():
            self.log(f'test_{name}', metric, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        
        from util.lr_decay import param_groups_lrd_pp
        #heuristic:
        # eff_batch_size = self.trainer.accumulate_grad_batches * self.trainer.num_devices * self.train_batch_size
        # self.optimizer_cfg["lr"] = self.optimizer_cfg["lr"] * eff_batch_size / 48
        # print("effective learning rate:", self.optimizer_cfg["lr"], self.layer_decay)

        if self.layer_decay:
            params = param_groups_lrd_pp(
                model=self,
                weight_decay=self.optimizer_cfg["weight_decay"],
                no_weight_decay_list=self.no_weight_decay(),
                layer_decay=self.layer_decay, #scaling favtor for ech layer 0.75^layer ..--> 0.75^0
                decay_type=self.decay_type,
                last_layer_lr=self.ppnet_cfg.last_layer_lr,
                prototype_lr=self.ppnet_cfg.prototype_lr,
            )

            self.optimizer = hydra.utils.instantiate(
                self.optimizer_cfg, 
                params
            )

        else:
            print("TEST:",self.ppnet_cfg.last_layer_lr)
            # self.optimizer = hydra.utils.instantiate(
            #     self.optimizer_cfg, 
            #     params=self.parameters())
            optimizer_specifications = []

            #1) Add the add_on_layers group
            addon_params = list(self.ppnet.add_on_layers.parameters())
            optimizer_specifications.append({
                "params": addon_params,
                "lr": 3e-2,
                "weight_decay": 1e-4,
            })

            # 2) Add the prototype_vectors group
            #    (assuming this is either a list of Tensors or just one Tensor)
            proto_params = [self.ppnet.prototype_vectors]  # or list(...)
            optimizer_specifications.append({
                "params": proto_params,
                "lr": self.ppnet_cfg.prototype_lr,
            })

            # 3) Add the last_layer group
            last_params = list(self.ppnet.last_layer.parameters())
            optimizer_specifications.append({
                "params": last_params,
                "lr": self.ppnet_cfg.last_layer_lr,
                "weight_decay": 1e-4,
            })

            # 4) If there are truly "rest" parameters:
            all_params = set(self.parameters())
            already_in_groups = set(addon_params + proto_params + last_params)
            rest = [p for p in all_params if p not in already_in_groups]
            if len(rest) > 0:
                optimizer_specifications.append({"params": rest})

            # 5) Instantiate via Hydra
            self.optimizer = hydra.utils.instantiate(
                self.optimizer_cfg, 
                optimizer_specifications
            )
    
        if self.scheduler_cfg: 
            num_training_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = 0.067 # hard coded
            num_warmup_steps = num_training_steps * warmup_ratio

            # scheduler = get_cosine_schedule_with_warmup(
            #     optimizer=self.optimizer,
            #     num_warmup_steps=num_warmup_steps,
            #     num_training_steps=num_training_steps
            # )

            scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=num_warmup_steps,
                total_steps=num_training_steps
            )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  # Update at every step
                "frequency": 1,
                "name": "lr_cosine"
            }

            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}
        
        return {"optimizer": self.optimizer}      
    
    def load_pretrained_weights(self, pretrained_weights_path, dataset_name): 
        img_size = (self.target_length, 128)
        #img_size = (128, self.target_length) # should be correcter, but not pretrained this way

        if self.target_length == 512: #esc50, hsn, 5 seconds
            #num_patches = 512 # audioset
            if "xc" in self.pretrained_weights_path or "XCL" in self.pretrained_weights_path:
                num_patches = 256 # birdset
            else:
                num_patches = 512 # audioset

            self.patch_embed = PatchEmbed(img_size, 16, 1, self.embed_dim)
            #self.patch_embed = PatchEmbed_org(img_size, 16, 1, self.embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False) #to load pretrained pos embed
            try:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["model"]
            except:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["state_dict"]

            pretrained_state_dict = {}

            if "encoder_ema.cls_token" not in pre_state_dict: # without mim refiner
                for key, value in pre_state_dict.items():
                    if key.startswith("decoder."):
                        # Skip any key that starts with "decoder."
                        continue
                    elif key.startswith("encoder."):
                        # Remove the "encoder." prefix
                        new_key = key[len("encoder."):]
                    else:
                        # Use the original key if no prefix
                        new_key = key
                    
                    # Add the modified key-value pair to the new state dict
                    pretrained_state_dict[new_key] = value

            else: # with mim refiner
                for key, value in pre_state_dict.items():
                    if key.startswith("decoder."):
                        # Skip any key that starts with "decoder."
                        continue
                    elif key.startswith("encoder."):
                        # Remove the "encoder." prefix
                        continue
                    elif key.startswith("projectors."):
                        continue
                    elif key.startswith("predictors."):
                        continue
                    elif key.startswith("encoder_ema."):
                        # Remove the "encoder_ema." prefix
                        new_key = key[len("encoder_ema."):]
                    else:
                        # Use the original key if no prefix
                        new_key = key
                    
                    # Add the modified key-value pair to the new state dict
                    pretrained_state_dict[new_key] = value

            for k in ['head.weight', 'head.bias']:
                if k in pretrained_state_dict: #and pretrained_state_dict[k].shape != self.state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrained_state_dict[k]
            
            info = self.load_state_dict(pretrained_state_dict, strict=False)

            if not self.class_mask:
                for k in ['head.weight', 'head.bias']:
                    if k in pretrained_state_dict: #and pretrained_state_dict[k].shape != self.state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del pretrained_state_dict[k]

            patch_hw = (img_size[1] // 16, img_size[0] // 16) # 16=patchsize
            #patch_hw = (img_size[0] // 16, img_size[1] // 16) 
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.size(-1), patch_hw, cls_token=True) # not trained, overwrite from sincos
            self.pos_embed.data = torch.from_numpy(pos_embed).float().unsqueeze(0) 

        elif self.target_length == 1024: #audioset, 10 seconds

            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=self.embed_dim, stride=16) # no overlap. stride=img_size=16
           
            if "xc" in self.pretrained_weights_path:
                num_patches = 256 # birdset # does not work right now 
            else:
                num_patches =  num_patches = self.patch_embed.num_patches # audioset
            #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding

            checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
            try:
                pre_state_dict = checkpoint["model"]
            except:
                pre_state_dict = checkpoint["state_dict"]

            pretrained_state_dict = {}

            for key, value in pre_state_dict.items():
                if key.startswith("decoder."):
                    # Skip any key that starts with "decoder."
                    continue
                elif key.startswith("encoder."):
                    # Remove the "encoder." prefix
                    new_key = key[len("encoder."):]
                else:
                    # Use the original key if no prefix
                    new_key = key
                
                # Add the modified key-value pair to the new state dict
                pretrained_state_dict[new_key] = value

            state_dict = self.state_dict()

            for k in ["head.weight", "head.bias"]:
                if k in pretrained_state_dict and pretrained_state_dict[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrained_state_dict[k]

            info = self.load_state_dict(pretrained_state_dict, strict=False)

            try:
                trunc_normal_(self.head.weight, std=2e-5)
            except:
                print("no head")

            # try: 
            #     trunc_normal_(self.ppnet.last_layer.weight, std=2e-5)
            # except:
            #     print("no prototype vectors")

    def calculate_orthogonality_loss(self) -> torch.Tensor:
        """
        Calculate the normalized orthogonality loss.

        Returns:
            torch.Tensor: The normalized orthogonality loss.
        """
        orthogonalities = self.ppnet.get_prototype_orthogonalities()
        orthogonality_loss = torch.norm(orthogonalities)

        # Normalize the orthogonality loss by the number of elements
        normalized_orthogonality_loss = orthogonality_loss / orthogonalities.numel()

        return normalized_orthogonality_loss
    

class VIT_MIM(L.LightningModule):

    def __init__(self, 
                 img_size_x,
                 img_size_y,
                 patch_size,
                 in_chans,
                 embed_dim,
                 global_pool,
                 norm_layer,
                 mlp_ratio,
                 qkv_bias,
                 eps,
                 drop_path,
                 num_heads,
                 depth,
                 pretrained_weights_path, 
                 target_length,
                 mim_cfg,
                 optimizer_cfg
    ):
        L.LightningModule.__init__(self)

        self.encoder = VisionTransformer(
            img_size = (img_size_x, img_size_y),
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            qkv_bias = qkv_bias,
            norm_layer = partial(nn.LayerNorm, eps=eps),
            num_classes = 1,
            drop_path_rate=drop_path,
        )
        self.pretrained_weights_path = pretrained_weights_path
        self.target_length = target_length

        self.load_pretrained_weights(pretrained_weights_path, "XCL")
        
        #self.encoder.load_pretrained_weights(pretrained_weights_path, "XCL")

        self.encoder_ema = copy.deepcopy(self.encoder)
        for p in self.encoder_ema.parameters():
            p.requires_grad = False

        #del self.encoder.head
        # del self.encoder.norm
        # del self.encoder.fc_norm
        # del self.encoder.head_drop

        # del self.encoder_ema.norm
        # del self.encoder_ema.fc_norm
        # del self.encoder_ema.head_drop

        
        proj_dim = mim_cfg.proj_dim
        out_dim = mim_cfg.out_dim
        pred_dim = mim_cfg.pred_dim
        self.queue_size = mim_cfg.queue_size
        self.momentum = mim_cfg.momentum
        self.temperature = mim_cfg.temperature

        # number of last layers to modify based on total depth
        if depth == 24: # Large
            modify_last_n = 8 # MIM-Refiner
        elif depth == 32: # Huge
            modify_last_n = 12 # MIM-Refiner
        elif depth == 12: # Base
            modify_last_n = 4 # hmm let's experiment with 4 or 6 ?
        else:
            modify_last_n = int(0.35*depth) # default, maybe last 35% of layers?

        self.modify_last_n = modify_last_n
        self.start_modify = depth - self.modify_last_n

        # Create MLP projectors for the last N layers
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, proj_dim, bias=False), # don't use bias as it is followed by BN
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim, bias=False),
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, out_dim, bias=False),
                nn.BatchNorm1d(out_dim, affine=False),
            ) for _ in range(self.modify_last_n)
        ])

        self.predictors = nn.ModuleList([
            Predictor(hidden_dim=pred_dim, out_dim=out_dim) for _ in range(self.modify_last_n)
        ]) 

        # load pretrained weights here 
        # second step: 20 epochs with different learning rate, 30 for mim update. 

        self.save_hyperparameters()
        self.img_size = (img_size_x, img_size_y)
        self.global_pool = global_pool

        norm_layer = partial(nn.LayerNorm, eps=eps)
        self.fc_norm = norm_layer(embed_dim)

        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias 
        self.optimizer_cfg = optimizer_cfg

        

        self.queues = [torch.randn(self.queue_size, out_dim).to("cuda") for _ in range(self.modify_last_n)]

        

    # def forward_features(self, x):
    #     B = x.shape[0]
    #     x = self.patch_embed(x) # batch, patch, embed
    #     x = x + self.pos_embed[:, 1:, :] 
    #     cls_token = self.cls_token + self.pos_embed[:, :1, :]
    #     cls_tokens = cls_token.expand(B, -1, -1) 
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     x = self.pos_drop(x)        

    #     for blk in self.blocks:
    #         x = blk(x)

    #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
    #     outcome = self.fc_norm(x)


    #     return outcome

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     pred = self.head(x)
    #     return pred 


    def forward(self, x):
        B = x.shape[0]
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed[:, 1:, :] 
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.encoder.pos_drop(x)      
        # # pre-transformer layers
        # x = self.vit.to_patch_embedding(x)
        # x = self.vit.dropout(x)

        # transformer layers
        zs = []
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)

            if i >= self.start_modify:
                #intermediate_x = x[:,0] #cls_token
                intermediate_x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                intermediate_x = self.fc_norm(intermediate_x)
                # projector
                z = self.projectors[i - self.start_modify](intermediate_x)
                zs.append(z)

        # # post-transformer layers
        # x = self.encoder.to_latent(x)
        # x = self.vit.mlp_head(x)

        return zs


    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        zs = self(audio)

        contrastive_loss = 0 

        NN_zs = []
        for idx, (z, queue, predictor) in enumerate(zip(zs, self.queues, self.predictors)):
            NN_z = self.NN(z, queue)
            NN_zs.append(NN_z) # retrieve nearest neighbor for each layer
            self.queues[idx] = self.update_queue(queue, z) # update queue for each layer

            h = predictor(z)
            contrastive_loss += loss_fn(NN_z, h, self.temperature)

        contrastive_loss /= len(zs) # average over layers, maybe give different weights to different layers?

        self.update_ema(self.encoder, self.encoder_ema, self.momentum) # keeping track of EMA for downstream tasks

        self.log("train_contrastive_loss", contrastive_loss, on_step=True, on_epoch=True, prog_bar=True)

        return contrastive_loss
    
    def update_queue(self, queue, new_embeddings):
        return torch.cat((queue, new_embeddings.detach()), dim=0)[-self.queue_size:]  # Keep only the most recent `queue_size` entries
    
    def update_ema(self, model, ema_model, decay):
        with torch.no_grad():
            for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                ema_param.data = decay * ema_param.data + (1 - decay) * param.data

    def NN(self, key, Queue):
        # Nearest Neighbor function to retrieve the positive in the queue
        key = F.normalize(key, dim=1)
        Queue = F.normalize(Queue, dim=1)
        similarity = torch.mm(key, Queue.t())
        nearest_neighbors = similarity.max(dim=1)[1]
        return Queue[nearest_neighbors]

    def configure_optimizers(self):
        #heuristic:
        # eff_batch_size = self.trainer.accumulate_grad_batches * self.trainer.num_devices * self.train_batch_size
        # self.optimizer_cfg["lr"] = self.optimizer_cfg["lr"] * eff_batch_size / 48
        # print("effective learning rate:", self.optimizer_cfg["lr"], self.layer_decay)

        params = list(self.encoder.parameters())

        for i in range(len(self.predictors)):
            params += list(self.projectors[i].parameters()) + list(self.predictors[i].parameters())
        
        self.optimizer = torch.optim.AdamW(
            lr=self.optimizer_cfg.target["lr"],
            weight_decay=self.optimizer_cfg.target["weight_decay"],
            betas=(0.9, 0.95),
            params=params
        )
    
        num_training_steps = self.trainer.estimated_stepping_batches
        warmup_ratio = 0.2 # hard coded
        num_warmup_steps = num_training_steps * warmup_ratio

        scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=num_warmup_steps,
            total_steps=num_training_steps
        )

        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",  # Update at every step
            "frequency": 1,
            "name": "lr_cosine"
        }

        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}
        
    def load_pretrained_weights(self, pretrained_weights_path, dataset_name): 
        img_size = (self.target_length, 128)
        #img_size = (128, self.target_length) # should be correcter, but not pretrained this way

        if self.target_length == 512: #esc50, hsn, 5 seconds
            #num_patches = 512 # audioset
            if "xc" in self.pretrained_weights_path or "XCL" in self.pretrained_weights_path:
                num_patches = 256 # birdset
            else:
                num_patches = 512 # audioset

            self.encoder.patch_embed = PatchEmbed(img_size, 16, 1, self.encoder.embed_dim)
            #self.patch_embed = PatchEmbed_org(img_size, 16, 1, self.embed_dim)
            self.encoder.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.encoder.embed_dim), requires_grad=False) #to load pretrained pos embed
            try:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["model"]
            except:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["state_dict"]

            pretrained_state_dict = {}

            for key, value in pre_state_dict.items():
                if key.startswith("decoder."):
                    # Skip any key that starts with "decoder."
                    continue
                elif key.startswith("encoder."):
                    # Remove the "encoder." prefix
                    new_key = key[len("encoder."):]
                else:
                    # Use the original key if no prefix
                    new_key = key
                
                # Add the modified key-value pair to the new state dict
                pretrained_state_dict[new_key] = value
            
            info = self.encoder.load_state_dict(pretrained_state_dict, strict=False)

            patch_hw = (img_size[1] // 16, img_size[0] // 16) # 16=patchsize
            #patch_hw = (img_size[0] // 16, img_size[1] // 16) 
            pos_embed = get_2d_sincos_pos_embed_flexible(self.encoder.pos_embed.size(-1), patch_hw, cls_token=True) # not trained, overwrite from sincos
            self.encoder.pos_embed.data = torch.from_numpy(pos_embed).float().unsqueeze(0) 

        elif self.target_length == 1024: #audioset, 10 seconds

            self.encoder.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=self.embed_dim, stride=16) # no overlap. stride=img_size=16
           
            if "xc" in self.pretrained_weights_path:
                num_patches = 256 # birdset # does not work right now 
            else:
                num_patches =  num_patches = self.patch_embed.num_patches # audioset
            #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
            self.encoder.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding

            checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
            try:
                pre_state_dict = checkpoint["model"]
            except:
                pre_state_dict = checkpoint["state_dict"]

            pretrained_state_dict = {}

            for key, value in pre_state_dict.items():
                if key.startswith("decoder."):
                    # Skip any key that starts with "decoder."
                    continue
                elif key.startswith("encoder."):
                    # Remove the "encoder." prefix
                    new_key = key[len("encoder."):]
                else:
                    # Use the original key if no prefix
                    new_key = key
                
                # Add the modified key-value pair to the new state dict
                pretrained_state_dict[new_key] = value

            state_dict = self.state_dict()

            for k in ["head.weight", "head.bias"]:
                if k in pretrained_state_dict and pretrained_state_dict[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrained_state_dict[k]

            self.encoder.load_state_dict(pretrained_state_dict, strict=False)

            trunc_normal_(self.head.weight, std=2e-5)

class Predictor(nn.Module):
    def __init__(self, hidden_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(out_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def normalize(x, dim=-1):
    return F.normalize(x, p=2, dim=dim)

def loss_fn(nn, p, temperature=0.1):
    nn = normalize(nn, dim=1)
    p = normalize(p, dim=1)
    logits = torch.matmul(nn, p.T)
    logits /= temperature
    labels = torch.arange(p.size(0), device=p.device)
    return F.cross_entropy(logits, labels)