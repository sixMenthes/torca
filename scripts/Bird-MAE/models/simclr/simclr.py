import lightning as L
import torch
import torch.nn as nn
import hydra
from torchmetrics import MetricCollection
from ..components.cosine_warmup import CosineWarmupScheduler
from ..ppnet.ppnet import PPNet

#### SimCLR
#from https://huggingface.co/ilyassmoummad/ProtoCLR
from .cvt import ConvolutionalVisionTransformer
from .melspectrogram import MelSpectrogramProcessor

class SimCLR(L.LightningModule, nn.Module):

    def __init__(self,
                 num_classes,
                 optimizer,
                 scheduler,
                 loss,
                 metric_cfg,
                 model_spec_cfg,
                 proto_clr_weights_path,
    ):

        L.LightningModule.__init__(self)
        self.cuda()
        self.preprocessor = MelSpectrogramProcessor(device=self.device)
        self.encoder = ConvolutionalVisionTransformer(spec=model_spec_cfg)
        self.encoder.load_state_dict(torch.load(proto_clr_weights_path, map_location="cpu"))
        self.encoder.cuda()

        self.embed_dim = 384
        self.head = nn.Linear(self.embed_dim, num_classes)

        self.save_hyperparameters()

        self.loss = hydra.utils.instantiate(loss)
        self.optimizer = None
        self.optimizer_cfg = optimizer.target
        self.train_batch_size = optimizer.extras.train_batch_size
        self.layer_decay = optimizer.extras.layer_decay
        self.decay_type = optimizer.extras.decay_type
        self.scheduler_cfg = scheduler

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

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.preprocessor.process(x)
        features = self.encoder(x)
        pred = self.head(features)
        return pred

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

        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())

        self.val_predictions.append(pred.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

        #self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_predictions)
        targets = torch.cat(self.val_targets)
        metric = self.val_metric(preds, targets)
        self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=False)
        print("val metric:", metric.detach().cpu().item())

        self.val_add_metrics(preds, targets)
        for name, metric in self.val_add_metrics.items():
            self.log(f'valid_{name}', metric, on_epoch=True, prog_bar=False)

        self.val_predictions = []
        self.val_targets = []

    def test_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        pred = self(audio)

        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())

        self.test_predictions.append(pred.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False)

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_predictions)
        targets = torch.cat(self.test_targets)
        self.test_metric(preds, targets)
        self.log(f'test_{self.test_metric.__class__.__name__}', self.test_metric, on_epoch=True, prog_bar=True)

        self.test_add_metrics(preds, targets)
        for name, metric in self.test_add_metrics.items():
            self.log(f'test_{name}', metric, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        params = []
        params += list(self.encoder.parameters())
        params += list(self.head.parameters())

        self.optimizer = hydra.utils.instantiate(
            self.optimizer_cfg,
            params=params)

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
                "interval": "step",  # Update at every step
                "frequency": 1,
                "name": "lr_cosine"
            }

            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}

        return {"optimizer": self.optimizer}


class SimCLR_ppnet(SimCLR):
    def __init__(self, ppnet_cfg, *args, **kwargs, ):
        super().__init__(*args, **kwargs)

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

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.preprocessor.process(x)

        for i in range(self.encoder.num_stages):
            x, cls_token = getattr(self.encoder, f'stage{i}')(x)

        if self.ppnet_cfg.focal_similarity == True:
            x_cls = cls_token.squeeze().unsqueeze(-1).unsqueeze(-1) # [64, 384, 1, 1]
            x_patch = x # [64, 384, 8, 19]
            x = x_patch - x_cls
        else:
            x = x

        logits, _ = self.ppnet(x)

        return logits

    def configure_optimizers(self):
        optimizer_specifications = []

        # 1) Add the add_on_layers group
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
            warmup_ratio = 0.067  # hard coded
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

        return {"optimizer": self.optimizer}