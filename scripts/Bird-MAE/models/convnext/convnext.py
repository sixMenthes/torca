import lightning as L
import torch
import hydra
from torchmetrics import MetricCollection
from util.lr_decay import param_groups_lrd
import torch
from transformers import AutoConfig, ConvNextForImageClassification

from ..components.cosine_warmup import CosineWarmupScheduler

class ConvNext(L.LightningModule):
    """
    ConvNext model for audio classification.
    """

    def __init__(
        self,
        num_channels,
        num_classes,
        optimizer,
        scheduler,
        hf_checkpoint, 
        loss,
        metric_cfg,
        model_dir
    ):
  
        super().__init__()
        self.save_hyperparameters()
        self.hf_checkpoint = hf_checkpoint
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.model_dir = model_dir

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

        self.train_metric = metric.clone()
        self.val_metric = metric.clone()
        self.test_metric = metric.clone()

        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []


        if self.hf_checkpoint: 
            self.model = ConvNextForImageClassification.from_pretrained(
                self.hf_checkpoint,
                num_labels=self.num_classes,
                num_channels=self.num_channels,
                cache_dir=self.model_dir,
                ignore_mismatched_sizes=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                "facebook/convnext-base-224-22k",
                num_labels=self.num_classes,
                num_channels=self.num_channels,
            )
            self.model = ConvNextForImageClassification(config)
    

    def forward(self, x):
        output = self.model(x)
        logits = output.logits

        return logits

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

        pred = torch.sigmoid(pred)
        self.val_predictions.append(pred.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

        #self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_predictions)
        targets = torch.cat(self.val_targets)
        metric = self.val_metric(preds, targets)
        self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)
        print("val metric:", metric.detach().cpu().item())

        self.val_predictions = []
        self.val_targets = []
    
    def test_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        self.mask_t_prob = 0.0
        self.mask_f_prob = 0.0 #fix later!

        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())
        
        pred = torch.sigmoid(pred)
        
        self.test_predictions.append(pred.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_test_epoch_end(self):
        preds = torch.cat(self.test_predictions)
        targets = torch.cat(self.test_targets)
        self.test_metric(preds, targets)
        self.log(f'test_{self.test_metric.__class__.__name__}', self.test_metric, on_epoch=True, prog_bar=True)

        self.test_add_metrics(preds, targets)
        for name, metric in self.test_add_metrics.items():
            self.log(f'test_{name}', metric, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):

        #heuristic:
        # eff_batch_size = self.trainer.accumulate_grad_batches * self.trainer.num_devices * self.train_batch_size
        # self.optimizer_cfg["lr"] = self.optimizer_cfg["lr"] * eff_batch_size / 256
        # print("effective learning rate:", self.optimizer_cfg["lr"], self.layer_decay)

        if self.layer_decay:
            params = param_groups_lrd(
                model=self,
                weight_decay=self.optimizer_cfg["weight_decay"],
               # no_weight_decay_list=self.no_weight_decay(), was ist das überhaupt
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