import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAveragePrecision

from ..components.beats import BEATs, BEATsConfig
from ..components.cosine_warmup import CosineWarmupScheduler
from ..components.fsq import FSQ
from ..components.nn_utils import clones, get_alibi, make_mask
from ..components.transformer_stack import TransformerStack


def load_beats(beats_ckpt):
    """Load a frozen BEATs encoder from a checkpoint."""
    checkpoint = torch.load(beats_ckpt, map_location="cpu")
    cfg_beats = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg_beats)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class Quantizer(nn.Module):
    """Frozen BEATs -> linear proj -> FSQ. Returns continuous codes + indices."""

    def __init__(self, beats_ckpt, levels):
        super().__init__()
        self.encoder = load_beats(beats_ckpt)
        self.proj = nn.Linear(768, len(levels))
        self.fsq = FSQ(levels)

    def forward(self, soundwave, mask):
        h, m = self.encoder.extract_features(soundwave, padding_mask=mask)
        h = self.proj(h)
        q = self.fsq.quantize(h)
        indices = self.fsq.codes_to_indices(q)
        return q, indices, m

    def train(self, mode=True):
        # Keep BEATs in eval regardless of module mode: its weights are frozen
        # (requires_grad_=False) and its dropout/LN behaviour must stay fixed.
        super().train(mode)
        self.encoder.eval()
        return self


class FSQNet(L.LightningModule):
    """
    BEATs (frozen) -> FSQ -> linear embed of continuous codes -> ALiBi transformer
    stack -> dual heads (masked FSQ-index prediction + classification).

    Ported from the standalone `Torca` model (no_lookup branch). Objective:
        loss = mask_loss + class_loss_weight * clas_loss
                          + diversity_loss_weight * diversity_loss
    """

    def __init__(
        self,
        # --- data / patch grid ---
        num_mel_bins=128,
        patch_size=16,
        time_step=10,
        chunk_duration=3.0,
        num_classes=8,
        # --- model ---
        d_model=256,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        dropout=0.1,
        fsq_levels=(8, 6, 6),
        # --- masking ---
        mask_prob=0.15,
        span_len=3,
        # --- loss weights ---
        class_loss_weight=1.0,
        diversity_loss_weight=0.1,
        # --- optim ---
        lr=3e-4,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.0,
        warmup_ratio=0.05,
        # --- paths / extras ---
        beats_ckpt="../models/BEATs_iter3.pt",
        class_weights=None,
        label_map=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        fsq_levels = tuple(fsq_levels)
        self.fsq_levels = fsq_levels
        self.codebook_size = math.prod(fsq_levels)

        self.grid_freq = int(num_mel_bins // patch_size)
        self.grid_time = int((chunk_duration * 1000) / time_step // patch_size)
        self.seq_len = self.grid_freq * self.grid_time

        self.mask_prob = mask_prob
        self.span_len = span_len
        self.num_classes = num_classes
        self.class_loss_weight = class_loss_weight
        self.diversity_loss_weight = diversity_loss_weight

        self.lr = lr
        self.betas = tuple(betas)
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio

        self.register_buffer(
            "alibi_bias", get_alibi(num_heads, self.grid_freq, self.grid_time)
        )
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.to(torch.float32))
        else:
            self.class_weights = None

        self.quant = Quantizer(beats_ckpt, fsq_levels)
        # no_lookup: embed the CONTINUOUS FSQ codes directly (was nn.Embedding lookup)
        self.emb = nn.Linear(len(fsq_levels), d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.trans = clones(
            TransformerStack(
                d_model=d_model,
                number_heads=num_heads,
                d_ff=d_ff,
                bias=self.alibi_bias,
                dropout=dropout,
            ),
            num_layers,
        )
        self.classif_head = nn.Linear(d_model, num_classes)
        self.masked_head = nn.Linear(d_model, self.codebook_size)

        self._init_weights()

        self.idx_to_label = (
            {i: name for name, i in label_map.items()} if label_map else None
        )
        self.val_map = MulticlassAveragePrecision(
            num_classes=num_classes, average="macro"
        )

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.quant.proj.weight, std=1)
        nn.init.zeros_(self.quant.proj.bias)

    def _diversity_loss(self, indices):
        """
        Negative entropy of the used-index histogram (maximize codebook usage).

        NOTE: `indices` are discrete (argmax-like) FSQ codes, so this term is
        NON-differentiable w.r.t. the model parameters as written -- bincount of
        int indices has no gradient. Ported faithfully from the no_lookup branch,
        but to actually regularize codebook usage it should be computed from a
        soft/continuous assignment. Flagged for review before relying on it.
        """
        K = self.codebook_size
        counts = torch.bincount(indices.flatten().long(), minlength=K).float()
        p = counts / counts.sum()
        entropy = -(p * (p + 1e-9).log()).sum()
        return -entropy

    def _encode(self, x, padding_mask):
        q, indices, patch_mask = self.quant(x, padding_mask)
        h = self.emb(q)
        batch_size = x.size(0)
        tgt = indices.long()
        masks = torch.stack(
            [
                make_mask(
                    seq_len=self.seq_len,
                    grid_freq=self.grid_freq,
                    obj_masked=self.mask_prob,
                    span=self.span_len,
                )
                for _ in range(batch_size)
            ]
        ).to(h.device)
        h = h.clone()
        h[masks] = self.mask_token
        for layer in self.trans:
            h = layer(h, padding_mask=patch_mask)
        return h, masks, tgt, indices

    def _shared_step(self, batch):
        padded, mask, labels = batch
        h, masks, tgt, indices = self._encode(padded, mask)
        clas_logits = self.classif_head(h.mean(dim=1))  # mean-pool over patches
        mask_logits = self.masked_head(h)
        mask_loss = F.cross_entropy(mask_logits[masks], tgt[masks])
        clas_loss = F.cross_entropy(clas_logits, labels, weight=self.class_weights)
        diversity_loss = self._diversity_loss(indices)
        total = (
            mask_loss
            + self.class_loss_weight * clas_loss
            + self.diversity_loss_weight * diversity_loss
        )
        parts = {
            "mask_loss": mask_loss,
            "clas_loss": clas_loss,
            "diversity_loss": diversity_loss,
            "n_unique_idx": indices.unique().numel(),
        }
        return total, clas_logits, labels, parts

    def training_step(self, batch, batch_idx):
        total, _, _, parts = self._shared_step(batch)
        self.log("train_loss", total, on_step=True, on_epoch=True, prog_bar=True)
        for name, value in parts.items():
            self.log(f"train_{name}", float(value), on_step=True, on_epoch=True)
        return total

    def validation_step(self, batch, batch_idx):
        total, clas_logits, labels, parts = self._shared_step(batch)
        self.log("val_loss", total, on_step=False, on_epoch=True, prog_bar=True)
        for name, value in parts.items():
            self.log(f"val_{name}", float(value), on_step=False, on_epoch=True)
        self.val_map.update(clas_logits.softmax(dim=-1), labels)

    def on_validation_epoch_end(self):
        self.log("val_mAP", self.val_map.compute(), prog_bar=True)
        self.val_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=int(self.warmup_ratio * total_steps),
            total_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr_cosine",
            },
        }
