"""EMA self-distillation (iBOT-shaped, FSQ targets) for domain adaptation.

The treatment arm of the ablation: does masked self-distillation convert the
recording-condition confound in frozen audio-SSL features into ecotype/call-type
signal? One recipe, both backbones, so adaptation is a homogeneous on/off treatment
rather than a different objective per backbone.

  teacher (EMA weights)  <- CLEAN view,      unmasked  -> FSQ indices   (targets)
  student (live weights) <- AUGMENTED view,  masked    -> codebook logits

Loss = CE at masked+valid positions (invariance) + VICReg variance/covariance on the
pre-FSQ projections (anti-collapse). The augmentation asymmetry is the de-confounding
lever: the student sees the clip buried in noise from a DIFFERENT hydrophone, so the
only way to predict the teacher's tokens is to ignore the channel.

Five non-negotiables from the design, all load-bearing:
 1. EMA covers the WHOLE teacher path (encoder + projector), not the encoder alone —
    otherwise the target keeps moving under a live projector and there is no fixed
    thing to distil into.
 2. Masking asymmetry: teacher full, student masked, CE at masked positions only.
 3. PCEN (when used) stays FIXED — backprop into it reintroduces the r->0 collapse.
    Not wired here: this path uses the fixed kaldi fbank front-end.
 4. VICReg variance as anti-collapse guard — EMA makes collapse non-attracting, not
    impossible.
 5. FSQ on both backbones, so the objective is identical and the tokenizer falls out.
"""

import math

import hydra
import lightning as L
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf
from timm.layers import mlp

from models.components.beats_encoder import BEATsEncoder
from models.components.birdmae_encoder import BirdMAEEncoder
from models.components.fsq import FSQ
from models.components.selfdistill_loss import (
    code_counts,
    codebook_diversity,
    codebook_logits,
    entropy_bits,
    masked_ce,
    sample_student_mask,
    saturation_frac,
    valid_token_mask,
    vicreg,
)


def _plain(cfg):
    """Config subtree -> plain python container, interpolations resolved.

    Accepts either a hydra DictConfig (normal run) or the plain dict that comes back
    out of a checkpoint's hyper_parameters (reload), so __init__ is symmetric between
    the two paths.
    """
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


class MIMDistillation(L.LightningModule):
    def __init__(self, encoder_cfg, tokenizer_cfg, optimizer_cfg, distill_cfg):
        super().__init__()

        # Resolve interpolations and drop to plain containers BEFORE storing. Two
        # reasons, both load-bearing:
        #  * resolve first, while the nodes still have their parent config attached —
        #    ${data.transform.target_length} is unresolvable once detached;
        #  * a checkpoint holding omegaconf DictConfigs cannot be reloaded, because
        #    load_from_checkpoint calls torch.load(weights_only=True) (the PyTorch>=2.6
        #    default) which refuses to unpickle them. Plain dicts make the checkpoint
        #    self-contained and loadable anywhere.
        encoder_cfg = _plain(encoder_cfg)
        tokenizer_cfg = _plain(tokenizer_cfg)
        optimizer_cfg = _plain(optimizer_cfg)
        distill_cfg = _plain(distill_cfg)
        self.save_hyperparameters(
            {
                "encoder_cfg": encoder_cfg,
                "tokenizer_cfg": tokenizer_cfg,
                "optimizer_cfg": optimizer_cfg,
                "distill_cfg": distill_cfg,
            }
        )
        # back to OmegaConf for attribute access / hydra.utils.instantiate
        encoder_cfg = OmegaConf.create(encoder_cfg)
        tokenizer_cfg = OmegaConf.create(tokenizer_cfg)
        optimizer_cfg = OmegaConf.create(optimizer_cfg)
        distill_cfg = OmegaConf.create(distill_cfg)

        self.name = encoder_cfg.name
        levels = list(tokenizer_cfg.levels)

        # Student and teacher are each ONE module holding encoder+projector, so the
        # EMA provably covers the whole teacher path (non-negotiable 1) and the two
        # halves cannot drift apart.
        self.student = self._build_branch(encoder_cfg, tokenizer_cfg, levels)
        # Built fresh and synced by state_dict rather than deepcopy'd: BEATs applies
        # weight_norm to its conv positional embedding, and weight-norm'd modules
        # cannot be deepcopied (the reparametrised weight is a non-leaf tensor).
        # Rebuilding is backbone-agnostic and gives a bit-identical starting teacher.
        self.teacher = self._build_branch(encoder_cfg, tokenizer_cfg, levels)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        # FSQ holds no learned parameters (only buffers), so student and teacher
        # quantise on the same fixed grid by construction — "shared head" for free.
        self.fsq = FSQ(levels)

        self.optimizer_cfg = optimizer_cfg
        self.mask_ratio = distill_cfg.mask_ratio
        self.temperature = distill_cfg.temperature
        self.ema_decay = distill_cfg.ema_decay
        self.ema_decay_end = distill_cfg.get("ema_decay_end", None)
        self.mask_padding = distill_cfg.get("mask_padding", True)
        self.ce_weight = distill_cfg.get("ce_weight", 1.0)
        self.var_weight = distill_cfg.get("var_weight", 1.0)
        self.cov_weight = distill_cfg.get("cov_weight", 0.04)
        self.within_clip_var_weight = distill_cfg.get("within_clip_var_weight", 0.0)
        self.gamma = distill_cfg.get("gamma", 1.0)
        self.val_seed = distill_cfg.get("val_seed", 59)

        # Entropy penalty on code usage. VICReg's variance term measures spread, not
        # coverage, so it reads healthy (vic_var = 0) while tanh saturation parks every
        # token on a cube corner and the tokenizer carries ~1 bit. Default 0.0 keeps the
        # objective unchanged until switched on. Watch train/saturation_frac.
        self.diversity_weight = distill_cfg.get("diversity_weight", 0.0)
        self.diversity_sample_weight = distill_cfg.get("diversity_sample_weight", 0.0)
        # null = score coverage at the CE's own temperature, which measurement says is
        # right: softening both breaks the ordering (collapsed scores better than
        # healthy) and shrinks the gradient 10x. See codebook_diversity's docstring.
        self.diversity_temperature = distill_cfg.get("diversity_temperature", None)

        # Epoch-level code-usage histograms. Entropy of pooled counts is not the mean
        # of per-batch entropies, so the headline number has to come from an
        # accumulator rather than from Lightning averaging per-step values.
        # persistent=False: transient statistics, not model state — keep them out of
        # checkpoints. NOTE single-process only; under DDP these need an all_reduce.
        K = self.fsq.codebook_size
        self.register_buffer("train_code_counts", torch.zeros(K, dtype=torch.long),
                             persistent=False)
        self.register_buffer("val_code_counts", torch.zeros(K, dtype=torch.long),
                             persistent=False)

    def _build_branch(self, encoder_cfg, tokenizer_cfg, levels):
        """One encoder+projector branch. Called twice (student, teacher)."""
        enc_kwargs = {k: v for k, v in encoder_cfg.items() if k != "name"}
        if self.name == "BEATs":
            encoder = BEATsEncoder(**enc_kwargs)
        elif self.name == "BirdMAE":
            encoder = BirdMAEEncoder(**enc_kwargs)
        else:
            raise ValueError(f"unknown encoder {self.name}")

        # in_features is the EMBEDDING dim (768), not the input length: the projector
        # maps each patch token down to the L quantised dimensions.
        projector = mlp.Mlp(
            encoder.embed_dim,
            hidden_features=tokenizer_cfg.get("hidden_dim", 512),
            out_features=len(levels),
        )
        return nn.ModuleDict({"encoder": encoder, "projector": projector})

    # ---------------------------------------------------------------- forward paths

    def _project(self, module_dict, wave, mask=None):
        """wave -> (B, N, L) pre-quantisation projections."""
        tokens = module_dict["encoder"].tokens(wave, mask=mask)     # (B, N, D)
        return module_dict["projector"](tokens)                     # (B, N, L)

    def forward(self, wave, mask=None):
        return self._project(self.student, wave, mask=mask)

    @torch.no_grad()
    def teacher_forward(self, wave):
        """Clean view through the EMA path -> discrete FSQ indices (the targets).

        eval() matters: dropout/droppath active in the teacher would make the target
        stochastic, so the student would chase noise it cannot possibly predict.
        """
        self.teacher.eval()
        z = self._project(self.teacher, wave, mask=None)
        return self.fsq.codes_to_indices(self.fsq.quantize(z)).long()

    # ---------------------------------------------------------------- masks

    def _valid_mask(self, batch, num_patches, device):
        """(B, N) bool: positions backed by audio a human actually heard.

        90% of DCLDE clips are shorter than the 3 s window and are zero-padded at the
        tail. Letting that padding into the loss would hand the student a constant
        "silence" code to predict for ~44% of positions — the loss would fall while
        nothing was learned. Set distill_cfg.mask_padding=false only once the clips
        are re-sliced as centred context windows (then every position IS real).
        """
        B = batch["student"].shape[0]
        if not self.mask_padding or "n_valid" not in batch:
            return torch.ones(B, num_patches, dtype=torch.bool, device=device)
        return valid_token_mask(
            batch["n_valid"].to(device),
            num_patches,
            self.student["encoder"].freq_patches,
            self.student["encoder"].sample_rate,
        )

    # ---------------------------------------------------------------- step

    def _diversity(self, logits, valid):
        """Coverage penalty, evaluated at its own (softer) temperature.

        `logits` already carry the CE's 1/temperature, so rescaling by
        temperature/diversity_temperature recovers -d^2/diversity_temperature without
        recomputing the distances.
        """
        scale = 1.0
        if self.diversity_temperature:
            scale = self.temperature / self.diversity_temperature
        return codebook_diversity(
            logits, valid,
            softmax_scale=scale,
            sample_entropy_weight=self.diversity_sample_weight,
        )

    def training_step(self, batch, batch_idx):
        # collate_fn_skip returns None when EVERY clip in the batch failed to load
        # (missing file, unreadable wav). Returning None here is lightning's own
        # "skip this batch"; without it the None propagates into the subscript below
        # and the run dies with a TypeError that says nothing about missing clips.
        # Common while the prestage is still running, or after a clip_duration change.
        if batch is None:
            return None
        student_wave, teacher_wave = batch["student"], batch["teacher"]

        # Teacher first: it fixes N (BEATs has no static num_patches — the token count
        # follows the waveform) and the targets the student is scored against.
        target_indices = self.teacher_forward(teacher_wave)          # (B, N)
        num_patches = target_indices.shape[1]

        valid = self._valid_mask(batch, num_patches, student_wave.device)
        student_mask = sample_student_mask(valid, self.mask_ratio)

        z = self(student_wave, mask=student_mask)                    # (B, N, L)
        logits = codebook_logits(z, self.fsq, self.temperature)      # (B, N, K)

        ce = masked_ce(logits, target_indices, student_mask)
        vic, vic_parts = vicreg(
            z, valid,
            gamma=self.gamma,
            var_weight=self.var_weight,
            cov_weight=self.cov_weight,
            within_clip_var_weight=self.within_clip_var_weight,
        )
        div, div_parts = self._diversity(logits, valid)
        loss = self.ce_weight * ce + vic + self.diversity_weight * div

        with torch.no_grad():
            # Codebook usage is THE health metric: FSQNet's earlier failure was a
            # collapsed codebook that a falling loss curve happily concealed.
            counts = code_counts(target_indices, valid, self.fsq.codebook_size)
            self.train_code_counts += counts
            used = int((counts > 0).sum())
            acc = (logits.argmax(-1) == target_indices)[student_mask].float().mean()

        self.log_dict(
            {
                "train/loss": loss,
                "train/ce": ce,
                "train/vic_var": vic_parts["vic_var"],
                "train/vic_cov": vic_parts["vic_cov"],
                "train/masked_acc": acc,
                "train/valid_frac": valid.float().mean(),
                "train/ema_decay": self._current_decay(),
            },
            prog_bar=True, on_step=True, on_epoch=True, batch_size=student_wave.shape[0],
        )
        # Per-BATCH codebook stats, logged separately with on_epoch=False. Inside
        # log_dict they would inherit on_epoch=True, and Lightning would append
        # "_epoch" — colliding with the pooled epoch metrics of the same name written
        # by on_train_epoch_end, silently mixing two different quantities. They are
        # different: "how much of the codebook does one batch touch" (here) vs "how
        # much does the whole epoch touch" (there). Entropy additionally cannot be
        # epoch-averaged at all, being concave.
        self.log_dict(
            {
                "train/codes_used_batch": float(used),
                "train/codebook_frac_batch": used / self.fsq.codebook_size,
                "train/token_bits_batch": entropy_bits(counts),
                # Logged whatever diversity_weight is: saturation_frac near 1.0 with
                # vic_var at 0.0 is the signature of the collapse the VICReg terms
                # cannot see, and it costs nothing to watch for it.
                "train/saturation_frac": saturation_frac(z, self.fsq, valid),
                "train/diversity": div_parts["diversity"],
                "train/soft_bits": div_parts["soft_bits"],
            },
            on_step=True, on_epoch=False, batch_size=student_wave.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Same objective on a held-out RECORDING CONDITION (CarmanahPt).

        The mask is drawn from a generator seeded by batch index, so the same patches
        are hidden every epoch. Combined with the dataset's deterministic augmentation
        that makes the val curve move only when the model moves — otherwise a fresh
        noise draw and a fresh mask each epoch add variance that swamps the signal you
        are trying to read.
        """
        if batch is None:                      # see training_step
            return None
        student_wave, teacher_wave = batch["student"], batch["teacher"]

        target_indices = self.teacher_forward(teacher_wave)
        valid = self._valid_mask(batch, target_indices.shape[1], student_wave.device)

        gen = torch.Generator(device=student_wave.device)
        gen.manual_seed(self.val_seed + batch_idx)
        student_mask = sample_student_mask(valid, self.mask_ratio, generator=gen)

        z = self(student_wave, mask=student_mask)
        logits = codebook_logits(z, self.fsq, self.temperature)

        ce = masked_ce(logits, target_indices, student_mask)
        vic, vic_parts = vicreg(
            z, valid, gamma=self.gamma, var_weight=self.var_weight,
            cov_weight=self.cov_weight,
            within_clip_var_weight=self.within_clip_var_weight,
        )
        # The diversity term MUST be here, exactly as in training_step. It was added to
        # the objective later and this line was not updated, which made val/loss a
        # DIFFERENT objective from the one being optimised — and specifically the one
        # objective under which collapse looks good.
        #
        # Measured (see the sweep in configs/module/network/mim_distillation.yaml): a
        # collapsed tokenizer using 9 codes of 1000 scores ce 0.699 where a healthy one
        # scores 2.776, and the VICReg guards read healthy throughout collapse, because
        # saturating every coordinate gives large per-dimension variance and near-zero
        # covariance. So on `ce + vic` alone, collapse wins by a wide margin. Since
        # model_checkpoint monitors val/loss with mode=min, the callback would have
        # preferentially saved the collapsed epoch — and that checkpoint is what
        # probe_selfdistill.py reads for every adapted cell.
        div, div_parts = self._diversity(logits, valid)
        loss = self.ce_weight * ce + vic + self.diversity_weight * div

        counts = code_counts(target_indices, valid, self.fsq.codebook_size)
        self.val_code_counts += counts
        used = int((counts > 0).sum())
        acc = (logits.argmax(-1) == target_indices)[student_mask].float().mean()
        self.log_dict(
            {
                "val/loss": loss,
                "val/ce": ce,
                "val/diversity": div_parts["diversity"],
                # per-batch support, averaged; the pooled epoch figure is val/codebook_frac
                "val/codebook_frac_batch": used / self.fsq.codebook_size,
                # NOT a success metric. High masked_acc is a COLLAPSE signature: the
                # collapsed cell above scored 0.821 against the healthy cell's 0.378,
                # because predicting one of nine codes is easy. Read it together with
                # val/codebook_frac and token_bits, never on its own.
                "val/masked_acc": acc,
            },
            prog_bar=True, on_step=False, on_epoch=True,
            batch_size=student_wave.shape[0], sync_dist=True,
        )
        return loss

    # ---------------------------------------------------------------- codebook health

    def _log_epoch_codebook(self, stage, counts):
        """Entropy and support of the code distribution pooled over a whole epoch.

        `token_bits` is the real capacity of the tokenizer: log2(K) is only a ceiling,
        reached when codes are used equally often. `token_bits_frac` normalises by that
        ceiling, so runs with DIFFERENT `levels` are directly comparable — which
        `codebook_frac` and raw bits are not.
        """
        bits = entropy_bits(counts)
        ceiling = math.log2(self.fsq.codebook_size)
        used = int((counts > 0).sum())
        self.log_dict(
            {
                f"{stage}/token_bits": bits,
                f"{stage}/token_bits_frac": bits / ceiling,
                # pooled over the whole epoch — the headline support figure. The
                # per-batch counterpart is {stage}/codebook_frac_batch.
                f"{stage}/codebook_frac": used / self.fsq.codebook_size,
                f"{stage}/codes_used": float(used),
            },
            prog_bar=False, sync_dist=True,
        )
        counts.zero_()

    def on_train_epoch_end(self):
        self._log_epoch_codebook("train", self.train_code_counts)

    def on_validation_epoch_end(self):
        self._log_epoch_codebook("val", self.val_code_counts)

    # ---------------------------------------------------------------- EMA

    def _current_decay(self):
        """Optionally ramp the decay to `ema_decay_end` over training (DINO/iBOT do
        this: a slower-moving teacher late on stabilises the targets)."""
        if not self.ema_decay_end:
            return self.ema_decay
        total = max(int(self.trainer.estimated_stepping_batches), 1)
        progress = min(self.global_step / total, 1.0)
        return self.ema_decay_end - (self.ema_decay_end - self.ema_decay) * (
            math.cos(math.pi * progress) + 1
        ) / 2

    @torch.no_grad()
    def update_ema(self, decay=None):
        """teacher <- decay*teacher + (1-decay)*student, parameters AND buffers.

        Buffers are copied outright: they are running statistics, not gradients, and
        an EMA of an EMA lags twice over.
        """
        decay = self.ema_decay if decay is None else decay
        for ema_p, p in zip(self.teacher.parameters(), self.student.parameters()):
            ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for ema_b, b in zip(self.teacher.buffers(), self.student.buffers()):
            ema_b.copy_(b)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # After the optimizer step, so the teacher trails the weights that were
        # actually used — not the pre-update ones.
        self.update_ema(self._current_decay())

    # ---------------------------------------------------------------- optim

    def configure_optimizers(self):
        params = [p for p in self.student.parameters() if p.requires_grad]
        optimizer = hydra.utils.instantiate(self.optimizer_cfg.target, params=params)

        if not self.optimizer_cfg.get("scheduler", False):
            return {"optimizer": optimizer}

        from models.components.cosine_warmup import CosineWarmupScheduler

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=total_steps * self.optimizer_cfg.get("warmup_ratio", 0.067),
            total_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step",
                             "frequency": 1, "name": "lr_cosine"},
        }
