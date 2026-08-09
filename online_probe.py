"""Online linear probe on the val hydrophone, run during self-distillation training.

The SSL loss says whether the pretext task is being solved; it does not say whether
ecotype became more decodable, which is the thing the study is actually about. This
callback answers that during training instead of only at the end.

It probes ONE hydrophone alone, named by `online_probe_hydro` in the dataset config,
and the single-hydrophone restriction is a feature: with recording condition held
constant the probe cannot exploit a channel shortcut, so the number is a clean read on
ecotype separability rather than a mix of ecotype and recording condition.

That site is StrGeoS1, which is deliberately NOT simply "the val hydrophone". val_hydros
now holds two sites, because val/loss is a better selection signal when measured across
more than one unseen channel, and with two sites in play the probe could score by
identifying the channel instead of the call. StrGeoS1 rather than the other val site
because it carries 204 HW, 60 SRKW and 168 TKW, so it poses a genuine three-way ecotype
discrimination (chance 0.333), whereas Cpe_Elz is ~89% TKW and bush_point is SRKW
against Background with no HW or TKW at all — a probe there would be measuring whale
detection, not ecotype.

It is a MONITORING signal, not the reported metric. The reported number comes from
probe_selfdistill.py, which fits on train hydrophones and evaluates once on the sealed
test split. This one trains and tests inside a single hydrophone by cross-validation,
so it is in-domain and not comparable to that.

Naming scheme for the ecotype balanced accuracies
-------------------------------------------------
Three probes in this project report an ecotype balanced accuracy under three different
protocols, and their values are not comparable to one another. The metric names carry
the protocol so that a plot cannot silently mix them:

  val/probe_eco_1site     OnlineProbe, here. StratifiedKFold WITHIN the single site
                          named by `online_probe_hydro`. In-domain and channel-free,
                          but one recording condition and a few hundred clips.
  val/probe_eco_cvtrain   TrainCVProbe, below. GroupKFold BY HYDROPHONE across the
                          train sites, so each fold scores sites its own probe never
                          saw. This is the closest in-training mirror of the reported
                          protocol, and it is still optimistic because the BACKBONE
                          adapted on all of these sites even though the probe did not.
  test/probe_eco_traintest  probe_selfdistill.py, offline. Fit on train hydrophones,
                          scored ONCE on the sealed test split. The reported number,
                          and the only one that may not be used to choose anything.

The nuisance metric follows the same convention. `val/probe_nuis_cvtrain` is hydrophone
decodability from Background clips over the train sites, which is the SAME population
the reported nuisance figure uses (probe.probe_nuisance_background masks on
tags == "train"), so unlike the task metric it differs from the reported one only by
the per-site cap and can be read as a genuine preview.
"""

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

import probe as probe_lib
from util.pylogger import get_pylogger

log = get_pylogger(__name__)


class OnlineProbe(Callback):
    """Linear-probe the student encoder on labelled val clips every N epochs.

    every_n_epochs: probing costs a forward pass over the val clips (~1.8k for
      CarmanahPt), so it is cheap but not free; every epoch is usually unnecessary.
    layer: optional intermediate block to probe instead of the final output.
    """

    def __init__(self, every_n_epochs=1, layer=None, max_clips=None, C=1.0):
        super().__init__()
        self.every_n_epochs = int(every_n_epochs)
        self.layer = layer
        self.max_clips = max_clips
        self.C = C

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        dm = trainer.datamodule
        if dm is None or not hasattr(dm, "probe_dataloader"):
            return

        encoder = pl_module.student["encoder"]
        was_training = encoder.training
        encoder.eval()

        # fast_dev_run limits the train and val LOOPS to one batch, but this callback
        # builds its own dataloader and would otherwise iterate all of it — 435 clips,
        # which is nine minutes of ViT-B forward passes on a login node's CPUs. The
        # preflight's whole purpose is to exercise this code path in under a minute, so
        # cap it there and let the real run see everything.
        max_batches = 2 if trainer.fast_dev_run else None

        feats, labels = [], []
        try:
            for i, batch in enumerate(dm.probe_dataloader()):
                if max_batches is not None and i >= max_batches:
                    break
                if batch is None:
                    continue
                wave = batch["wave"].to(pl_module.device)
                feats.append(encoder.pooled(wave, layer=self.layer).float().cpu().numpy())
                labels.append(batch["label"].numpy())
                if self.max_clips and sum(len(f) for f in feats) >= self.max_clips:
                    break
        except Exception as e:                       # never let monitoring kill a run
            log.warning(f"online probe failed: {e}")
            return
        finally:
            if was_training:
                encoder.train()

        if not feats:
            return
        X = np.concatenate(feats, 0)
        y = np.concatenate(labels, 0)
        if len(np.unique(y)) < 2:
            return

        # stratified, not grouped: there is only one hydrophone here, so there is no
        # group to hold out — and equally no channel shortcut to guard against.
        res = probe_lib.probe_cv(X, y, cv="stratified", C=self.C)
        # Named _1site, not probe_ecotype, because three probes with three different
        # protocols now report an ecotype balanced accuracy and the numbers are NOT
        # comparable to each other. See the module docstring for the naming scheme.
        # Runs before 2026-08-09 logged this same quantity as val/probe_ecotype.
        pl_module.log_dict(
            {
                "val/probe_eco_1site": res["balanced_acc"],
                "val/probe_eco_1site_chance": res["majority_baseline"],
            },
            prog_bar=True, sync_dist=True,
        )


class TrainCVProbe(Callback):
    """Both co-primary metrics, cross-validated inside TRAIN, every N validation epochs.

    The reason this exists is methodological rather than convenient. The reported task
    number is fit on train and scored once on the sealed test hydrophones, so it cannot
    be watched while tuning without spending the test split on hyperparameter search:
    look at it, change something, look again, and it has stopped being evidence. Both
    numbers here are computed entirely on train hydrophones, so there is nothing to burn
    and they can be read every validation epoch for as many runs as it takes.

    One forward pass, two probes, because the expensive part is the encoder and both
    probes want features on the same clips:

      val/probe_eco_cvtrain    ecotype balanced accuracy, GroupKFold BY HYDROPHONE. The
                               grouping is what makes it informative: with sites split
                               across folds the probe is scored on recording conditions
                               its own fit never saw, which is the structure of the
                               reported protocol. Ungrouped CV here would let the probe
                               identify the channel instead of the call, and hydrophone
                               correlates with ecotype in this dataset.
      val/probe_nuis_cvtrain   hydrophone decodability from BACKGROUND clips,
                               StratifiedKFold. Down is good. Restricting to Background
                               is what makes it a measurement of channel rather than of
                               content. This is the co-primary metric that the pretext
                               diagnostics cannot see at all: token entropy, code usage
                               and cross-entropy all describe the tokenizer and none of
                               them says whether adaptation is moving the confound.

    Read the pair jointly, as the study does: task up AND nuisance down together. Either
    one alone is satisfiable by something uninteresting.

    Interpretation caveat for the task number. The backbone adapted on every hydrophone
    in this pool, so even with the probe's folds grouped by site the FEATURES are
    in-domain, and this will read higher than the sealed-test figure. It is a trend
    instrument. Comparing it across epochs of one run is sound; comparing its level
    against the reported number is not.
    """

    def __init__(self, every_n_epochs=1, max_per_hydro=300, batch_size=32,
                 min_bg_per_hydro=50, C=1.0, seed=59):
        super().__init__()
        self.every_n_epochs = int(every_n_epochs)
        self.max_per_hydro = max_per_hydro
        self.batch_size = batch_size
        self.min_bg_per_hydro = min_bg_per_hydro
        self.C = C
        self.seed = seed

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        dm = trainer.datamodule
        if dm is None or not hasattr(dm, "train_probe_dataloader"):
            return

        encoder = pl_module.student["encoder"]
        was_training = encoder.training
        encoder.eval()

        # See OnlineProbe: fast_dev_run bounds the train and val LOOPS but not a
        # dataloader a callback builds for itself, and this one is the largest of the
        # three at roughly max_per_hydro x 10 sites.
        max_batches = 2 if trainer.fast_dev_run else None

        feats, labels, sites = [], [], []
        try:
            loader = dm.train_probe_dataloader(
                max_per_hydro=self.max_per_hydro, batch_size=self.batch_size
            )
            for i, batch in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                if batch is None:
                    continue
                wave = batch["wave"].to(pl_module.device)
                feats.append(encoder.pooled(wave).float().cpu().numpy())
                labels.append(batch["label"].numpy())
                sites.extend(batch["dataset"])
        except Exception as e:                       # never let monitoring kill a run
            log.warning(f"train-CV probe feature extraction failed: {e}")
            return
        finally:
            if was_training:
                encoder.train()

        if not feats:
            return
        X = np.concatenate(feats, 0)
        y = np.concatenate(labels, 0)
        h = np.asarray(sites)
        metrics = {}

        # Task: grouped by hydrophone, so no fold shares a recording condition.
        try:
            if len(np.unique(y)) >= 2 and len(np.unique(h)) >= 2:
                res = probe_lib.probe_cv(X, y, groups=h, cv="group", C=self.C,
                                         seed=self.seed)
                metrics["val/probe_eco_cvtrain"] = res["balanced_acc"]
                metrics["val/probe_eco_cvtrain_std"] = res["std"]
                metrics["val/probe_eco_cvtrain_chance"] = res["majority_baseline"]
        except Exception as e:
            log.warning(f"train-CV ecotype probe failed: {e}")

        # Nuisance: the same X, restricted to Background rows. probe_nuisance_background
        # owns the min_per_hydro filter and the Background masking, so it is called with
        # an all-"train" tag array rather than reimplemented here — the reported figure
        # then comes from exactly the same function.
        try:
            bg_index = getattr(dm, "label_map", {}).get("Background", None)
            if bg_index is not None:
                res = probe_lib.probe_nuisance_background(
                    X, h, np.full(len(y), "train"), y == bg_index,
                    min_per_hydro=self.min_bg_per_hydro, C=self.C, seed=self.seed,
                )
                metrics["val/probe_nuis_cvtrain"] = res["balanced_acc"]
                metrics["val/probe_nuis_cvtrain_std"] = res["std"]
                metrics["val/probe_nuis_cvtrain_chance"] = res["majority_baseline"]
                metrics["val/probe_nuis_n_hydros"] = float(res["n_hydrophones"])
        except Exception as e:
            log.warning(f"train-CV nuisance probe failed: {e}")

        if metrics:
            pl_module.log_dict(metrics, prog_bar=False, sync_dist=True)


def _entropy_bits(counts):
    """H = -sum p log2 p over a code histogram. Same estimator as entropy_bits.

    Plug-in, so biased DOWNWARD when the sample is small relative to K (roughly
    (K-1)/(2 N ln2) bits). That bias is why the mutual information below is reported
    against a shuffled control rather than against zero.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _site_code_mi(by_site):
    """Mutual information in bits between hydrophone and code, I(S;C) = H(C) - H(C|S).

    The tokenizer-level version of the nuisance question. If the codes carry nothing
    about which site a Background clip came from, this is zero; if the tokenizer is
    spending codes on recording condition, it is not. Unlike a linear probe it needs no
    fitting, no hyperparameter and no cross-validation — it is a property of the two
    histograms.

    by_site: dict of hydrophone -> (K,) count array.
    """
    sites = [s for s, c in by_site.items() if np.asarray(c).sum() > 0]
    if len(sites) < 2:
        return None
    totals = np.array([np.asarray(by_site[s]).sum() for s in sites], dtype=np.float64)
    pooled = np.sum([np.asarray(by_site[s]) for s in sites], axis=0)
    h_c = _entropy_bits(pooled)
    # H(C|S) = sum_s p(s) H(C | S=s), weighted by each site's share of the tokens.
    h_c_given_s = sum(
        (totals[i] / totals.sum()) * _entropy_bits(by_site[s])
        for i, s in enumerate(sites)
    )
    return h_c - h_c_given_s


class CodeUsageProbe(Callback):
    """Where does the tokenizer spend its codebook — on noise, or on calls?

    Tokenises two subsets of the TRAIN hydrophones with the teacher and compares their
    code histograms:

      * Background clips     -> val/code_bg_token_bits, _frac, _codes_used
      * vocalisation clips   -> val/code_call_token_bits, _frac, _codes_used

    If the two are similar, the codebook is spending as much capacity representing noise
    as representing vocalisations, which is the direct form of the argument that a
    1000-code codebook has room to spare for recording condition. If Background is much
    lower, capacity is concentrated where it should be.

    It then splits the BACKGROUND histogram by hydrophone and reports the mutual
    information between site and code (`val/code_bg_site_mi`). That is the
    tokenizer-level nuisance measurement: zero means the codes say nothing about which
    site a noise clip came from.

    Because a plug-in MI estimate from histograms is biased upward when the sample is
    small relative to K — and K is 1000 — the same quantity is recomputed with the site
    labels SHUFFLED ACROSS CLIPS (`val/code_bg_site_mi_shuffled`). That is the level
    finite sampling alone produces, and `val/code_bg_site_mi_excess` is the difference,
    which is the number to read. Shuffling at clip level rather than token level is
    deliberate: tokens within a clip are not independent, so a token-level shuffle would
    understate the null.

    NOT a replacement for the reported nuisance metric. That one asks whether a linear
    probe can recover the site from the ENCODER's features; this asks what the
    TOKENIZER's codes reveal. A tokenizer that ignores site does not guarantee an
    encoder that does.
    """

    def __init__(self, every_n_epochs=1, max_per_hydro=200, batch_size=32, seed=59):
        super().__init__()
        self.every_n_epochs = int(every_n_epochs)
        self.max_per_hydro = max_per_hydro
        self.batch_size = batch_size
        self.seed = seed

    @torch.no_grad()
    def _tokenise(self, pl_module, loader, max_batches=None):
        """(pooled counts, {site: counts}) over a loader, padding excluded."""
        from models.components.selfdistill_loss import code_counts, valid_token_mask

        K = pl_module.fsq.codebook_size
        pooled = np.zeros(K, dtype=np.int64)
        by_site, per_clip = {}, []

        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            if batch is None:
                continue
            wave = batch["wave"].to(pl_module.device)
            idx = pl_module.teacher_forward(wave)          # (B, N)

            # Padding tokenises to a constant silence code, so it must be excluded or it
            # dominates the histogram — and it would dominate the two subsets UNEQUALLY
            # if their clip durations differ. valid_token_mask is called directly rather
            # than through _valid_mask, which reads batch["student"] and so expects the
            # SSL batch format rather than the probe loader's.
            if pl_module.mask_padding and "n_valid" in batch:
                valid = valid_token_mask(
                    batch["n_valid"].to(pl_module.device), idx.shape[1],
                    pl_module.student["encoder"].freq_patches,
                    pl_module.student["encoder"].sample_rate,
                )
            else:
                valid = torch.ones_like(idx, dtype=torch.bool)

            pooled += code_counts(idx, valid, K).cpu().numpy()
            # Per-clip counts, kept so the site labels can be shuffled at CLIP level.
            for b, site in enumerate(batch["dataset"]):
                c = code_counts(idx[b:b + 1], valid[b:b + 1], K).cpu().numpy()
                per_clip.append((site, c))
                by_site[site] = by_site.get(site, np.zeros(K, dtype=np.int64)) + c
        return pooled, by_site, per_clip

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        dm = trainer.datamodule
        if dm is None or not hasattr(dm, "code_usage_dataloader"):
            return

        K = pl_module.fsq.codebook_size
        ceiling = np.log2(K)
        metrics, bg_per_clip = {}, None

        # See OnlineProbe: fast_dev_run does not reach a callback's own dataloader, and
        # this one iterates 4,243 clips across its two subsets. On the preflight's CPU
        # path that is roughly ninety minutes on a shared login node. Two batches is
        # enough to prove the code path — the NUMBERS from a fast_dev_run are meaningless
        # anyway, since the model has taken one step.
        max_batches = 2 if trainer.fast_dev_run else None

        try:
            for tag, background in (("bg", True), ("call", False)):
                loader = dm.code_usage_dataloader(
                    background, max_per_hydro=self.max_per_hydro,
                    batch_size=self.batch_size,
                )
                pooled, by_site, per_clip = self._tokenise(
                    pl_module, loader, max_batches=max_batches
                )
                if pooled.sum() == 0:
                    continue
                bits = _entropy_bits(pooled)
                metrics[f"val/code_{tag}_token_bits"] = bits
                metrics[f"val/code_{tag}_token_bits_frac"] = bits / ceiling
                metrics[f"val/code_{tag}_codes_used"] = float((pooled > 0).sum())
                if background:
                    bg_per_clip = per_clip
                    mi = _site_code_mi(by_site)
                    if mi is not None:
                        metrics["val/code_bg_site_mi"] = mi
                        metrics["val/code_bg_n_sites"] = float(len(by_site))

            # The null: same estimator, same sample size, site labels destroyed. Any MI
            # below this is finite-sample bias rather than structure.
            if bg_per_clip and "val/code_bg_site_mi" in metrics:
                rng = np.random.default_rng(self.seed + trainer.current_epoch)
                sites = [s for s, _ in bg_per_clip]
                perm = rng.permutation(len(sites))
                shuffled = {}
                for i, (_, c) in enumerate(bg_per_clip):
                    s = sites[perm[i]]
                    shuffled[s] = shuffled.get(s, np.zeros(K, dtype=np.int64)) + c
                null = _site_code_mi(shuffled)
                if null is not None:
                    metrics["val/code_bg_site_mi_shuffled"] = null
                    metrics["val/code_bg_site_mi_excess"] = (
                        metrics["val/code_bg_site_mi"] - null
                    )
        except Exception as e:                       # never let monitoring kill a run
            log.warning(f"code-usage probe failed: {e}")
            return

        if metrics:
            pl_module.log_dict(metrics, prog_bar=False, sync_dist=True)
