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

        feats, labels = [], []
        try:
            for batch in dm.probe_dataloader():
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
        pl_module.log_dict(
            {
                "val/probe_ecotype": res["balanced_acc"],
                "val/probe_ecotype_chance": res["majority_baseline"],
            },
            prog_bar=True, sync_dist=True,
        )


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
    def _tokenise(self, pl_module, loader):
        """(pooled counts, {site: counts}) over a loader, padding excluded."""
        from models.components.selfdistill_loss import code_counts, valid_token_mask

        K = pl_module.fsq.codebook_size
        pooled = np.zeros(K, dtype=np.int64)
        by_site, per_clip = {}, []

        for batch in loader:
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

        try:
            for tag, background in (("bg", True), ("call", False)):
                loader = dm.code_usage_dataloader(
                    background, max_per_hydro=self.max_per_hydro,
                    batch_size=self.batch_size,
                )
                pooled, by_site, per_clip = self._tokenise(pl_module, loader)
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
