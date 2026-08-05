"""Online linear probe on the val hydrophone, run during self-distillation training.

The SSL loss says whether the pretext task is being solved; it does not say whether
ecotype became more decodable, which is the thing the study is actually about. This
callback answers that during training instead of only at the end.

It probes CarmanahPt alone, and the single-hydrophone restriction is a feature: with
recording condition held constant the probe cannot exploit a channel shortcut, so the
number is a clean read on ecotype separability rather than a mix of ecotype and
recording condition.

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
