import polars as pl
from torch.utils.data import DataLoader

from util.pylogger import get_pylogger
from torca_datamodule import LabelDataModule, collate_fn_skip
from torca_transforms import RandomShift, RandomGain, AddBackgroundNoise
from selfdistill_dataset import SelfDistillDataset, WaveformViewAug

log = get_pylogger(__name__)


class SelfDistillDataModule(LabelDataModule):
    """Self-supervised adaptation datamodule (EMA self-distillation).

    Reuses LabelDataModule's manifest loading, GCS download and hydrophone
    bookkeeping, but differs from the classification pool in three ways:
      * the pool INCLUDES low-SR hydrophones and is label-agnostic — no class
        loss runs here, and more channel variety helps the invariance objective;
      * only ``test_hydros`` and ``val_hydros`` are held out (test must stay
        unseen; val is kept for monitoring);
      * each item is a (teacher, student) two-view pair of augmented WAVEFORMS,
        with the mel/PCEN front-end deferred to the model.
    """

    def build_selfdistill_set(self):
        # Everything except held-out test/val; low-SR INCLUDED (the classifier
        # pool excludes it, we don't). No Labels filter — SSL is label-agnostic.
        held_out = self.test_hydros + self.val_hydros
        return self.df.filter(~pl.col("Dataset").is_in(held_out))

    def _ssl_background_bank(self):
        # Cross-hydrophone (and cross-SR) background bank: Background-labelled
        # clips from any non-test/val hydrophone, low-SR included. This is the
        # de-confounding lever — the student view sees the channel variety we
        # want the backbone to become invariant to.
        held_out = self.test_hydros + self.val_hydros
        return (
            self.df.filter(
                pl.col("Labels") == "Background", ~pl.col("Dataset").is_in(held_out)
            )
            .get_column("LocalPath")
            .to_list()
        )

    def _build_view_aug(self, view_cfg, sr, max_length, bank):
        """Build one view's augmentation pipeline from config.

        Ops with p == 0 are dropped entirely rather than constructed-and-skipped, so
        sweeping a single op off costs one override and doesn't pay for e.g. the
        background bank's file IO. Order is fixed (shift -> gain -> background):
        background noise must be mixed at the SNR of the final signal, so it goes last.
        """
        ops = []
        if not view_cfg:
            return WaveformViewAug(ops)

        shift = view_cfg.get("shift", None)
        if shift and shift.p > 0:
            ops.append(RandomShift(
                max_shift_samples=int(shift.max_shift_seconds * sr), p=shift.p))

        gain = view_cfg.get("gain", None)
        if gain and gain.p > 0:
            ops.append(RandomGain(
                min_gain_db=gain.min_gain_db, max_gain_db=gain.max_gain_db, p=gain.p))

        bg = view_cfg.get("background", None)
        if bg and bg.p > 0:
            ops.append(AddBackgroundNoise(
                background_paths=bank, target_length=max_length, sample_rate=sr,
                min_snr_db=bg.min_snr_db, max_snr_db=bg.max_snr_db, p=bg.p))

        return WaveformViewAug(ops)

    def setup(self, stage: str):
        if stage in ("fit", None):
            sr = int(self.transform_config.input.sample_rate)
            max_length = int(sr * self.transform_config.clip_duration)
            bank = self._ssl_background_bank()

            # Asymmetric views: teacher clean, student strong. This asymmetry is what
            # turns the masked-prediction objective into a denoising / channel-
            # invariance objective — the cheapest form of the "multiple views"
            # iBOT/DINO use, no second head or multi-crop yet.
            aug_cfg = self.transform_config.get("augmentations", {})
            teacher_aug = self._build_view_aug(aug_cfg.get("teacher", {}), sr, max_length, bank)
            student_aug = self._build_view_aug(aug_cfg.get("student", {}), sr, max_length, bank)

            self.ssl_set = SelfDistillDataset(
                self.build_selfdistill_set(),
                teacher_aug,
                student_aug,
                sample_rate=sr,
                max_length=max_length,
            )

            # Held-out RECORDING CONDITION (CarmanahPt), which is the right thing to
            # monitor for an invariance objective: it answers "is the masked-prediction
            # task improving on a channel the model never adapted on", not just on the
            # training channels. Same augmentation policy as training (the student view
            # keeps cross-hydrophone noise, so this measures denoising of an unseen
            # channel), but deterministic — each clip is augmented and masked
            # identically every epoch, so the curve moves only when the model does.
            self.val_ssl_set = SelfDistillDataset(
                self.df.filter(pl.col("Dataset").is_in(self.val_hydros)),
                teacher_aug,
                student_aug,
                sample_rate=sr,
                max_length=max_length,
                deterministic=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.ssl_set,
            num_workers=self.train_loader_configs.num_workers,
            batch_size=self.train_loader_configs.batch_size,
            shuffle=True,
            persistent_workers=self.train_loader_configs.persistent_workers,
            pin_memory=self.train_loader_configs.pin_memory,
            collate_fn=collate_fn_skip,
        )

    def val_dataloader(self):
        """Two-view val loader. MUST override the inherited one.

        LabelDataModule.val_dataloader returns a loader over `self.val_set`, which this
        datamodule never builds — so the moment a validation_step exists, the inherited
        version raises AttributeError. shuffle=False keeps batch composition fixed
        across epochs, which the deterministic augmentation relies on.
        """
        return DataLoader(
            self.val_ssl_set,
            num_workers=self.val_loader_configs.num_workers,
            batch_size=self.val_loader_configs.batch_size,
            shuffle=False,
            persistent_workers=self.val_loader_configs.persistent_workers,
            pin_memory=self.val_loader_configs.pin_memory,
            collate_fn=collate_fn_skip,
        )

    def probe_dataloader(self, batch_size=32):
        """LABELLED val clips for the online ecotype probe.

        Single hydrophone by design: with recording condition held constant, the probe
        cannot exploit a channel shortcut, so it is a clean read on whether ecotype is
        linearly separable — which is exactly the quantity the adaptation is supposed
        to improve.
        """
        from probe_features import build_loader

        labelled = self.df.filter(
            pl.col("Dataset").is_in(self.val_hydros)
            & pl.col("Labels").is_in(list(self.labels))
        )
        return build_loader(
            labelled, int(self.transform_config.input.sample_rate),
            self.clip_duration, self.label_map, self.call_map,
            batch_size=batch_size, num_workers=self.val_loader_configs.num_workers,
        )
