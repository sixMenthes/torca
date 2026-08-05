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
