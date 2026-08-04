import os

import numpy as np
import lightning as L
import polars as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fsqnet_dataset import LocalDataset, collate_fn
from util.pylogger import get_pylogger

log = get_pylogger(__name__)


class LocalDataModule(L.LightningDataModule):
    """
    FSQNet datamodule that consumes the SAME pre-sliced DCLDE clips and the SAME
    hydrophone split as the augmentations/LabelDataModule path, for apples-to-apples
    comparability -- and, crucially, without needing the full 1.5 TiB dataset on
    disk (only the small per-annotation clips), which suits the Alliance/rorqual
    cluster.

    Reads the raw annotation parquet, filters to the configured labels, and rebuilds
    each clip's local path with the EXACT scheme LabelDataModule.load_df used to
    write them:

        {data_dir}/{Provider}/{Dataset}/{stem}/{start_ms}-{end_ms}.wav

    where the window is `slice_duration` seconds centered on the annotation
    (start_ms/end_ms are zero-padded ms of new_start/new_end). `slice_duration`
    MUST equal the clip_duration the clips were cut at (VIT used 5.0). Splits by
    hydrophone (`Dataset`): test/val held out, low-SR excluded from train. The clip
    is center-cropped to the network's `chunk_duration` in LocalDataset.

    If `drop_missing_clips`, rows whose file is absent are dropped at init (also a
    sanity check that the path scheme / data_dir match what's on disk).
    """

    def __init__(
        self,
        dataset_configs: DictConfig,
        loader_configs: DictConfig,
        transform_configs: DictConfig = None,
    ):
        super().__init__()
        dc = dataset_configs
        self.parquet_path = dc.parquet_path
        self.name = dc.name
        self.labels = list(dc.labels)
        self.num_classes = dc.num_classes
        self.chunk_duration = dc.get("chunk_duration", 3.0)
        self.sample_rate = dc.get("sample_rate", 16000)
        self.data_dir = str(dc.data_dir).rstrip("/")
        self.slice_duration = dc.slice_duration
        self.drop_missing_clips = dc.get("drop_missing_clips", True)

        self.test_hydros = list(dc.test_hydros)
        self.val_hydros = list(dc.val_hydros)
        self.low_sr_hydros = list(dc.get("low_sr_hydros", []))

        self.label_map = dict(zip(self.labels, range(self.num_classes)))
        self.df = self._load_df()
        self.class_weights = self._class_weights(self._split_df("train"))

        self.train_loader_configs = loader_configs.train
        self.val_loader_configs = loader_configs.val
        self.test_loader_configs = loader_configs.test

    def _load_df(self) -> pl.DataFrame:
        df = pl.read_parquet(self.parquet_path)
        if "NewFileOk" in df.columns:
            df = df.filter(pl.col("NewFileOk"))
        df = df.filter(pl.col("Labels").is_in(self.labels))

        # Rebuild LabelDataModule.load_df's LocalPath (must match on-disk clips).
        duration = pl.col("FileEndSec") - pl.col("FileBeginSec")
        center_time = pl.col("FileBeginSec") + duration / 2.0
        new_start = pl.max_horizontal(pl.lit(0.0), center_time - self.slice_duration / 2.0)
        new_end = new_start + self.slice_duration
        stem = pl.col("Soundfile").str.replace(r"\.[^.]+$", "")
        start_ms = (new_start * 1000).round().cast(pl.Int64).cast(pl.String).str.zfill(10)
        end_ms = (new_end * 1000).round().cast(pl.Int64).cast(pl.String).str.zfill(10)
        clip_path = pl.format(
            "{}/{}/{}/{}/{}-{}.wav",
            pl.lit(self.data_dir),
            pl.col("Provider"),
            pl.col("Dataset"),
            stem,
            start_ms,
            end_ms,
        )
        df = df.with_columns(clip_path.alias("clip_path"))

        if self.drop_missing_clips:
            n_before = df.height
            exists = [os.path.exists(p) for p in df.get_column("clip_path").to_list()]
            df = df.filter(pl.Series(exists))
            log.info(f"clips on disk: {df.height}/{n_before} (dropped {n_before - df.height} missing)")
            if df.height == 0:
                raise RuntimeError(
                    "No clips found on disk. Check data_dir, slice_duration, and that "
                    "the path scheme matches how the clips were written."
                )
        return df

    def _split_df(self, split: str) -> pl.DataFrame:
        if split == "test":
            return self.df.filter(pl.col("Dataset").is_in(self.test_hydros))
        if split == "val":
            return self.df.filter(pl.col("Dataset").is_in(self.val_hydros))
        held = self.test_hydros + self.val_hydros + self.low_sr_hydros
        return self.df.filter(~pl.col("Dataset").is_in(held))

    def _class_weights(self, train_df: pl.DataFrame) -> torch.Tensor:
        """Inverse-frequency weights, normalized to sum to num_classes."""
        counts = [
            train_df.filter(pl.col("Labels") == label).height for label in self.labels
        ]
        weights = 1.0 / np.maximum(counts, 1)
        weights = weights * len(self.labels) / weights.sum()
        return torch.from_numpy(weights).float()

    def _make_set(self, split: str) -> LocalDataset:
        return LocalDataset(
            self._split_df(split),
            label_map=self.label_map,
            clip_duration=self.chunk_duration,
            sample_rate=self.sample_rate,
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_set = self._make_set("train")
            self.val_set = self._make_set("val")
        if stage == "test":
            self.test_set = self._make_set("test")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            num_workers=self.train_loader_configs.num_workers,
            batch_size=self.train_loader_configs.batch_size,
            shuffle=self.train_loader_configs.shuffle,
            drop_last=self.train_loader_configs.get("drop_last", True),
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            num_workers=self.val_loader_configs.num_workers,
            batch_size=self.val_loader_configs.batch_size,
            shuffle=self.val_loader_configs.shuffle,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            num_workers=self.test_loader_configs.num_workers,
            batch_size=self.test_loader_configs.batch_size,
            shuffle=self.test_loader_configs.shuffle,
            collate_fn=collate_fn,
        )
