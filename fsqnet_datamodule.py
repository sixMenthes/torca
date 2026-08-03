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
    FSQNet datamodule aligned to the augmentations DCLDE parquet + hydrophone split
    (Option B) so FSQNet trains on the SAME data and splits as the VIT/LabelDataModule
    path and stays comparable to it.

    Reads the raw annotation parquet (e.g. DCLDE_w_Buzzes.parquet), filters to the
    configured labels, maps each annotation's GCS `NewPath` to a LOCAL full-soundfile
    path (mirror rooted at `data_dir`), and splits by hydrophone (`Dataset`):
    test/val hydrophones are held out and low-SR hydrophones are excluded from train,
    matching LabelDataModule.build_set. `LocalDataset` then trims a `chunk_duration`
    window around [FileBeginSec, FileEndSec] and resamples to `sample_rate` on decode.
    Yields raw-waveform batches `(padded, mask, labels)`; exposes `label_map` and
    inverse-frequency `class_weights` (from the train split).

    ASSUMES full soundfiles are mirrored locally under `data_dir` (e.g. via
    fsq/download_preprocess.py: local = data_dir / NewPath.removeprefix(gs_root)).
    If instead you have per-clip pre-sliced wavs, this mapping and LocalDataset's
    range-trim are wrong -- point clip_path at the clips and load them whole.
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
        self.gs_root = dc.gs_root

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
        clip_path = pl.format(
            "{}/{}",
            pl.lit(self.data_dir),
            pl.col("NewPath").str.strip_prefix(self.gs_root),
        )
        return df.with_columns(clip_path.alias("clip_path"))

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
