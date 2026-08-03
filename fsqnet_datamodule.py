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
    Lean datamodule for FSQNet: reads a pre-built local manifest (parquet) with a
    `Split` column and yields raw-waveform batches `(padded, mask, labels)` via
    `collate_fn`. Exposes `label_map` and inverse-frequency `class_weights`
    (computed from the train split) for the model's weighted classification loss.
    """

    def __init__(
        self,
        dataset_configs: DictConfig,
        loader_configs: DictConfig,
        transform_configs: DictConfig = None,
    ):
        super().__init__()
        self.parquet_path = dataset_configs.parquet_path
        self.name = dataset_configs.name
        self.labels = list(dataset_configs.labels)
        self.num_classes = dataset_configs.num_classes
        self.clip_duration = dataset_configs.get("clip_duration", 3.0)
        self.sample_rate = dataset_configs.get("sample_rate", 16000)
        self.split_column = dataset_configs.get("split_column", "Split")

        self.label_map = dict(zip(self.labels, range(self.num_classes)))

        self.df = pl.read_parquet(self.parquet_path).filter(
            pl.col("Labels").is_in(self.labels)
        )
        self.class_weights = self._class_weights(self._split("train"))

        self.train_loader_configs = loader_configs.train
        self.val_loader_configs = loader_configs.val
        self.test_loader_configs = loader_configs.test

    def _split(self, split: str) -> pl.DataFrame:
        return self.df.filter(pl.col(self.split_column) == split)

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
            self._split(split),
            label_map=self.label_map,
            clip_duration=self.clip_duration,
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
