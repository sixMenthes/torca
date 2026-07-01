import lightning as L
from omegaconf import DictConfig, OmegaConf
from util.pylogger import get_pylogger
import polars as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchaudio.compliance.kaldi import fbank
import torch
import os
import gcsfs
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from torca_transforms import BaseTransform, TrainTransform
from tqdm import tqdm
from pathlib import Path, PurePath
from torca_dataset import LabelDataset, CallDataset

log = get_pylogger(__name__)
# input: B, C, H, W

def collate_fn_skip(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.default_collate(batch) if batch else None


class LabelDataModule(L.LightningDataModule):
    def __init__(
            self,
            dataset_configs: DictConfig, 
            loader_configs: DictConfig, 
            transform_configs: DictConfig
    ):
        super().__init__()

        self.parquet_path = dataset_configs.parquet_path
        self.name = dataset_configs.name
        self.columns = dataset_configs.columns

        self.test_hydros = dataset_configs.test_hydros
        self.low_sr_hydros = dataset_configs.low_sr_hydros
        self.val_hydros = dataset_configs.val_hydros
        self.class_to_balance = dataset_configs.class_to_balance
        # desired relative class proportions for the weighted train sampler;
        # None -> balanced (inverse-frequency) across all present classes.
        self.class_sampling_weights = dataset_configs.get("class_sampling_weights", None)

        self.data_dir = dataset_configs.dataset_dir
        self.num_workers = dataset_configs.num_workers
        self.clip_duration = dataset_configs.clip_duration
        self.gcl = gcsfs.core.GCSFileSystem(token='anon')
        self.failed_files = []


################

        self.labels = dataset_configs.labels
        self.calls = dataset_configs.calls
        self.num_calls = dataset_configs.num_calls
        self.num_classes = dataset_configs.num_classes
        self.label_map = dict(zip(self.labels, range(self.num_classes)))
        self.call_map = dict(zip(self.calls, range(self.num_calls)))

################

        self.df = self.load_df()
        self.transform_config = transform_configs

        self.train_loader_configs = loader_configs.train
        self.val_loader_configs = loader_configs.val
        self.test_loader_configs = loader_configs.test

    def prepare_data(self):
        self.download_set(self.df)
        self.df = self.df.filter(~pl.col("Soundfile").is_in(set(self.failed_files)))
        self.df.write_parquet(os.path.join(self.data_dir, self.name))


    def setup(self, stage:str):

        if stage == "fit":
            train_transform = TrainTransform(self.transform_config, background_paths=self._background_bank())
            self.train_set = LabelDataset(self.build_set("train"), train_transform, self.label_map)
            val_transform = BaseTransform(self.transform_config)
            self.val_set = LabelDataset(self.build_set("val"), val_transform, self.label_map)
            self.call_set = CallDataset(self.build_set("calls"), val_transform, self.call_map)

        if stage == "test":
            test_transform = BaseTransform(self.transform_config)
            self.test_set = LabelDataset(self.build_set("test"), test_transform, self.label_map)
            

    def train_dataloader(self):
        # WeightedRandomSampler oversamples the rare classes (HW/TKW); it is
        # mutually exclusive with shuffle, so shuffle is forced off.
        sampler = self._make_train_sampler(self.train_set.df)
        return DataLoader(
            self.train_set,
            num_workers = self.train_loader_configs.num_workers,
            batch_size=self.train_loader_configs.batch_size,
            sampler=sampler,
            shuffle=False,
            persistent_workers=self.train_loader_configs.persistent_workers,
            pin_memory=self.train_loader_configs.pin_memory
        )

    def _background_bank(self):
        # Background-labelled clips reused as an additive-noise bank. Exclude
        # test/val hydrophones so held-out recording conditions don't leak in.
        non_train = self.test_hydros + self.val_hydros
        return (self.df
                .filter(pl.col("Labels") == "Background",
                        ~pl.col("Dataset").is_in(non_train))
                .get_column("LocalPath")
                .to_list())

    def _make_train_sampler(self, df):
        labels = df.get_column("Labels").to_list()
        counts = df.group_by("Labels").len()
        count_map = dict(zip(counts.get_column("Labels").to_list(),
                             counts.get_column("len").to_list()))
        if self.class_sampling_weights:
            target = {l: float(self.class_sampling_weights.get(l, 1.0)) for l in count_map}
        else:
            target = {l: 1.0 for l in count_map}  # balanced across present classes
        per_class_w = {l: target[l] / max(count_map[l], 1) for l in count_map}
        weights = torch.tensor([per_class_w[l] for l in labels], dtype=torch.double)
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def val_dataloader(self):
        val_dataloader =  DataLoader(
            self.val_set,
            num_workers=self.val_loader_configs.num_workers,
            batch_size=self.val_loader_configs.batch_size,
            shuffle=self.val_loader_configs.shuffle,
            persistent_workers=self.val_loader_configs.persistent_workers,
            pin_memory=self.val_loader_configs.pin_memory
        )
        call_dataloader = DataLoader(
            self.call_set,
            num_workers=self.val_loader_configs.num_workers,
            batch_size=self.val_loader_configs.batch_size,
            shuffle=self.val_loader_configs.shuffle,
            persistent_workers=self.val_loader_configs.persistent_workers,
            pin_memory=self.val_loader_configs.pin_memory
        )
        return [val_dataloader, call_dataloader]


    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            num_workers=self.test_loader_configs.num_workers,
            batch_size=self.test_loader_configs.batch_size,
            shuffle=self.test_loader_configs.shuffle
        )

    def download_file(self, row:pl.Series):
        gcs_path = row["GCSPath"]
        starts = row["new_start_time"]
        ends = row["new_end_time"]
        paths = row["LocalPath"]

        to_do = [(s, e, p) for s, e, p in zip(starts, ends, paths) if not Path(p).exists()]

        if not to_do:
            return

        try:
            with self.gcl.open(gcs_path, "rb", block_size=2**20) as f:
                with sf.SoundFile(f) as snd:
                    sr = snd.samplerate
                    for st, et, sp in to_do:
                        start_frame = int(round(st * sr))
                        end_frame = int(round(et * sr))
                        duration = end_frame - start_frame
                        snd.seek(start_frame)
                        audiodata = snd.read(duration, dtype='float32')
                        Path(sp).parent.mkdir(parents=True, exist_ok=True)
                        sf.write(sp, audiodata, sr)

        except Exception as e:
            log.warning(f"Failed {gcs_path}: {e}")
            return gcs_path


    def download_set(self, df:pl.DataFrame):
        to_download = (df
                       .group_by('GCSPath', maintain_order=True)
                       .agg("Soundfile", "new_start_time", "new_end_time", "LocalPath"))

        l = to_download.height

        with tqdm(total=l, desc="Downloading files") as pbar:
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(self.download_file, row) for row in to_download.iter_rows(named=True)]
                for fut in as_completed(futures):
                    failed = fut.result()
                    if failed:
                        self.failed_files.append(PurePath(failed).name)
                    pbar.update(1)


    def build_set(self, split:str):

        #assert (split in {"test", "val", "train"}), "split must be one of train, val or test"
        #also i should assert that classes to balance is a subset of labels
        df = self.df.filter(pl.col('Labels').is_in(self.labels))
        
        if split == "test":
            return (df
                    .filter(pl.col('Dataset').is_in(self.test_hydros))
                    .with_columns(pl.lit("test").alias("split")))
        elif split == "val":
            return (df
                    .filter(pl.col('Dataset').is_in(self.val_hydros))
                    .with_columns(pl.lit("val").alias("split")))
        elif split == "calls":
            non_train_hydros = self.low_sr_hydros + self.val_hydros + self.test_hydros
            return(df
                   .filter(~pl.col('Dataset').is_in(non_train_hydros))
                   .filter(pl.col('Labels') == 'SRKW')
                   .drop_nulls(pl.col('CalltypeCategory'))
                   .filter(pl.col('CalltypeCategory').is_in(self.calls)))
        else: # train
            # Class balancing is handled at load time by the WeightedRandomSampler
            # in train_dataloader, not by materialising oversampled rows here.
            return (df
                    .filter(
                        ~pl.col("Dataset").is_in(self.test_hydros),
                        ~pl.col("Dataset").is_in(self.low_sr_hydros),
                        ~pl.col("Dataset").is_in(self.val_hydros)
                    )
                    .with_columns(pl.lit("train").alias("split")))

    def load_df(self):

        df = pl.read_parquet(self.parquet_path)

        duration = pl.col('FileEndSec') - pl.col('FileBeginSec')
        center_time = pl.col('FileBeginSec') + (duration / 2.0)
        new_start_time = pl.max_horizontal(pl.lit(0), center_time - self.clip_duration/2.0)
        new_end_time = (new_start_time + self.clip_duration)

        df = (df
            .filter(pl.col("NewFileOk") & (pl.col("Labels") != "KW_und"))
            .with_columns(
                duration.alias('true_duration'),
                center_time.alias('center_time'),
                new_start_time.alias('new_start_time'),
                new_end_time.alias('new_end_time')
                )
            .drop(
                pl.col("FileBeginSec"),
                pl.col("FileEndSec"),
                pl.col("Duration"),
                )
            )

        stem = pl.col("Soundfile").str.replace(r"\.[^.]+$", "")
        start_ms = (pl.col("new_start_time") * 1000).round().cast(pl.Int64).cast(pl.String).str.zfill(10)
        end_ms   = (pl.col("new_end_time")   * 1000).round().cast(pl.Int64).cast(pl.String).str.zfill(10)

        local_path = pl.format(
            "{}/{}/{}/{}/{}-{}{}",
            pl.lit(self.data_dir),
            pl.col("Provider"),
            pl.col("Dataset"),
            stem,
            start_ms,
            end_ms,
            pl.lit(".wav"),
            )
        
        return (df
                .rename({"NewPath": "GCSPath"})
                .with_columns(local_path.alias("LocalPath")))


if __name__ == "__main__":
    data_conf = OmegaConf.load("configs/data/dataset/DCLDE_test.yaml")
    loader_conf = OmegaConf.load("configs/data/loaders/default.yaml")
    transform_conf = OmegaConf.load("/Users/leo/projects/orcas/torca/configs/data/transform/melbank_dclde_test.yaml")
    module = LabelDataModule(data_conf, loader_conf, transform_conf)


