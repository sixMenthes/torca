import lightning as L
from omegaconf import DictConfig, OmegaConf
from util.pylogger import get_pylogger
import polars as pl
from torch.utils.data import DataLoader
from torchaudio.compliance.kaldi import fbank
import torch
import os
import gcsfs
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from torca_transforms import BaseTransform, TrainTransform
from tqdm import tqdm
from pathlib import Path, PurePath
from torca_dataset import TorcaDataset

log = get_pylogger(__name__)
# input: B, C, H, W

def collate_fn_skip(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.default_collate(batch) if batch else None


class TorcaDataModule(L.LightningDataModule):
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
        
        self.data_dir = dataset_configs.dataset_dir
        self.num_workers = dataset_configs.num_workers
        self.clip_duration = dataset_configs.clip_duration
        self.gcl = gcsfs.core.GCSFileSystem(token='anon')
        self.labels = dataset_configs.labels
        self.num_classes = dataset_configs.num_classes
        self.failed_files = []

        self.df = self.load_df()

################

        self.transform_config = transform_configs

        self.train_loader_configs = loader_configs.train
        self.val_loader_configs = loader_configs.val
        self.test_loader_configs = loader_configs.test

    def prepare_data(self):
        self.download_set(self.df)
        self.df = self.df.filter(~pl.col("Soundfile").is_in(set(self.failed_files)))
        self.df.write_parquet(os.path.join(self.data_dir, self.name))


    def setup(self, stage:str):

        self.label_map = dict(zip(self.labels, range(self.num_classes)))

        if stage == "fit":
            train_transform = TrainTransform(self.transform_config)
            self.train_set = TorcaDataset(self.build_set("train"), train_transform, self.label_map)
            val_transform = BaseTransform(self.transform_config)
            self.val_set = TorcaDataset(self.build_set("val"), val_transform, self.label_map)

        if stage == "test":
            test_transform = BaseTransform(self.transform_config)
            self.test_set = TorcaDataset(self.build_set("test"), test_transform, self.label_map)
            

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            num_workers = self.train_loader_configs.num_workers,
            batch_size=self.train_loader_configs.batch_size,
            shuffle=self.train_loader_configs.shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            num_workers=self.val_loader_configs.num_workers,
            batch_size=self.val_loader_configs.batch_size,
            shuffle=self.val_loader_configs.shuffle
        )

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

        assert (split in {"test", "val", "train"}), "split must be one of train, val or test"
        df = self.df.filter(pl.col('Labels').is_in(self.labels))
        
        if split == "test":
            return (df
                    .filter(pl.col('Dataset').is_in(self.test_hydros))
                    .with_columns(pl.lit("test").alias("split")))
        elif split == "val":
            return (df
                    .filter(pl.col('Dataset').is_in(self.val_hydros))
                    .with_columns(pl.lit("val").alias("split")))
        else: # I have to solve this
            strat_samples = []
            if self.class_to_balance:
                pool = df.filter(
                    ~pl.col("Dataset").is_in(self.test_hydros),
                    ~pl.col("Dataset").is_in(self.low_sr_hydros),
                    ~pl.col("Dataset").is_in(self.val_hydros)
                )

                rest = pool.filter(~pl.col("Labels").is_in(self.class_to_balance.keys()))
                srkw_time = self.df.filter(pl.col("Labels") == "SRKW")['true_duration'].sum()
                for c, frac in self.class_to_balance.items():
                    dur = srkw_time * frac
                    strat_samples.append(stratified_sampling(c, dur, pool))
                return pl.concat([rest, *strat_samples]).with_columns(pl.lit("train").alias("split"))
            return self.df.filter(
                    ~pl.col("Dataset").is_in(self.test_hydros),
                    ~pl.col("Dataset").is_in(self.low_sr_hydros),
                    ~pl.col("Dataset").is_in(self.val_hydros)
            )

    

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


def stratified_sampling(label:str, tgt_duration:float, df:pl.DataFrame, curr_duration=0, seed=59):
    seed += 1
    remaining_duration = tgt_duration - curr_duration
    if remaining_duration < 300:
        return df.clear()

    pool = df.filter(pl.col('Labels') == label)
    duration_per_hydro = remaining_duration / pool['Dataset'].n_unique()
    cumul_per_hydro = pl.col('true_duration').cum_sum().over('Dataset')

    new_samples = (pool
                   .sample(fraction=1.0, shuffle=True)
                   .filter(cumul_per_hydro <= duration_per_hydro))

    added = new_samples['true_duration'].sum()

    if added == 0:
        return df.clear()

    return pl.concat([new_samples, stratified_sampling(label, tgt_duration, df, curr_duration+added, seed=seed)])


if __name__ == "__main__":
    data_conf = OmegaConf.load("configs/data/dataset/DCLDE_test.yaml")
    loader_conf = OmegaConf.load("configs/data/loaders/default.yaml")
    transform_conf = OmegaConf.load("/Users/leo/projects/orcas/torca/configs/data/transform/melbank_dclde_test.yaml")
    module = TorcaDataModule(data_conf, loader_conf, transform_conf)


