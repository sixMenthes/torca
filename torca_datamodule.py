import lightning as L
from omegaconf import DictConfig, OmegaConf
from util.pylogger import get_pylogger
import polars as pl
from torch.utils.data import DataLoader, Dataset
from torchaudio.compliance.kaldi import fbank
import torch.nn.functional as F
from torchcodec.decoders import AudioDecoder
import torch
import torchaudio
import gcsfs
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from torca_transforms import BaseTransform
from pathlib import Path

log = get_pylogger(__name__)
# input: B, C, H, W
#todo: claculate mean std, in setup one hot encode and set labels-audios



class TorcaDataset(Dataset):
    def __init__(self, df:pl.DataFrame, transform_configs:DictConfig, label_map: dict):
        self.df = df
        self.transform = BaseTransform(transform_configs)
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.row(index, named=True)
        path = row["LocalPath"]
        label = row["Labels"]
        wave = AudioDecoder(path, sample_rate=32000).get_all_samples()
        features = self.transform(wave.data)
        return {"audio": features, "label": self.label_map[label]}


class TorcaDataModule(L.LightningDataModule):
    def __init__(
            self,
            dataset_configs: DictConfig, 
            loader_configs: DictConfig, 
            transform_configs: DictConfig, 
            sampling_rate: int
    ):
        super().__init__()

        self.parquet_path = dataset_configs.parquet_path
        self.columns = dataset_configs.columns
        self.df = self.load_df()

        self.test_hydros = dataset_configs.test_hydros
        self.low_sr_hydros = dataset_configs.low_sr_hydros
        self.val_hydros = dataset_configs.val_hydros
        self.class_to_balance = dataset_configs.class_to_balance
        
        self.data_dir = dataset_configs.dataset_dir
        self.num_workers = dataset_configs.num_workers
        self.gcl = gcsfs.core.GCSFileSystem(token='anon')
        self.failed_files = []

##############
        self.clip_duration = dataset_configs.clip_duration
        self.train_split = dataset_configs.train_split
        self.test_split = dataset_configs.test_split

        self.num_classes = dataset_configs.num_classes
        self.test_size = dataset_configs.test_size
        self.sampling_rate = sampling_rate
        self.save_to_disk = dataset_configs.save_to_disk
        self.test_in_val = dataset_configs.test_in_val
        self.saved_images = dataset_configs.saved_images

################

        self.train_loader_configs = loader_configs.train
        self.val_loader_configs = loader_configs.val
        self.test_loader_configs = loader_configs.test

    def prepare_data(self):
        test_set = self.build_test_set()
        train_set = self.build_train_set()
        self.download_set(pl.concat([test_set, train_set]))
        val_set = train_set.filter(pl.col("Dataset").is_in(self.val_hydros))
        train_set = train_set.filter(~pl.col("Dataset").is_in(self.val_hydros))
        return train_set, val_set, test_set


    def setup(self, stage:str):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def download_file(self, row:pl.Series):
        gcs_path = row["GCSPath"]
        starts = row["SampleBeginSec"]
        ends = row["SampleEndSec"]
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
                       .agg("Soundfile", "SampleBeginSec", "SampleEndSec", "LocalPath"))

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.download_file, row) for row in to_download.iter_rows(named=True)]
            for fut in as_completed(futures):
                failed = fut.result()
                if failed:
                    self.failed_files.append(failed)


    def build_test_set(self):
        duration = pl.col('SampleEndSec') - pl.col('SampleBeginSec')
        center_time = pl.col('SampleBeginSec') + (duration / 2.0)
        new_start_time = pl.max_horizontal(pl.lit(0), center_time - self.clip_duration/2.0)
        new_end_time = (new_start_time + self.clip_duration)

        return self.df.filter(pl.col('Dataset').is_in(self.test_hydros)).with_columns(
            duration.alias('true_duration'),
            center_time.alias('center_time'),
            new_start_time.alias('new_start_time'),
            new_end_time.alias('new_end_time')
        )


    def build_train_set(self):
        duration = pl.col('SampleEndSec') - pl.col('SampleBeginSec')
        center_time = pl.col('SampleBeginSec') + (duration / 2.0)
        new_start_time = pl.max_horizontal(pl.lit(0), center_time - self.clip_duration/2.0)
        new_end_time = (new_start_time + self.clip_duration)
    
        pool = (self.df.filter( 
                ~pl.col('Dataset').is_in(self.test_hydros),
                ~pl.col('Dataset').is_in(self.low_sr_hydros))
            .with_columns(
                duration.alias('true_duration'),
                center_time.alias('center_time'),
                new_start_time.alias('new_start_time'),
                new_end_time.alias('new_end_time')
            ))

        rest = pool.filter(~pl.col('Labels').is_in(self.class_to_balance.keys()))

        srkw_time = self.df.filter(pl.col('Labels') == 'SRKW')['SampleDuration'].sum()

        strat_samples = []
        for c, frac in self.class_to_balance.items():
            dur = srkw_time * frac
            strat_samples.append(stratified_sampling(c, dur, pool))
    
        return pl.concat([rest, *strat_samples])

    

    def load_df(self):
      stem = pl.col("Soundfile").str.replace(r"\.[^.]+$", "")
      start_ms = (pl.col("SampleBeginSec") * 1000).round().cast(pl.Int64).cast(pl.String).str.zfill(10)
      end_ms   = (pl.col("SampleEndSec")   * 1000).round().cast(pl.Int64).cast(pl.String).str.zfill(10)

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

      return (
          pl.read_parquet(self.parquet_path)
          .select(*self.columns)
          .rename({
              "FileBeginSec": "SampleBeginSec",
              "FileEndSec":   "SampleEndSec",
              "Duration":     "SampleDuration",
              "NewPath":      "GCSPath",
          })
          .with_columns(local_path.alias("LocalPath"))
      )



def stratified_sampling(label:str, tgt_duration:float, df:pl.DataFrame, curr_duration=0, seed=59):
    seed += 1
    remaining_duration = tgt_duration - curr_duration
    if remaining_duration < 300:
        return df.clear()

    pool = df.filter(pl.col('Labels') == label)
    duration_per_hydro = remaining_duration / pool['Dataset'].n_unique()
    cumul_per_hydro = pl.col('SampleDuration').cum_sum().over('Dataset')

    new_samples = (pool
                   .sample(fraction=1.0, shuffle=True)
                   .filter(cumul_per_hydro <= duration_per_hydro))

    added = new_samples['SampleDuration'].sum()

    if added == 0:
        return df.clear()

    return pl.concat([new_samples, stratified_sampling(label, tgt_duration, df, curr_duration+added, seed=seed)])


if __name__ == "__main__":
    my_conf = OmegaConf.load("./configs/data/transform/melbank_dclde.yaml")
    my_df = pl.read_parquet("./dataset/DCLDE_w_Buzzes.parquet")
    my_df = my_df.with_columns(LocalPath = pl.lit("../ds/try_birdmae.wav"))
    dataset = TorcaDataset(my_df, my_conf)


