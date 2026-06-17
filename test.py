from torch.utils.data import Dataset, DataLoader
import gcsfs
import soundfile as sf
from util.pylogger import get_pylogger
import polars as pl
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torca_transforms import BaseTransform
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchcodec.decoders import AudioDecoder
import lightning as L

log = get_pylogger(__name__)

class TorcaDataset(Dataset):
    def __init__(self, df:pl.DataFrame, transform_config:DictConfig):
        self.df = df
        self.transform = BaseTransform(transform_config) #no freqm nor timem

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.row(index, named=True)
        path = row["LocalPath"]
        wave = AudioDecoder(path, sample_rate=32000).get_all_samples()
        features = self.transform(wave.data)
        return features

class TorcaTest(L.LightningDataModule):
    def __init__(self, dataset_config: DictConfig, transform_config:DictConfig):
        self.parquet_path = dataset_config.parquet_path
        self.columns = dataset_config.columns
        self.df = self.load_df()
        self.test_hydros = dataset_config.test_hydros
        self.low_sr_hydros = dataset_config.low_sr_hydros
        self.val_hydros = dataset_config.val_hydros
        self.class_to_balance = dataset_config.class_to_balance
        self.data_dir = dataset_config.dataset_dir
        self.num_workers = dataset_config.num_workers
        self.gcl = gcsfs.core.GCSFileSystem(token='anon')
        self.transform_config = transform_config
        self.clip_duration = transform_config.clip_duration
        self.failed_files = []

    def run_test(self):
        train_set = self.build_train_set()
        train_set = train_set.filter(~pl.col("Dataset").is_in(self.val_hydros))
        subsample = (
            train_set
                .group_by("Dataset")
                .map_groups(lambda g: g.sample(fraction=0.15, shuffle=True, seed=59))
        )
        self.download_set(subsample)
        sub_test = TorcaDataset(subsample, transform_config=self.transform_config)
        loader = DataLoader(sub_test, batch_size=32, num_workers=self.num_workers)
        n = 0
        sum_x = 0.0
        sum_x2 = 0.0
        for batch in loader:
            x = batch.float()
            n += x.numel()
            sum_x += x.sum().item()
            sum_x2 += (x * x).sum().item()
        mean = sum_x / n
        var = sum_x2 / n - mean ** 2
        std = var ** 0.5
        
        with open(os.path.join(self.data_dir, "info.txt"), "w") as f:
            f.write(f"Mean:\t{mean}\nStd:\t{std}")
        with open(os.path.join(self.data_dir, "failed.txt"), "w") as f:
            f.write("\n".join(self.failed_files))
        subsample.write_parquet(os.path.join(self.data_dir, "subsample.parquet"))
        

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
    data_conf = OmegaConf.load("./configs/data/dataset/DCLDE_test.yaml")
    trans_conf = OmegaConf.load("./configs/data/transform/melbank_dclde_test.yaml")
    test = TorcaTest(dataset_config=data_conf, transform_config=trans_conf)
    test.run_test()