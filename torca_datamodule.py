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
        transform_configs: DictConfig,
    ):
        super().__init__()

        # kept whole so subclasses can read fields the base class does not know about
        # (e.g. SelfDistillDataModule's `model_name`, which selects the front-end)
        self.dataset_configs = dataset_configs
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
        self.gcl = gcsfs.core.GCSFileSystem(token="anon")
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
        # On a pre-staged (offline) node the clips and a curated manifest already
        # exist, so skip the GCS download entirely and keep only the rows whose clip
        # actually landed.
        # Name comes from the dataset config so that caches for different
        # clip_durations cannot collide: the clips themselves are named
        # {start_ms}-{end_ms}.wav, so a 5 s stage and a 3 s stage share no files, and
        # a shared manifest name would make one silently masquerade as the other.
        # Default preserves the original name for datasets that don't set it.
        manifest = self.dataset_configs.get("manifest_name", "DCLDE_no_balance")
        cached = Path(self.data_dir) / manifest
        if cached.exists():
            # Match per CLIP. This used to filter on Soundfile, which is shared by ~16
            # annotations, so it really asked "did ANY window from this source file
            # land". A partially-fetched file therefore kept every one of its rows,
            # including windows that were never written; those reached __getitem__,
            # failed os.path.exists, logged "Failed loading file" and were dropped by
            # collate_fn_skip — shrinking batches by a silently varying amount, every
            # epoch. The manifest has always carried per-clip rows; only the read side
            # was coarse.
            #
            # Keyed on the tail of LocalPath rather than the whole thing because the
            # cached manifest was written on the PRESTAGE node and its absolute paths
            # carry that node's dataset_dir (on the cluster the tarball is extracted
            # somewhere else entirely). The tail is dataset_dir-independent, and taking
            # it — rather than re-deriving the path here — leaves build_clip_manifest
            # the single source of truth for how clips are named.
            cached_df = pl.read_parquet(cached)
            if "LocalPath" not in cached_df.columns:
                raise RuntimeError(
                    f"the cached manifest {cached} has no LocalPath column, so it "
                    f"predates per-clip matching (the old inline prepare_data wrote "
                    f"self.df and only Soundfile was ever read back). Renaming such a "
                    f"file to the current manifest_name does NOT convert it — it lists "
                    f"source files, not clips. Rebuild it from what is on disk:\n"
                    f"  python prestage_clips.py --dataset-dir {self.data_dir} "
                    f"--clip-duration {self.clip_duration} "
                    f"--manifest-name {manifest} --manifest-only"
                )
            ok = set(clip_key(cached_df.get_column("LocalPath")))
            self.df = self.df.filter(clip_key(pl.col("LocalPath")).is_in(ok))
            if self.df.height == 0:
                raise RuntimeError(
                    f"the cached manifest {cached} shares no clips with this run's "
                    f"manifest. Almost always clip_duration: the window is baked into "
                    f"every filename as {{start_ms}}-{{end_ms}}.wav, so a stage at "
                    f"another duration matches nothing at all (this run: "
                    f"clip_duration={self.clip_duration}). Restage at this duration, "
                    f"or point manifest_name at the right cache."
                )
            return

        # No manifest. That used to mean "download everything", which was right when
        # staging happened inline. Prestaging is now a separate deliberate step
        # (prestage_clips.py, and on the cluster a login-node job, since compute nodes
        # have no internet), so a missing manifest almost always means a wrong
        # dataset_dir or a renamed manifest — and kicking off a 206k-clip GCS download
        # is a slow and expensive way to discover that. Fail loudly; opt back in with
        # data.dataset.allow_download=true.
        data_dir = Path(self.data_dir)
        has_clips = data_dir.exists() and next(data_dir.rglob("*.wav"), None) is not None

        if not self.dataset_configs.get("allow_download", False):
            if has_clips:
                hint = (f"clips ARE present under {data_dir}, so the manifest is just "
                        f"missing or renamed. Rebuild it without downloading:\n"
                        f"  python prestage_clips.py --dataset-dir {data_dir} "
                        f"--clip-duration {self.clip_duration} "
                        f"--manifest-name {manifest} --manifest-only")
            else:
                hint = (f"no .wav files under {data_dir} either — check "
                        f"paths.dataset_dir, then stage with:\n"
                        f"  python prestage_clips.py --dataset-dir <dir> "
                        f"--clip-duration {self.clip_duration} "
                        f"--manifest-name {manifest}")
            raise FileNotFoundError(
                f"manifest not found: {cached}\n{hint}\n"
                f"(set data.dataset.allow_download=true to download inline instead)"
            )

        self.download_set(self.df)
        self.df = self.df.filter(~pl.col("Soundfile").is_in(set(self.failed_files)))
        self.df.write_parquet(cached)

    def setup(self, stage: str):

        if stage == "fit":
            train_transform = TrainTransform(
                self.transform_config, background_paths=self._background_bank()
            )
            self.train_set = LabelDataset(
                self.build_set("train"), train_transform, self.label_map
            )
            val_transform = BaseTransform(self.transform_config)
            self.val_set = LabelDataset(
                self.build_set("val"), val_transform, self.label_map
            )
            self.call_set = CallDataset(
                self.build_set("calls"), val_transform, self.call_map
            )

        if stage == "test":
            test_transform = BaseTransform(self.transform_config)
            self.test_set = LabelDataset(
                self.build_set("test"), test_transform, self.label_map
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            num_workers=self.train_loader_configs.num_workers,
            batch_size=self.train_loader_configs.batch_size,
            shuffle=self.train_loader_configs.shuffle,
            # .get with a default: not every loaders config defines these
            # (default.yaml defines neither, audioset_balanced.yaml omits
            # persistent_workers), and a missing key is a ConfigAttributeError
            # under hydra's struct mode rather than a fallback.
            persistent_workers=self.train_loader_configs.get(
                "persistent_workers", False
            ),
            pin_memory=self.train_loader_configs.get("pin_memory", False),
        )

    def _background_bank(self):
        # Background-labelled clips reused as an additive-noise bank. Exclude
        # test/val hydrophones so held-out recording conditions don't leak in.
        non_train = self.test_hydros + self.val_hydros
        return (
            self.df.filter(
                pl.col("Labels") == "Background", ~pl.col("Dataset").is_in(non_train)
            )
            .get_column("LocalPath")
            .to_list()
        )

    def val_dataloader(self):
        persistent = self.val_loader_configs.get("persistent_workers", False)
        pin = self.val_loader_configs.get("pin_memory", False)
        val_dataloader = DataLoader(
            self.val_set,
            num_workers=self.val_loader_configs.num_workers,
            batch_size=self.val_loader_configs.batch_size,
            shuffle=self.val_loader_configs.shuffle,
            persistent_workers=persistent,
            pin_memory=pin,
        )
        call_dataloader = DataLoader(
            self.call_set,
            num_workers=self.val_loader_configs.num_workers,
            batch_size=self.val_loader_configs.batch_size,
            shuffle=self.val_loader_configs.shuffle,
            persistent_workers=persistent,
            pin_memory=pin,
        )
        return [val_dataloader, call_dataloader]

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            num_workers=self.test_loader_configs.num_workers,
            batch_size=self.test_loader_configs.batch_size,
            shuffle=self.test_loader_configs.shuffle,
        )

    def download_file(self, row: pl.Series):
        gcs_path = row["GCSPath"]
        starts = row["new_start_time"]
        ends = row["new_end_time"]
        paths = row["LocalPath"]

        to_do = [
            (s, e, p) for s, e, p in zip(starts, ends, paths) if not Path(p).exists()
        ]

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
                        audiodata = snd.read(duration, dtype="float32")
                        Path(sp).parent.mkdir(parents=True, exist_ok=True)
                        sf.write(sp, audiodata, sr)

        except Exception as e:
            log.warning(f"Failed {gcs_path}: {e}")
            return gcs_path

    def download_set(self, df: pl.DataFrame):
        to_download = df.group_by("GCSPath", maintain_order=True).agg(
            "Soundfile", "new_start_time", "new_end_time", "LocalPath"
        )

        l = to_download.height

        with tqdm(total=l, desc="Downloading files") as pbar:
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = [
                    executor.submit(self.download_file, row)
                    for row in to_download.iter_rows(named=True)
                ]
                for fut in as_completed(futures):
                    failed = fut.result()
                    if failed:
                        self.failed_files.append(PurePath(failed).name)
                    pbar.update(1)

    def build_set(self, split: str):

        # assert (split in {"test", "val", "train"}), "split must be one of train, val or test"
        # also i should assert that classes to balance is a subset of labels
        df = self.df.filter(pl.col("Labels").is_in(self.labels))

        if split == "test":
            return df.filter(pl.col("Dataset").is_in(self.test_hydros)).with_columns(
                pl.lit("test").alias("split")
            )
        elif split == "val":
            return df.filter(pl.col("Dataset").is_in(self.val_hydros)).with_columns(
                pl.lit("val").alias("split")
            )
        elif split == "calls":
            non_train_hydros = self.low_sr_hydros + self.val_hydros + self.test_hydros
            return (
                df.filter(~pl.col("Dataset").is_in(non_train_hydros))
                .filter(pl.col("Labels") == "SRKW")
                .drop_nulls(pl.col("CalltypeCategory"))
                .filter(pl.col("CalltypeCategory").is_in(self.calls))
            )
        else:  # train
            pool = df.filter(
                ~pl.col("Dataset").is_in(self.test_hydros),
                ~pl.col("Dataset").is_in(self.low_sr_hydros),
                ~pl.col("Dataset").is_in(self.val_hydros),
            )
            if self.class_to_balance:
                rest = pool.filter(
                    ~pl.col("Labels").is_in(self.class_to_balance.keys())
                )
                srkw_time = self.df.filter(pl.col("Labels") == "SRKW")[
                    "true_duration"
                ].sum()
                strat_samples = [
                    stratified_sampling(c, srkw_time * frac, pool)
                    for c, frac in self.class_to_balance.items()
                ]
                return pl.concat([rest, *strat_samples]).with_columns(
                    pl.lit("train").alias("split")
                )
            return pool.with_columns(pl.lit("train").alias("split"))

    def load_df(self):
        return build_clip_manifest(
            pl.read_parquet(self.parquet_path), self.clip_duration, self.data_dir
        )


def clip_key(col):
    """Identity of a clip, independent of which machine staged it.

    LocalPath is `{data_dir}/{Provider}/{Dataset}/{stem}/{start_ms}-{end_ms}.wav`.
    Everything after data_dir is reproducible from the annotation row alone, so the
    last four components identify a clip across a prestage node and a training node
    that mount the tree at different paths. Works on both a Series and an Expr, so
    the two sides of the manifest join are written the same way.
    """
    return col.str.extract(r"([^/]+/[^/]+/[^/]+/[^/]+\.wav)$", 1)


def build_clip_manifest(df, clip_duration, data_dir):
    """Annotation rows -> centred clip windows + LocalPath. SINGLE SOURCE OF TRUTH.

    Both `LabelDataModule.load_df` and the standalone prestage script call this, and
    they must: `LocalPath` encodes the window as `{start_ms}-{end_ms}.wav`, so any
    divergence in how the window is computed silently renames every clip. The
    datamodule then finds nothing on disk, every row returns None, `collate_fn_skip`
    returns None, and you get empty batches rather than an error. Changing
    `clip_duration` renames every file for the same reason — a 5 s prestage does not
    serve a 3 s run.

    The window is centred on the annotation's midpoint and clamped at the file start,
    so clips carry real recorded context rather than padding; only annotations near a
    file's end come up short.
    """
    duration = pl.col("FileEndSec") - pl.col("FileBeginSec")
    center_time = pl.col("FileBeginSec") + (duration / 2.0)
    new_start_time = pl.max_horizontal(pl.lit(0), center_time - clip_duration / 2.0)
    new_end_time = new_start_time + clip_duration

    df = (
        df.filter(pl.col("NewFileOk") & (pl.col("Labels") != "KW_und"))
        .with_columns(
            duration.alias("true_duration"),
            center_time.alias("center_time"),
            new_start_time.alias("new_start_time"),
            new_end_time.alias("new_end_time"),
        )
        .drop(
            pl.col("FileBeginSec"),
            pl.col("FileEndSec"),
            pl.col("Duration"),
        )
    )

    stem = pl.col("Soundfile").str.replace(r"\.[^.]+$", "")
    start_ms = (
        (pl.col("new_start_time") * 1000).round().cast(pl.Int64).cast(pl.String)
        .str.zfill(10)
    )
    end_ms = (
        (pl.col("new_end_time") * 1000).round().cast(pl.Int64).cast(pl.String)
        .str.zfill(10)
    )

    local_path = pl.format(
        "{}/{}/{}/{}/{}-{}{}",
        pl.lit(data_dir),
        pl.col("Provider"),
        pl.col("Dataset"),
        stem,
        start_ms,
        end_ms,
        pl.lit(".wav"),
    )

    return df.rename({"NewPath": "GCSPath"}).with_columns(local_path.alias("LocalPath"))


def stratified_sampling(
    label: str, tgt_duration: float, df: pl.DataFrame, curr_duration=0, seed=59
):
    seed += 1
    remaining_duration = tgt_duration - curr_duration
    if remaining_duration < 300:
        return df.clear()

    pool = df.filter(pl.col("Labels") == label)
    duration_per_hydro = remaining_duration / pool["Dataset"].n_unique()
    cumul_per_hydro = pl.col("true_duration").cum_sum().over("Dataset")

    new_samples = pool.sample(fraction=1.0, shuffle=True).filter(
        cumul_per_hydro <= duration_per_hydro
    )

    added = new_samples["true_duration"].sum()

    if added == 0:
        return df.clear()

    return pl.concat(
        [
            new_samples,
            stratified_sampling(
                label, tgt_duration, df, curr_duration + added, seed=seed
            ),
        ]
    )


if __name__ == "__main__":
    data_conf = OmegaConf.load("configs/data/dataset/DCLDE_test.yaml")
    loader_conf = OmegaConf.load("configs/data/loaders/default.yaml")
    transform_conf = OmegaConf.load(
        "/Users/leo/projects/orcas/torca/configs/data/transform/melbank_dclde_test.yaml"
    )
    module = LabelDataModule(data_conf, loader_conf, transform_conf)
