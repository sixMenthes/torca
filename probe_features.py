"""Backbone -> pooled features, the missing link between a checkpoint and probe.py.

`probe.py` is the measuring instrument but takes numpy arrays; this module produces
them. Three extraction sources, all yielding the same (X, meta) contract so they are
directly comparable:

  * an ADAPTED encoder    (a MIMDistillation checkpoint),
  * a FROZEN encoder      (pretrained weights, no adaptation) — the control,
  * MFCC                  — the non-neural baseline.

Everything is mean-pooled per clip and every row carries its hydrophone, so the
co-primary metric (task decodability up, nuisance decodability down) can be computed
straight from the output.
"""

import numpy as np
import polars as pl
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset

from util.pylogger import get_pylogger

log = get_pylogger(__name__)


class ProbeDataset(Dataset):
    """Labelled clips for probing. Mirrors SelfDistillDataset's loading exactly.

    Deliberately shares `_load_wave`'s semantics with the SSL dataset (highest-energy
    channel, resample, centre-crop / tail-pad) so that probe features are extracted
    from the same signal the adaptation saw. Any divergence here would confound the
    frozen-vs-adapted comparison with a preprocessing difference.
    """

    def __init__(self, df, sample_rate, clip_duration, label_map, call_map=None):
        self.df = df
        self.sample_rate = int(sample_rate)
        self.max_length = int(sample_rate * clip_duration)
        self.label_map = label_map
        self.call_map = call_map or {}

    def __len__(self):
        return self.df.height

    def _load_wave(self, path):
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        wave = torch.from_numpy(audio).T
        idx = int(torch.argmax((wave ** 2).mean(1))) if wave.size(0) > 1 else 0
        wave = wave[idx].unsqueeze(0)
        if sr != self.sample_rate:
            wave = AF.resample(wave, sr, self.sample_rate)
        t = wave.size(-1)
        if t > self.max_length:
            start = (t - self.max_length) // 2
            wave = wave[..., start:start + self.max_length]
        elif t < self.max_length:
            wave = F.pad(wave, (0, self.max_length - t))
        return wave

    def __getitem__(self, index):
        row = self.df.row(index, named=True)
        try:
            wave = self._load_wave(row["LocalPath"])
        except Exception:
            return None
        return {
            "wave": wave,
            "label": self.label_map.get(row["Labels"], -1),
            "call": self.call_map.get(row.get("CalltypeCategory"), -1),
            "dataset": row["Dataset"],
        }


def _collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "wave": torch.stack([b["wave"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
        "call": torch.tensor([b["call"] for b in batch]),
        "dataset": [b["dataset"] for b in batch],
    }


def build_loader(df, sample_rate, clip_duration, label_map, call_map=None,
                 batch_size=32, num_workers=4):
    return DataLoader(
        ProbeDataset(df, sample_rate, clip_duration, label_map, call_map),
        batch_size=batch_size, num_workers=num_workers, shuffle=False,
        collate_fn=_collate,
    )


@torch.no_grad()
def extract_backbone(encoder, loader, layer=None, device="cpu"):
    """Encoder -> (X, meta). `layer` selects an intermediate block (0-based).

    Mean-pooled PRE-projector tokens: the standard SSL evaluation target. Sweeping
    `layer` is how you locate the confound — recording condition is a low-level
    property, so it plausibly lives in early blocks, and watching nuisance
    decodability fall across depth after adaptation is a mechanism result rather than
    just an endpoint number.
    """
    encoder = encoder.to(device).eval()
    X, labels, calls, datasets = [], [], [], []
    for batch in loader:
        if batch is None:
            continue
        feats = encoder.pooled(batch["wave"].to(device), layer=layer)
        X.append(feats.float().cpu().numpy())
        labels.append(batch["label"].numpy())
        calls.append(batch["call"].numpy())
        datasets.extend(batch["dataset"])
    return _stack(X, labels, calls, datasets)


@torch.no_grad()
def extract_mfcc(loader, sample_rate, n_mfcc=40, include_deltas=True):
    """Non-neural baseline: mean+std pooled MFCCs, the floor every cell must beat.

    Mean AND std over time (not mean alone) because a single mean vector discards all
    temporal structure, which would make the baseline weaker than it should be — an
    artificially low floor flatters every neural cell above it. Deltas add coarse
    dynamics for the same reason.
    """
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sample_rate, n_mfcc=n_mfcc,
        melkwargs={"n_fft": 1024, "hop_length": 320, "n_mels": 128},
    )
    X, labels, calls, datasets = [], [], [], []
    for batch in loader:
        if batch is None:
            continue
        m = mfcc(batch["wave"].squeeze(1))                 # (B, n_mfcc, T)
        feats = [m.mean(-1), m.std(-1)]
        if include_deltas:
            d = torchaudio.functional.compute_deltas(m)
            feats += [d.mean(-1), d.std(-1)]
        X.append(torch.cat(feats, dim=-1).float().numpy())
        labels.append(batch["label"].numpy())
        calls.append(batch["call"].numpy())
        datasets.extend(batch["dataset"])
    return _stack(X, labels, calls, datasets)


def _stack(X, labels, calls, datasets):
    return np.concatenate(X, 0), {
        "label": np.concatenate(labels, 0),
        "call": np.concatenate(calls, 0),
        "hydrophone": np.array(datasets),
    }


def encoder_from_checkpoint(ckpt_path, map_location="cpu"):
    """Student encoder out of a MIMDistillation checkpoint (the ADAPTED cell).

    The student, not the teacher: the teacher is an EMA trailing copy kept for target
    generation. Reporting the teacher would measure a lagged average of the thing you
    actually trained.
    """
    from mim_distillation import MIMDistillation

    model = MIMDistillation.load_from_checkpoint(ckpt_path, map_location=map_location)
    return model.student["encoder"].eval()


def labelled_pool(df, labels):
    """Rows usable for probing: a known ecotype label.

    Materialisation is NOT checked here — it is the caller's job to pass a df that
    prepare_data has already filtered to clips on disk. Doing it in two places would
    mean two definitions of "present", and the manifest is the one that also travels
    to the cluster inside the tarball.
    """
    return df.filter(pl.col("Labels").is_in(list(labels)))
