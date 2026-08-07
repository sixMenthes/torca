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


def probe_pool(df, dataset_cfg, include_low_sr=False):
    """The rows a probe run extracts features for, and their split tags. One call.

    This replaces `labelled_pool` + a separate `split_tags` + a hand-written filter at
    the call site. That arrangement had three gates on three different lines and only
    two of them ran before the backbone: rows were cut to labelled ones, features were
    computed for ALL of them, and the split was applied afterwards by masking the tag
    array. Low-SR clips therefore cost a full forward pass each and were then never
    selected by any mask — invisible in the results, expensive in wall-clock. Worse,
    df and tags were built independently, so any filter added to one had to be
    mirrored onto the other by hand or every split assignment shifted silently.

    Returning both together makes that class of bug unrepresentable: there is one
    filter, applied once, and the tags are derived from the rows that survive it.

    Three gates, all of them here:
      * LABELLED   — Labels in dataset_cfg.labels. The parquet carries rows whose
                     label is outside the 4-class ecotype set; they have no target.
      * MATERIALISED — NOT checked. Deliberately: prepare_data owns "exists on disk",
                     via the cached manifest, and that manifest is what travels to the
                     cluster inside the tarball. A second definition here would drift.
      * IN-SPLIT   — low_sr dropped unless include_low_sr. See split_tags for why the
                     tag exists and configs/probe.yaml for why it stays off.

    Returns (df, tags) aligned row-for-row.
    """
    import probe as probe_lib

    pool = df.filter(pl.col("Labels").is_in(list(dataset_cfg.labels)))
    tags = probe_lib.split_tags(
        pool.get_column("Dataset").to_list(),
        list(dataset_cfg.test_hydros),
        list(dataset_cfg.val_hydros),
        list(dataset_cfg.low_sr_hydros),
    )
    if not include_low_sr:
        keep = tags != "low_sr"
        pool, tags = pool.filter(pl.Series(keep)), tags[keep]
    return pool, tags


def split_report(df, tags, dataset_cfg, source_df=None):
    """Human-readable composition of every split, per probe task. Returns a string.

    Exists because the three probes each build their own mask over `tags` in a
    different file, so "what is this number computed on" was only answerable by
    reading three call sites. Print it with the results, or standalone via
    report_splits.py.
    """
    import numpy as np

    labels = list(dataset_cfg.labels)
    calls = set(dataset_cfg.calls)
    hyd = np.asarray(df.get_column("Dataset").to_list())
    lab = np.asarray(df.get_column("Labels").to_list())
    has_call = np.asarray([
        c in calls for c in df.get_column("CalltypeCategory").to_list()
    ]) if "CalltypeCategory" in df.columns else np.zeros(df.height, dtype=bool)
    is_bg = lab == "Background"

    # Provider is here because it is the constraint on ever CHANGING the split: a test
    # hydrophone that shares a provider with a train hydrophone is not a held-out
    # recording chain, it is the same rig at a different spot. BarkleyCanyon was chosen
    # for test precisely because ONC appears nowhere else.
    prov = (np.asarray(df.get_column("Provider").to_list())
            if "Provider" in df.columns else np.array([""] * df.height))

    # One column PER ECOTYPE CLASS rather than a single Background count. Two clip
    # counts cannot tell you whether a site is usable as a probe target: bush_point
    # reads as a healthy 741 clips with 231 Background, and is in fact SRKW against
    # Background with no HW and no TKW at all, so a probe fitted there measures whale
    # detection and not ecotype discrimination. That is only visible per class.
    L = []
    w = max([len(h) for h in np.unique(hyd)] + [10])
    p = max([len(x) for x in np.unique(prov)] + [8])
    cw = max([len(x) for x in labels] + [5])
    L.append("=== hydrophone -> split tag ===")
    L.append(f"  {'hydrophone':<{w}} {'provider':<{p}} {'tag':<8} {'clips':>8} "
             + " ".join(f"{lb:>{cw}}" for lb in labels)
             + f" {'calltype':>9} {'classes':>8}")
    L.append("  " + "-" * (w + p + 30 + (cw + 1) * len(labels)))
    order = {"train": 0, "val": 1, "test": 2, "low_sr": 3}
    for t, h in sorted({(str(tags[i]), hyd[i]) for i in range(len(hyd))},
                       key=lambda p_: (order[p_[0]], p_[1])):
        m = hyd == h
        pv = "/".join(sorted(set(prov[m])))
        per = [int((m & (lab == lb)).sum()) for lb in labels]
        # A site with one dominant class is not a usable probe target even when its
        # total looks generous: balanced accuracy averages per-class recall, so a class
        # holding a handful of clips contributes mostly noise, and a cross-validation
        # fold may not contain it at all.
        n_usable = sum(1 for c in per if c >= 20)
        L.append(f"  {h:<{w}} {pv:<{p}} {t:<8} {m.sum():>8} "
                 + " ".join(f"{c:>{cw}}" for c in per)
                 + f" {(m & has_call).sum():>9} {n_usable:>8}")
    L.append(f"  (the 'classes' column counts ecotype classes with >=20 clips at that "
             f"site — the online probe needs several)")
    # Any provider spanning the train/test boundary breaks the "held-out recording
    # chain" claim, which is what the whole test protocol rests on.
    shared = sorted(set(prov[tags == "train"]) & set(prov[tags == "test"]))
    if shared:
        L.append(f"  !! provider(s) on BOTH sides of train/test: {shared} — the test "
                 f"split is not a held-out recording chain")

    tr, va, te = tags == "train", tags == "val", tags == "test"
    nh = lambda m: len(np.unique(hyd[m]))

    L.append("")
    L.append("=== what each probe is computed on ===")
    L.append(f"  task/ecotype   fit  {tr.sum():>7} clips / {nh(tr):>2} hydros  (train)")
    L.append(f"                 score{te.sum():>7} clips / {nh(te):>2} hydros  (test)")
    L.append(f"                 val split holds {va.sum()} clips / {nh(va)} hydros — used ONLY when")
    L.append(f"                 c_selection=val; at the default train_cv it is untouched.")
    bgm = tr & is_bg
    small = [h for h in np.unique(hyd[bgm]) if (hyd[bgm] == h).sum() < 50]
    L.append(f"  nuisance/bg    fit  {bgm.sum():>7} clips / {nh(bgm):>2} hydros  "
             f"(train AND Background)")
    L.append(f"                 stratified CV; sites with <50 background clips dropped"
             f"{' -> ' + ', '.join(small) if small else ' (none)'}")
    L.append(f"  task/calltype  fit  {(tr & has_call).sum():>7} clips / {nh(tr & has_call):>2} hydros  "
             f"(train AND calltype)")
    L.append(f"                 score{(te & has_call).sum():>7} clips / {nh(te & has_call):>2} hydros  "
             f"(test AND calltype)")

    L.append("")
    L.append("=== excluded ===")
    if source_df is not None:
        n_unlab = source_df.height - source_df.filter(
            pl.col("Labels").is_in(labels)).height
        L.append(f"  unlabelled     {n_unlab:>7} rows  (Labels outside {labels})")
        low = source_df.filter(
            pl.col("Dataset").is_in(list(dataset_cfg.low_sr_hydros)))
        L.append(f"  low_sr         {low.height:>7} rows / "
                 f"{low.get_column('Dataset').n_unique()} hydros  "
                 f"(no probe mask selects these; now dropped before extraction)")
        # A config name with no rows in the manifest has TWO very different causes and
        # only one of them is a bug, so do not report them the same way:
        #
        #   absent      the site genuinely has no materialised clips (never fetched, or
        #               no rows at this clip_duration). Nothing is misclassified —
        #               there are no rows to misclassify. Benign for the split; it is a
        #               data-completeness fact, and one worth knowing separately.
        #   misspelled  the site IS in the manifest under a different string, and THAT
        #               string is the one that matters: split_tags falls through to
        #               "train", so a band-limited site lands in the probe's fit set
        #               and nothing reports it.
        #
        # The two are indistinguishable from the config alone, which is why the check
        # offers close matches among the names that ARE present — that is what tells
        # them apart.
        import difflib

        present = set(source_df.get_column("Dataset").unique().to_list())
        untagged = sorted(
            present
            - set(dataset_cfg.test_hydros)
            - set(dataset_cfg.val_hydros)
            - set(dataset_cfg.low_sr_hydros)
        )
        for key in ("test_hydros", "val_hydros", "low_sr_hydros"):
            for h in dataset_cfg[key]:
                if h in present:
                    continue
                near = difflib.get_close_matches(h, untagged, n=3, cutoff=0.6)
                if near:
                    L.append(f"  !! {key}: '{h}' has no rows, but these UNTAGGED "
                             f"manifest names resemble it: {near}")
                    L.append(f"     -> if one of those is the same site, it is being "
                             f"treated as TRAIN. Fix the spelling in the config.")
                else:
                    L.append(f"  -- {key}: '{h}' has no rows in the manifest and no "
                             f"similar name is untagged (site absent, not misspelled)")
    L.append("")
    L.append("  NOTE the asymmetry with adaptation, which is deliberate: "
             "SelfDistillDataModule")
    L.append("  holds out test+val ONLY, so the SSL pool INCLUDES the low-SR sites, and "
             "the")
    L.append("  background bank draws noise from them too. Adapt on everything, "
             "evaluate clean.")
    return "\n".join(L)
