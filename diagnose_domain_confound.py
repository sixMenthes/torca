"""Domain-confound diagnostic for the frozen Bird-MAE backbone.

Question this answers: do the frozen backbone embeddings of the *training* clips
organize by recording condition (hydrophone / date) rather than by biology
(call type / species)? If so, the frozen features are largely capturing the
channel and noise floor, not the calls -- which both explains poor ProtoPNet
transfer to held-out hydrophones and motivates continued (domain-adapted)
pretraining or the PCEN front-end.

What it does, in one pass over a stratified subset of the train split:
  1. embed each clip with VIT_ppnet.embed -> three (N, D) views:
       focal      = pooled focal-similarity grid the prototypes actually see
       patch_mean = plain global-pool baseline
       cls        = CLS token
  2. linear (logistic) probes on each view:
       hydrophone : plain stratified CV  -> HOW decodable is the station.
       date-bucket: within-hydrophone CV -> the *clean* nuisance test (same
                    node, different day should NOT be decodable from a call
                    representation).
       label      : GroupKFold by hydrophone -> can we decode biology WITHOUT
                    leaning on station identity. This is the signal reference.
  3. a 2-D UMAP (PCA fallback) coloured by hydrophone and by date-bucket.

Smoking gun = nuisance probes near-perfect while the grouped label probe sits
near its majority baseline.

Run ON THE CLUSTER (needs the parquet, the Bird-MAE weights, and ideally a GPU):

    python diagnose_domain_confound.py \
        --n-per-hydro 400 --out runs/diag_nopcen -- \
        module.network.pcen.enable=false

Re-run with module.network.pcen.enable=true (and thus the linear-mel transform,
which is interpolated from that flag) to test whether PCEN suppresses the
nuisance axis. Any override after `--` is forwarded to Hydra compose.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

# --- project root (dir containing .git), mirrors finetune.py -----------------
_ROOT = Path(__file__).resolve()
while not (_ROOT / ".git").exists():
    if _ROOT.parent == _ROOT:
        raise FileNotFoundError("could not find project root (no .git found)")
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

from build_model import build_model                      # noqa: E402
from torca_datamodule import LabelDataModule             # noqa: E402
from torca_transforms import BaseTransform               # noqa: E402

# --- date bucket -------------------------------------------------------------
# Recording month, preferred from a real timestamp column (UTC) and falling back
# per-row to the date embedded in the Soundfile name (e.g. "...20131011_..."),
# so this works whether or not UTC has been carried into the parquet yet.
def build_date_bucket(df: pl.DataFrame) -> pl.DataFrame:
    """Add a 'date_bucket' (YYYY-MM) column. coalesce = UTC first, Soundfile
    second, null if neither yields a date."""
    sources = []
    if "UTC" in df.columns:
        sources.append(
            pl.col("UTC").cast(pl.Utf8).str.to_datetime(strict=False).dt.strftime("%Y-%m")
        )
    if "Soundfile" in df.columns:
        y = pl.col("Soundfile").str.extract(r"(20\d{2})[-_]?\d{2}[-_]?\d{2}", 1)
        mo = pl.col("Soundfile").str.extract(r"20\d{2}[-_]?(\d{2})[-_]?\d{2}", 1)
        sources.append(
            pl.when(y.is_not_null() & mo.is_not_null())
            .then(y + pl.lit("-") + mo)
            .otherwise(None)
        )
    bucket = pl.coalesce(sources) if sources else pl.lit(None, dtype=pl.Utf8)
    return df.with_columns(bucket.alias("date_bucket"))


# --- dataset that carries metadata alongside the features --------------------
class MetaClipDataset(Dataset):
    """Like LabelDataset but returns (features, meta) so we keep hydrophone /
    date / label per clip. Missing files -> None (dropped in collate)."""

    def __init__(self, df: pl.DataFrame, transform: BaseTransform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.height

    def __getitem__(self, i):
        import os

        import soundfile as sf

        row = self.df.row(i, named=True)
        path = row["LocalPath"]
        if not os.path.exists(path):
            return None
        audio, _ = sf.read(path, dtype="float32", always_2d=True)
        wave = torch.from_numpy(audio).T
        feats = self.transform(wave.data)  # (1, T, F)
        meta = {
            "hydro": row["Dataset"],
            "date": row["date_bucket"],
            "label": row["Labels"],
            "call": row.get("CalltypeCategory"),
        }
        return feats, meta


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    feats = torch.stack([b[0] for b in batch])
    metas = {k: [b[1][k] for b in batch] for k in batch[0][1]}
    return feats, metas


# --- probes ------------------------------------------------------------------
def _majority_baseline(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    return counts.max() / counts.sum()


def probe(X, y, groups=None, min_per_class=10, max_splits=5):
    """Balanced-accuracy of a logistic probe via CV. GroupKFold when `groups`
    is given (label probe), else StratifiedKFold. Returns (score, baseline,
    n_classes) or None if there isn't enough signal to probe honestly."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                          cross_val_score)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y)
    keep_classes = [c for c, cnt in zip(*np.unique(y, return_counts=True))
                    if cnt >= min_per_class]
    if len(keep_classes) < 2:
        return None
    mask = np.isin(y, keep_classes)
    X, y = X[mask], y[mask]
    groups = None if groups is None else np.asarray(groups)[mask]

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    min_count = np.unique(y, return_counts=True)[1].min()
    n_splits = min(max_splits, int(min_count))
    if groups is not None:
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            return None
        cv = GroupKFold(n_splits=min(n_splits, n_groups))
        scores = cross_val_score(clf, X, y, groups=groups, cv=cv,
                                 scoring="balanced_accuracy")
    else:
        cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True,
                             random_state=0)
        scores = cross_val_score(clf, X, y, cv=cv,
                                 scoring="balanced_accuracy")
    return float(scores.mean()), _majority_baseline(y), len(keep_classes)


def within_hydro_date_probe(X, hydro, dates, min_clips=80):
    """Predict date-bucket *within* each hydrophone, then average balanced
    accuracy weighted by clip count. The clean nuisance test: same station,
    different day should not be decodable from a call representation."""
    hydro = np.asarray(hydro)
    dates = np.asarray([d if d is not None else "NA" for d in dates])
    have_date = dates != "NA"
    if have_date.sum() < min_clips:
        return None
    results, weights = [], []
    for h in np.unique(hydro):
        m = (hydro == h) & have_date
        if m.sum() < min_clips:
            continue
        r = probe(X[m], dates[m])
        if r is not None:
            results.append(r[0])
            weights.append(int(m.sum()))
    if not results:
        return None
    w = np.array(weights)
    return float(np.average(results, weights=w)), int(w.sum()), len(results)


# --- plotting ----------------------------------------------------------------
def scatter_2d(emb2d, labels, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = np.asarray([str(l) for l in labels])
    fig, ax = plt.subplots(figsize=(8, 7))
    for lab in sorted(np.unique(labels)):
        m = labels == lab
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=6, alpha=0.6, label=lab)
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=7, loc="best", ncol=2)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def project_2d(X):
    try:
        import umap
        return umap.UMAP(n_neighbors=30, min_dist=0.1,
                         random_state=0).fit_transform(X), "UMAP"
    except Exception as e:
        from sklearn.decomposition import PCA
        print(f"[warn] UMAP unavailable ({e}); falling back to PCA.")
        return PCA(n_components=2).fit_transform(X), "PCA"


# --- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-hydro", type=int, default=400,
                    help="max clips sampled per hydrophone from the train split")
    ap.add_argument("--max-total", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", type=str, default="runs/domain_diag")
    ap.add_argument("--seed", type=int, default=59)
    ap.add_argument("--config-name", type=str, default="torca")
    ap.add_argument("overrides", nargs="*",
                    help="Hydra overrides, e.g. module.network.pcen.enable=true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # --- config + datamodule (reuse the exact split/path logic) --------------
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base=None,
                               config_dir=str(_ROOT / "configs")):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    pcen_on = bool(cfg.module.network.get("pcen", {}).get("enable", False))
    print(f"[info] PCEN enabled: {pcen_on}")

    dm = LabelDataModule(
        dataset_configs=cfg.data.dataset,
        loader_configs=cfg.data.loaders,
        transform_configs=cfg.data.transform,
    )
    try:
        dm.prepare_data()  # on a prestaged node this just filters to cached clips
    except Exception as e:
        print(f"[warn] prepare_data failed ({e}); using unfiltered manifest.")

    train_df = dm.build_set("train")
    # stratified subsample: up to n_per_hydro clips per hydrophone.
    sub = (
        train_df.filter(pl.col("Dataset").is_not_null())
        .with_columns(
            pl.int_range(pl.len()).shuffle(seed=args.seed).over("Dataset").alias("_r")
        )
        .filter(pl.col("_r") < args.n_per_hydro)
        .drop("_r")
    )
    sub = build_date_bucket(sub)
    if sub.height > args.max_total:
        sub = sub.sample(n=args.max_total, seed=args.seed)
    print(f"[info] embedding {sub.height} clips across "
          f"{sub['Dataset'].n_unique()} hydrophones")

    transform = BaseTransform(cfg.data.transform)
    loader = DataLoader(
        MetaClipDataset(sub, transform),
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, collate_fn=collate,
    )

    # --- model ---------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg.module, dm.label_map)
    wpath = cfg.module.network.get("pretrained_weights_path", None)
    if wpath:
        print(f"[info] loading pretrained weights: {wpath}")
        model.load_pretrained_weights(wpath, cfg.data.dataset.name)
    model.to(device).eval()

    # --- embed ---------------------------------------------------------------
    views = {"focal": [], "patch_mean": [], "cls": []}
    meta = {"hydro": [], "date": [], "label": [], "call": []}
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            feats, metas = batch
            emb = model.embed(feats.to(device))
            for k in views:
                views[k].append(emb[k].float().cpu().numpy())
            for k in meta:
                meta[k].extend(metas[k])
    views = {k: np.concatenate(v) for k, v in views.items()}
    n = len(meta["hydro"])
    print(f"[info] embedded {n} clips (some may have been dropped as missing)")

    # persist raw embeddings + metadata for offline re-analysis
    np.savez(out / "embeddings.npz", **views)
    pl.DataFrame(meta).write_parquet(out / "metadata.parquet")

    # --- probes --------------------------------------------------------------
    lines = [f"# Domain-confound diagnostic (PCEN={'on' if pcen_on else 'off'})",
             f"clips embedded: {n}",
             f"hydrophones: {sorted(set(meta['hydro']))}", ""]
    hydro, dates, label = meta["hydro"], meta["date"], meta["label"]
    n_dated = sum(d is not None for d in dates)
    lines.append(f"clips with a parseable date: {n_dated}/{n}")
    if n_dated < 0.5 * n:
        lines.append("  [!] most dates are null -- no usable UTC column and the "
                     "Soundfile fallback failed. Date probe may be unreliable.")
    lines.append("")
    lines.append("balanced-accuracy (chance = majority baseline in parens):")
    lines.append(f"{'view':<12}{'hydrophone':<22}{'date|hydro':<22}"
                 f"{'label(grouped)':<22}")

    def fmt(r):
        return "n/a" if r is None else f"{r[0]:.2f} ({r[1]:.2f})"

    for view, X in views.items():
        h = probe(X, hydro)
        d = within_hydro_date_probe(X, hydro, dates)
        lab = probe(X, label, groups=hydro)
        # date/hydro tuple is (score, n, n_hydros); no baseline -> show n_hydros
        dstr = "n/a" if d is None else f"{d[0]:.2f} [{d[2]}h]"
        lines.append(f"{view:<12}{fmt(h):<22}{dstr:<22}{fmt(lab):<22}")

    lines += [
        "",
        "Reading it: hydrophone & date|hydro high while label(grouped) ~ its",
        "baseline  =>  frozen features are dominated by recording condition,",
        "not biology. label(grouped) uses GroupKFold by hydrophone, so it can't",
        "cheat via station identity.",
    ]
    report = "\n".join(lines)
    (out / "report.txt").write_text(report + "\n")
    print("\n" + report + "\n")

    # --- 2-D projection ------------------------------------------------------
    for view in ("focal", "patch_mean"):
        emb2d, method = project_2d(views[view])
        scatter_2d(emb2d, hydro,
                   f"{view} by hydrophone ({method}, PCEN={pcen_on})",
                   out / f"{view}_by_hydro.png")
        scatter_2d(emb2d, [d or "NA" for d in dates],
                   f"{view} by date ({method}, PCEN={pcen_on})",
                   out / f"{view}_by_date.png")
    print(f"[done] wrote embeddings, report, and plots to {out}")


if __name__ == "__main__":
    main()
