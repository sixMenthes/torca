"""Print exactly what every probe is computed on. No backbone, no GPU, seconds.

    python report_splits.py                          # default (birdmae) dataset config
    python report_splits.py data/dataset=dclde_selfdistill_beats
    python report_splits.py include_low_sr=true      # what folding them in would give

Reads the same config and the same cached manifest the probe does, and calls the same
probe_pool/split_report, so it cannot report a split the probe does not use. That is
the point: the three probes each build their own mask over the tag array in a different
file, and before this the only way to answer "how many clips, from which hydrophones,
went into this number" was to read all three call sites and hold the tag cascade in
your head.

Write it to a file for the writeup:

    python report_splits.py > splits.txt
"""

import collections
import os
import sys
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

from probe_features import probe_pool, split_report
from selfdistill_datamodule import SelfDistillDataModule

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
os.environ["PROJECT_ROOT"] = str(root)


def ssl_pool_report(dm, top=25):
    """What the SELF-DISTILLATION actually trains on, broken down by hydrophone.

    The probe report above describes the evaluation splits and says nothing about the
    adaptation, which is a different pool built by different rules: it holds out only
    test and val, it applies no label filter, and it deliberately KEEPS the low-SR
    hydrophones, because more channel variety is what the invariance objective feeds
    on. That last decision is the one worth watching. If a single site supplies a large
    share of the pool, the objective is dominated by one recording chain, and if it
    supplies a large share of the BACKGROUND BANK then the "cross-hydrophone" noise
    mixed into the student view is mostly one hydrophone's noise, which is the opposite
    of what the augmentation is for.

    Calls the datamodule's own methods rather than reimplementing the filters, so this
    cannot report a pool the training does not use.
    """
    # BOTH the pool as actually built and the pool before capping. Once the caps are
    # configured, build_selfdistill_set returns the capped pool, and a report showing
    # only that could not tell you whether the cap is doing anything or what it cost.
    raw = dm._ssl_pool_uncapped()
    pool = dm.build_selfdistill_set()
    bank = dm._ssl_background_bank()
    cap_pool = dm.dataset_configs.get("max_clips_per_hydro", None)
    cap_bank = dm.dataset_configs.get("max_bank_clips_per_hydro", None)

    # LocalPath is {data_dir}/{Provider}/{Dataset}/{stem}/{start}-{end}.wav, so the
    # hydrophone is the third component from the end. Parsed from the real bank list
    # rather than recomputed from df, so the two cannot disagree.
    bank_hydros = [p.split("/")[-3] for p in bank if p.count("/") >= 3]
    bank_count = collections.Counter(bank_hydros)
    bank_total = len(bank_hydros)
    labels = list(dm.labels)

    L = []
    L.append("=== self-distillation training pool (adaptation, NOT the probe) ===")
    L.append(f"  {pool.height} clips over {pool.get_column('Dataset').n_unique()} "
             f"hydrophones; low-SR sites INCLUDED by design, no label filter")
    L.append(f"  per-hydrophone cap: {cap_pool or 'none'} "
             f"(proportional within each site's class mix)")
    L.append(f"  background bank cap: {cap_bank or 'none'}")
    if cap_pool:
        L.append(f"  before capping: {raw.height} clips "
                 f"({raw.height - pool.height} dropped)")
    L.append("")
    L.append("  Counts below are the pool AS BUILT, i.e. after capping. The cap tables")
    L.append("  further down are computed on the UNCAPPED distribution, so they still")
    L.append("  show what each candidate ceiling would do.")
    L.append("")

    # Per-class counts as well as totals. The class mix of the low-SR giants is not
    # visible anywhere else — the probe report covers only the probe pool, which
    # excludes them — and it is what a class ceiling has to be sized against. "other"
    # collects labels outside the four ecotype classes (NRKW, OKW, KW_und and so on),
    # which are in the pool because the SSL objective is label-agnostic but are targets
    # for nothing.
    by_h = {}
    for row in (pool.group_by(["Dataset", "Labels"]).len().iter_rows(named=True)):
        d = by_h.setdefault(row["Dataset"], collections.Counter())
        d[row["Labels"] if row["Labels"] in labels else "other"] += row["len"]
    totals = {h: sum(c.values()) for h, c in by_h.items()}
    order = sorted(totals, key=totals.get, reverse=True)

    cols = labels + ["other"]
    w = max([len(str(h)) for h in order] + [10])
    cw = max([len(c) for c in cols] + [6])
    L.append(f"  {'hydrophone':<{w}} {'clips':>9} {'share':>7} "
             + " ".join(f"{c:>{cw}}" for c in cols)
             + f" {'bg bank':>9} {'share':>7}")
    L.append("  " + "-" * (w + 36 + (cw + 1) * len(cols)))
    for h in order[:top]:
        n = totals[h]
        nb = bank_count.get(h, 0)
        L.append(f"  {h:<{w}} {n:>9} {100.0 * n / pool.height:>6.1f}% "
                 + " ".join(f"{by_h[h].get(c, 0):>{cw}}" for c in cols)
                 + f" {nb:>9} {(100.0 * nb / bank_total if bank_total else 0):>6.1f}%")
    if len(order) > top:
        L.append(f"  ... and {len(order) - top} more hydrophones")

    # Per-class totals for the pool as a whole, before and after capping. The
    # per-hydrophone rows above cannot show this: the pool's class balance is a
    # property of the mixture, and it is the number that says what the cap actually
    # bought. A cap chosen to fix CHANNEL concentration also moves CLASS composition,
    # because the sites it truncates are not a random sample of the classes.
    L.append("")
    L.append("  class composition of the pool")
    L.append(f"    {'class':<{cw}} {'before cap':>11} {'share':>7} "
             f"{'after cap':>11} {'share':>7}")
    L.append("    " + "-" * (cw + 39))
    raw_cls, pool_cls = collections.Counter(), collections.Counter()
    for row in raw.group_by("Labels").len().iter_rows(named=True):
        raw_cls[row["Labels"] if row["Labels"] in labels else "other"] += row["len"]
    for counter in by_h.values():
        pool_cls.update(counter)
    for c in cols:
        b, a = raw_cls.get(c, 0), pool_cls.get(c, 0)
        L.append(f"    {c:<{cw}} {b:>11} {100.0 * b / max(raw.height, 1):>6.1f}% "
                 f"{a:>11} {100.0 * a / max(pool.height, 1):>6.1f}%")
    L.append(f"    {'TOTAL':<{cw}} {raw.height:>11} {100.0:>6.1f}% "
             f"{pool.height:>11} {100.0:>6.1f}%")
    L.append("    ('other' is labels outside the four ecotype classes — NRKW, OKW, "
             "KW_und and so on.")
    L.append("     They are in the pool because the SSL objective is label-agnostic, "
             "and they are")
    L.append("     targets for nothing, since no probe scores them.)")

    L.append("")
    L.append("=== channel concentration, and what a cap would do ===")
    L.append("  'effective channels' is the inverse Simpson index: 1 / sum of squared")
    L.append("  shares. It answers how many EQUALLY SIZED hydrophones would give the")
    L.append("  same probability that two clips drawn at random come from the same")
    L.append("  site. A raw count of hydrophones cannot answer that, because a site")
    L.append("  contributing nine clips counts as a full unit there and gives the")
    L.append("  student no real exposure to that channel.")
    L.append("")
    # Computed on the UNCAPPED distributions. Running these on the already-capped pool
    # would report what a second cap on top of the first would do, which is not a
    # question anyone is asking.
    raw_totals = {r["Dataset"]: r["len"]
                  for r in raw.group_by("Dataset").len().iter_rows(named=True)}
    raw_bank = {r["Dataset"]: r["len"]
                for r in raw.filter(pl.col("Labels") == "Background")
                            .group_by("Dataset").len().iter_rows(named=True)}
    L.append(_cap_table("adaptation pool (uncapped basis)", raw_totals,
                        [None, 40000, 20000, 10000, 5000]))
    L.append("")
    L.append(_cap_table("background bank (uncapped basis)", raw_bank,
                        [None, 2000, 1000, 500]))
    L.append("")
    L.append(f"  configured now: pool cap {cap_pool or 'none'}, "
             f"bank cap {cap_bank or 'none'}")
    L.append("")
    L.append("  The bank matters more than its size suggests. It is what the phrase")
    L.append("  'cross-hydrophone noise' refers to, so it is what cell C5 "
             "(background.p=0.0)")
    L.append("  is the control FOR. If two sites supply most of it, C4 against C5 tests")
    L.append("  whether adding those two sites' noise helps, not whether "
             "cross-hydrophone")
    L.append("  noise helps, and that is a weaker claim than the study is set up to "
             "make.")
    return "\n".join(L)


def _inverse_simpson(counts):
    """Effective number of classes: 1 / sum of squared shares.

    Equals n exactly when n classes are equally sized, which is what makes it readable
    as a count rather than as an abstract index. A linear function of the shares cannot
    do this job, because the shares sum to one by construction and carry no information
    about the shape of the distribution.
    """
    total = sum(counts)
    if total <= 0:
        return 0.0
    return 1.0 / sum((c / total) ** 2 for c in counts if c > 0)


def _cap_table(name, counts, caps):
    """What each candidate per-site ceiling would do to size and concentration.

    A ceiling only ever removes clips from sites that are above it, so it cannot
    manufacture diversity that is absent: the achievable maximum is bounded by how
    many sites contribute a meaningful number in the first place.
    """
    rows = [f"  {name}"]
    rows.append(f"    {'per-site cap':>12} {'total':>9} {'largest':>9} "
                f"{'effective channels':>20}")
    rows.append("    " + "-" * 52)
    for cap in caps:
        vals = [min(c, cap) if cap else c for c in counts.values()]
        tot = sum(vals)
        rows.append(f"    {('none' if cap is None else str(cap)):>12} {tot:>9} "
                    f"{(100.0 * max(vals) / tot if tot else 0):>8.1f}% "
                    f"{_inverse_simpson(vals):>20.1f}")
    return "\n".join(rows)


@hydra.main(version_base=None, config_path=str(root / "configs"),
            config_name="probe.yaml")
def run(cfg: DictConfig):
    dm = SelfDistillDataModule(
        dataset_configs=cfg.data.dataset,
        loader_configs=cfg.data.loaders,
        transform_configs=cfg.data.transform,
    )
    # Same gate the probe uses: this is what filters df down to clips that actually
    # landed on disk. Reporting the split off the un-materialised manifest would
    # overcount by however many clips failed to fetch (~7.3k in the current stage).
    dm.prepare_data()

    df, tags = probe_pool(dm.df, cfg.data.dataset, include_low_sr=cfg.include_low_sr)

    print(f"\ndataset config : {cfg.data.dataset.name}")
    print(f"manifest       : {cfg.paths.dataset_dir}/{cfg.data.dataset.manifest_name}")
    print(f"clip_duration  : {cfg.data.dataset.clip_duration} s")
    # The frozen cells are NOT bandwidth-matched: this interpolates
    # module.network.sampling_rate, so it is 16 kHz under BEATs and 32 kHz under
    # Bird-MAE. Print it next to the split so the two are read together.
    print(f"probe sample_rate : {int(cfg.data.transform.input.sample_rate)} Hz "
          f"(follows module.network.sampling_rate)")
    print(f"include_low_sr : {bool(cfg.include_low_sr)}")
    print(f"c_selection    : {cfg.c_selection}\n")
    print(split_report(df, tags, cfg.data.dataset, source_df=dm.df))
    print()
    print(ssl_pool_report(dm))


if __name__ == "__main__":
    run()
