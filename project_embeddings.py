"""Project probe features to 2-D and colour them by hydrophone, ecotype and call type.

    python project_embeddings.py --features mlruns_nibi/features/reported
    python project_embeddings.py --features <dir> --method tsne --split train

WHAT THIS IS FOR. The nuisance metric answers "is the recording site decodable" with a
number, on the TRAIN hydrophones, because that is where it always ran. This answers a
different question that the number cannot: does channel structure persist at the three
sites the encoder never adapted on, and does whatever structure exists line up with the
biology instead. A projection is not evidence on its own, and it must not be presented
as if it were, but it is the only view that shows WHERE the structure sits.

ONE FIT PER CELL, AND THE AXES ARE NOT SHARED. Every cell is a different embedding
space, and two of them are different backbones entirely, so co-embedding them would be
meaningless. Each cell gets its own projection and the coordinates are comparable only
within a figure, never between figures. What IS comparable between figures is the
structure: whether the sites separate, and how cleanly.

ONE PROJECTION PER CELL, COLOURED THREE WAYS. The three panels of a figure are the same
2-D coordinates with three different colourings, never three separate fits. Re-fitting
per panel would make "the call types sit inside the hydrophone clusters" unreadable,
because the clusters would move between panels.

WHY CALL TYPE IS NOT A THIRD COLOURED SCATTER. There are 21 call types. No categorical
palette is readable at 21 hues, and generating them is the standard way to produce a
figure that looks informative and is not. The main figure shows the three commonest call
types in the split with everything else in grey, and `--calltype-grid` writes the honest
full version as small multiples, one panel per call type against a grey background.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects

# --- palette ---------------------------------------------------------------
# The first three categorical slots of the reference palette, which are the three that
# clear the all-pairs CVD and normal-vision floors that a scatter needs (adjacent-pair
# validation is for stacks and lines, where only neighbours touch; in a scatter every
# pair is adjacent). Slot 4 onward fails those floors, which is the real reason the
# tail folds into grey rather than getting a colour of its own.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]     # blue, orange, aqua
GREY_MARK = "#c3c2b7"                          # the folded tail and unlabelled rows
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
# Aqua measures 2.74:1 against this surface, below the 3:1 bar, so the palette's relief
# rule applies: every class carries a visible direct label at its centroid AND a legend
# entry with its count. Identity is never left to colour alone here.
MAX_SLOTS = len(SERIES)


def load_cells(feature_dir):
    """Read every .npz the probe wrote, newest-first by cell name."""
    files = sorted(Path(feature_dir).glob("*.npz"))
    if not files:
        sys.exit(f"no .npz files in {feature_dir} — was the probe run with "
                 f"save_features set?")
    cells = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        cells.append({
            "file": f,
            "cell": str(d["cell"]), "layer": str(d["layer"]),
            "encoder": str(d["encoder"]), "source": str(d["source"]),
            "seal_test": bool(d["seal_test"]),
            "X": d["X"].astype(np.float32),        # stored float16; see probe.yaml
            "label": d["label"], "call": d["call"],
            "hydrophone": d["hydrophone"], "tags": d["tags"],
            "label_names": [str(s) for s in d["label_names"]],
            "call_names": [str(s) for s in d["call_names"]],
        })
    return cells


def project(X, method, seed, n_neighbors=30, min_dist=0.1):
    """2-D embedding of X. Cosine metric, because these are pooled transformer
    features whose norm carries loudness rather than content, and a Euclidean metric
    would let a loud clip and a quiet clip of the same call sit far apart."""
    if method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                            min_dist=min_dist, metric="cosine", random_state=seed)
        return reducer.fit_transform(X)
    from sklearn.manifold import TSNE
    # init="pca" rather than "random": a random init makes the global arrangement a
    # property of the seed, and the global arrangement is exactly what gets read off
    # a figure like this.
    return TSNE(n_components=2, perplexity=30, init="pca", metric="cosine",
                random_state=seed).fit_transform(X)


def fold(values, keep=None):
    """Split a label array into at most MAX_SLOTS named classes plus a grey tail.

    Returns (names, index_array) where index_array is the slot per row and -1 marks
    the tail. `keep` fixes the order when the caller knows it; otherwise the commonest
    classes win, which is what makes the folded tail small rather than arbitrary.
    """
    if keep is None:
        uniq, counts = np.unique(values, return_counts=True)
        keep = [u for u, _ in sorted(zip(uniq, counts), key=lambda p: -p[1])]
    keep = [k for k in keep if (values == k).sum() > 0][:MAX_SLOTS]
    idx = np.full(len(values), -1, dtype=int)
    for i, k in enumerate(keep):
        idx[values == k] = i
    return keep, idx


def mark_size(n):
    """Point area, scaled to the count. A fixed size is wrong at both ends: at 500
    clips it renders as dust, and at 20,000 it is a solid mass with no visible
    density structure, which is the one thing a projection is for."""
    return float(np.clip(20000.0 / max(n, 1), 3.0, 16.0))


def data_aspect(xy):
    """Height-over-width of the plotted cloud, used to size the figure."""
    w, h = np.ptp(xy[:, 0]), np.ptp(xy[:, 1])
    return float(h / w) if w > 0 else 1.0


def dense_point(xy, mask, bins=28):
    """The centre of the bin holding most of this class, rather than its centroid.

    A centroid — mean or median — is only a sensible label position for a class that
    forms one blob. These classes routinely do not: an ecotype recorded at three sites
    can land in three clusters, and then both the mean and the median sit in the empty
    space between them, labelling nothing. The modal bin is always somewhere the class
    actually is.
    """
    pts = xy[mask]
    h, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins)
    i, j = np.unravel_index(np.argmax(h), h.shape)
    return (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2


def place_labels(ax, xy, names, idx, min_n=5):
    """Direct labels at each class's densest region, nudged apart when they collide.

    Two labels rendered on top of each other read as one wrong word — "S01" over "S02"
    came out as "SS01" — so a candidate too close to one already placed is pushed down
    until it clears. Distances are measured as a fraction of the plotted extent, which
    is the only scale available when the axes carry no units.
    """
    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-9)
    placed = []
    for i, name in enumerate(names):
        m = idx == i
        if m.sum() < min_n:
            continue
        x, y = dense_point(xy, m)
        for _ in range(6):
            if all(abs(x - px) > 0.10 * span or abs(y - py) > 0.045 * span
                   for px, py in placed):
                break
            y -= 0.05 * span
        placed.append((x, y))
        t = ax.text(x, y, name, fontsize=8.5, color=INK, ha="center", va="center",
                    zorder=5)
        t.set_path_effects([patheffects.withStroke(linewidth=3, foreground=SURFACE)])


def panel(ax, xy, names, idx, title, tail_label, rng):
    """One scatter: the grey tail underneath, the named classes above it in one
    shuffled draw so that no class is systematically painted on top of another."""
    ax.set_facecolor(SURFACE)
    s = mark_size(len(xy))
    tail = idx < 0
    if tail.any():
        ax.scatter(xy[tail, 0], xy[tail, 1], s=s * 0.7, c=GREY_MARK, alpha=0.35,
                   linewidths=0, rasterized=True)

    named = np.flatnonzero(~tail)
    rng.shuffle(named)
    ax.scatter(xy[named, 0], xy[named, 1], s=s,
               c=[SERIES[i] for i in idx[named]], alpha=0.75,
               linewidths=0, rasterized=True)

    # Direct labels, in ink with a surface-coloured halo, one per class. Required
    # rather than optional: aqua is below 3:1 against this surface, so the palette's
    # relief rule says identity cannot rest on colour alone.
    place_labels(ax, xy, names, idx)

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                          markerfacecolor=SERIES[i], markeredgecolor="none",
                          label=f"{n}  (n={(idx == i).sum():,})")
               for i, n in enumerate(names)]
    if tail.any():
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                                  markerfacecolor=GREY_MARK, markeredgecolor="none",
                                  label=f"{tail_label}  (n={tail.sum():,})"))
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.02),
                    frameon=False, fontsize=8, labelcolor=INK_2,
                    handletextpad=0.4, borderpad=0)
    leg.set_title(None)

    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
    strip_axes(ax, xy)


def strip_axes(ax, xy=None):
    """UMAP and t-SNE coordinates have no units, so ticks and gridlines would invite a
    reading the data does not support. The frame goes too; only the marks remain.

    The aspect is equal and adjusted by BOX, not by datalim. With `datalim` matplotlib
    keeps the axes rectangle and widens the data range to match it, which on a wide
    subplot pulls the cloud apart horizontally and shrinks every cluster to a speck;
    with `box` the data keeps its shape and the rectangle shrinks instead.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if xy is not None and len(xy):
        pad = 0.04 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-6)
        ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box")


def figure_for_cell(c, xy, split, method, out_dir, rng):
    hydro, label, call = c["hydrophone"], c["label"], c["call"]

    # Ecotype: Background is drawn as the grey tail rather than taking a colour slot.
    # That is not a workaround for the three-slot cap, it is what the class means —
    # Background is the absence of a call, so it is the ground the other three sit on.
    eco_names = [n for n in c["label_names"] if n != "Background"]
    eco_vals = np.array([c["label_names"][i] if 0 <= i < len(c["label_names"]) else "?"
                         for i in label])
    eco_keep, eco_idx = fold(eco_vals, keep=eco_names)

    hyd_keep, hyd_idx = fold(hydro)

    # Most clips carry no call-type label at all, so the three commonest types are
    # ranked over the LABELLED rows only. Ranking over everything would spend a colour
    # slot on "unlabelled", which is the grey tail by definition.
    call_vals = np.array([c["call_names"][i] if 0 <= i < len(c["call_names"]) else ""
                          for i in call])
    labelled = call_vals != ""
    call_keep, _ = fold(call_vals[labelled])
    _, call_idx = fold(call_vals, keep=call_keep)

    # Size the panels from the projection's own shape. The aspect is locked equal and
    # adjusted by box, so matplotlib shrinks each axes to fit its data inside whatever
    # slot it was given; a slot of the wrong shape therefore turns into a band of empty
    # paper rather than a bigger picture.
    pw = 4.7
    ph = float(np.clip(pw * data_aspect(xy), 2.6, 6.5))
    fig, axes = plt.subplots(1, 3, figsize=(3 * pw, ph + 2.1), facecolor=SURFACE)
    panel(axes[0], xy, hyd_keep, hyd_idx, "by hydrophone", "other sites", rng)
    panel(axes[1], xy, eco_keep, eco_idx, "by ecotype", "Background", rng)
    panel(axes[2], xy, call_keep, call_idx,
          "by call type (three commonest)", "other or unlabelled", rng)

    seal = "sealed" if c["seal_test"] else "reported"
    fig.suptitle(
        f"{c['cell']}  ·  {c['encoder']}, {c['source']}  ·  {split} split, "
        f"{len(xy):,} clips  ·  {method.upper()}, cosine metric  ·  {seal} protocol",
        fontsize=11, color=INK, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.015,
             "Axes carry no units; distances are comparable within a panel and not "
             "between figures. Same projection in all three panels.",
             fontsize=8, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.20, wspace=0.06)

    out = Path(out_dir) / f"{c['cell']}_{method}_{split}.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return out


def calltype_grid(c, xy, split, method, out_dir, min_n=15):
    """The honest version of the call-type panel: one small multiple per call type.

    Small multiples rather than 21 hues, because 21 categorical colours cannot be told
    apart by anyone and the figure would imply a precision it does not have.
    """
    call_vals = np.array([c["call_names"][i] if 0 <= i < len(c["call_names"]) else ""
                          for i in c["call"]])
    present = [(v, int((call_vals == v).sum()))
               for v in sorted(set(call_vals) - {""})]
    present = [(v, n) for v, n in present if n >= min_n]
    present.sort(key=lambda p: -p[1])
    if not present:
        return None

    ncol = min(6, len(present))
    nrow = int(np.ceil(len(present) / ncol))
    # Same reasoning as the main figure: each cell's slot has to match the shape of the
    # cloud, or the equal aspect turns the surplus into empty paper between rows.
    cw = 2.3
    ch = float(np.clip(cw * data_aspect(xy), 1.4, 3.2))
    fig, axes = plt.subplots(nrow, ncol, figsize=(cw * ncol, (ch + 0.35) * nrow + 0.8),
                             facecolor=SURFACE, squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, (name, n) in zip(axes.ravel(), present):
        ax.set_visible(True)
        ax.set_facecolor(SURFACE)
        ax.scatter(xy[:, 0], xy[:, 1], s=2, c=GREY_MARK, alpha=0.3,
                   linewidths=0, rasterized=True)
        m = call_vals == name
        ax.scatter(xy[m, 0], xy[m, 1], s=6, c=SERIES[0], alpha=0.85,
                   linewidths=0, rasterized=True)
        ax.set_title(f"{name}  (n={n:,})", fontsize=8.5, color=INK, loc="left", pad=4)
        strip_axes(ax, xy)

    dropped = len(set(call_vals) - {""}) - len(present)
    note = (f"Each panel is the same {method.upper()} projection; the named call type "
            f"is in blue and every other clip is grey.")
    if dropped > 0:
        note += f" {dropped} call types with fewer than {min_n} clips are not shown."
    fig.suptitle(f"{c['cell']}  ·  {c['encoder']}  ·  call types in the {split} split",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, note, fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    out = Path(out_dir) / f"{c['cell']}_{method}_{split}_calltypes.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True,
                   help="directory of .npz files written by probe_selfdistill.py")
    p.add_argument("--out", default=None, help="where to write the PNGs "
                                               "(default: <features>/figures)")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--method", default="umap", choices=["umap", "tsne"])
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--max-points", type=int, default=20000,
                   help="subsample above this many clips, for run time")
    p.add_argument("--calltype-grid", action="store_true",
                   help="also write the per-call-type small multiples")
    p.add_argument("--cells", default=None,
                   help="comma-separated cell names; default is every file found")
    args = p.parse_args()

    out_dir = Path(args.out or Path(args.features) / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_cells(args.features)
    if args.cells:
        want = {s.strip() for s in args.cells.split(",")}
        cells = [c for c in cells if c["cell"] in want]
        if not cells:
            sys.exit(f"none of {sorted(want)} found in {args.features}")

    for c in cells:
        rng = np.random.default_rng(args.seed)
        m = np.ones(len(c["tags"]), dtype=bool) if args.split == "all" \
            else (c["tags"] == args.split)
        if m.sum() < 50:
            print(f"{c['cell']}: only {m.sum()} clips in the {args.split} split, "
                  f"skipping")
            continue
        sel = np.flatnonzero(m)
        if len(sel) > args.max_points:
            sel = rng.choice(sel, args.max_points, replace=False)
            print(f"{c['cell']}: subsampled to {args.max_points:,} clips")

        sub = {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(m) else v)
               for k, v in c.items()}
        print(f"{c['cell']}: projecting {len(sel):,} clips "
              f"({sub['X'].shape[1]}-d) with {args.method} ...", flush=True)
        xy = project(sub["X"], args.method, args.seed)

        out = figure_for_cell(sub, xy, args.split, args.method, out_dir, rng)
        print(f"  wrote {out}")
        if args.calltype_grid:
            g = calltype_grid(sub, xy, args.split, args.method, out_dir)
            print(f"  wrote {g}" if g else "  no call types with enough clips")


if __name__ == "__main__":
    main()
