"""Bar charts from the reported-protocol probe runs.

    python plot_reported.py                    # writes into figures/reported/
    python plot_reported.py --out somewhere/

Eight charts. Six are one per (task, backbone) and hold four bars: the adapted cell at
both seeds, the frozen backbone, and the MFCC floor. Two more compare the two frozen
backbones against the MFCC floor directly, for ecotype and for background.

The two backbones are never drawn on one axis in the adapted charts. They run at
different sample rates, 16 kHz for BEATs and 32 kHz for Bird-MAE, so they are not
bandwidth matched and the frozen-against-adapted comparison inside each arm is the
honest one. The two frozen-comparison charts do put them side by side, which is exactly
the cross-arm comparison that carries that caveat, so read them with it in mind.

Colour follows the entity rather than its position: blue is an adapted cell, orange is
a frozen backbone in every chart it appears in, and grey is the non-neural floor.

CHANCE FOR CALL TYPE IS COMPUTED, NOT LOGGED. The ecotype and hydrophone probes log
their own chance level; the call-type probe logs only its score. Balanced-accuracy
chance is one over the number of classes PRESENT in the scored rows, and the test split
carries 20 of the 21 configured call types, so it is 0.05 rather than 1/21.
"""

import argparse
import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/tundra/claude/torca/mlruns_nibi")
CUTOFF = dt.datetime(2026, 8, 10, 5, 0)          # the reported batch ran this morning

ADAPTED = "C0"      # matplotlib's default blue
FROZEN = "C1"       # matplotlib's default orange
FLOOR = "gray"

# metric key, short task slug, chance (None = read it off the run).
# The slug is used for BOTH the title and the filename, so a chart and the file it came
# from can never end up naming the task differently.
TASKS = [
    ("task_ecotype", "ecotype", None),
    ("nuisance_bg", "background", None),
    ("task_calltype", "calltype", 0.05),
]
BACKBONES = [("BEATs", "C1", "R_beats_s11", "R_beats_s17"),
             ("Bird-MAE", "C2", "R_birdmae_s11", "R_birdmae_s17")]


def load():
    """label -> {metric: value} for every reported-protocol run."""
    cells = {}
    for meta in ROOT.rglob("meta.yaml"):
        d = meta.parent
        name = start = None
        for line in meta.read_text().splitlines():
            if line.startswith("run_name:"):
                name = line.split(":", 1)[1].strip()
            if line.startswith("start_time:"):
                start = dt.datetime.fromtimestamp(
                    int(line.split(":", 1)[1].strip()) / 1000)
        if not name or start is None or start < CUTOFF or "probe_reported" not in name:
            continue
        m = d / "metrics" / "final"
        if not m.is_dir():
            continue
        vals = {}
        for f in m.iterdir():
            if f.is_file():
                lines = [l for l in f.read_text().splitlines() if l.strip()]
                if lines:
                    vals[f.name] = float(lines[-1].split()[1])
        cells.setdefault(name.split("_probe_reported_")[0], vals)
    return cells


def bars(names, vals, colors, chance, title, out):
    """One chart. Deliberately plain: matplotlib's defaults do the work."""
    fig, ax = plt.subplots()
    ax.bar(names, vals, color=colors, width=0.6)
    for x, v in enumerate(vals):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(chance, linestyle="--", linewidth=0.8, color="black")
    ax.set_title(title)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, max(max(vals), chance) * 1.15)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/home/tundra/claude/torca/figures/reported")
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load()
    need = ["C0", "C1", "C2", "R_beats_s11", "R_beats_s17",
            "R_birdmae_s11", "R_birdmae_s17"]
    absent = [c for c in need if c not in cells]
    if absent:
        raise SystemExit(f"missing cells: {', '.join(absent)}")

    for key, task, fixed_chance in TASKS:
        # A probe that logs its own chance level is believed over any constant.
        chance = fixed_chance
        if chance is None:
            chance = cells["C0"].get(key + "_chance") or cells["C0"]["nuisance_bg_chance"]

        for backbone, frozen, s11, s17 in BACKBONES:
            out = bars(
                ["seed 11", "seed 17", "frozen", "MFCC"],
                [cells[s11][key], cells[s17][key], cells[frozen][key], cells["C0"][key]],
                [ADAPTED, ADAPTED, FROZEN, FLOOR],
                chance, f"{backbone}_{task}",
                out_dir / f"{backbone.replace('-', '').lower()}_{task}.png")
            print(f"wrote {out}")

        # The two frozen backbones against the floor, with no adapted cell in the way.
        # Call type is left out: it was not asked for, and with 20 classes on 817 test
        # clips it is the noisiest of the three.
        if task in ("ecotype", "background"):
            out = bars(
                ["BEATs", "Bird-MAE", "MFCC"],
                [cells["C1"][key], cells["C2"][key], cells["C0"][key]],
                [FROZEN, FROZEN, FLOOR],
                chance, f"frozen_{task}",
                out_dir / f"frozen_{task}.png")
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
