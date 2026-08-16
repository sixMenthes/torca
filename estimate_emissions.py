#!/usr/bin/env python3
"""Estimate the energy and carbon cost of this project when the GPU term is unmeasurable.

    python3 estimate_emissions.py --mlruns mlruns_nibi
    python3 estimate_emissions.py --mlruns mlruns_nibi --per-epoch 125
    python3 estimate_emissions.py --mlruns mlruns_nibi --plan "birdmae_bg07:2:36"

WHY THIS SCRIPT EXISTS
----------------------
codecarbon reads GPU energy through NVML, and NVML accounts energy per DEVICE rather
than per MIG instance. Every job in this study ran on an `h100_3g.40gb` slice, so the
tracker had nothing to read and wrote `emissions/gpu_energy_kwh = 0.0` on all 38 runs.
The CPU and RAM figures in the same records ARE real measurements. The consequence is
that the logged `emissions/co2eq_kg` omits the single largest term in a training job,
so it is not a small underestimate but a structural one.

This script keeps the measured CPU and RAM energy, models the missing GPU term from the
slice fraction and the device TDP, and reports the total. Nothing here is a measurement
of GPU power. It is an accounting model with stated constants, and every constant is a
flag so the assumption can be moved and the answer re-read.

A SECOND, INDEPENDENT ERROR IN THE STORED NUMBERS
-------------------------------------------------
emissions.py builds an OfflineEmissionsTracker with `country_iso_code="CAN"` and leaves
`region` unset, and its own docstring warns that the Canadian national average is
dominated by Alberta and Saskatchewan fossil generation. The stored `co2eq_kg` therefore
uses a national-average intensity for a job that ran in Ontario. That error pushes the
other way from the missing GPU, so the two do not cancel in any principled amount and
the stored figure should not be quoted at all. This script recomputes from energy.

WHAT "FROM ONE EPOCH" MEANS HERE
--------------------------------
Wall-clock time is the only quantity that has to be measured, because every energy term
in the model is a rate multiplied by time. So one epoch of a new arm is enough to price
the whole campaign: measure seconds per epoch once, multiply by the planned epochs, and
apply the same rates. `--per-epoch` takes that measured number directly. With no
`--per-epoch`, the script calibrates from the completed runs already in the store.

HOW TO READ THE OUTPUT
----------------------
Two constants are genuinely uncertain: the average GPU utilisation while a job holds the
slice, and the grid carbon intensity. The script prints a low/central/high band over
both rather than a single number, because a single number here would be false precision.
Report the band.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# --- the constants, all overridable on the command line --------------------
# SLICE FRACTIONS. Nibi bills GPU time in RGU (Resource Group Units), and the RGU table
# in selfdistill.sh is the cluster's own statement of what a slice is worth relative to
# a whole card. Using the billing fraction rather than a hardware count is deliberate:
# it is the number the allocation is actually charged, and it is the one an auditor can
# check against the accounting records. The alternatives disagree, and by enough to
# matter — NVIDIA partitions an H100 into 7 compute instances, so a 3g slice is 3/7 by
# GPC count, while the comment at the top of selfdistill.sh calls the same slice 3/8.
# The RGU ratio for 3g.40gb is 6.1/12.2, which is exactly one half.
SLICE_RGU = {
    "h100": 12.2,
    "h100_80gb": 12.2,
    "h100_3g.40gb": 6.1,
    "h100_2g.20gb": 3.48,
    "h100_1g.10gb": 1.74,
}
FULL_RGU = 12.2

# H100 SXM5 board power. The PCIe card is 350 W; pass --tdp 350 if Nibi's nodes are PCIe.
DEFAULT_TDP_W = 700.0

# Average fraction of the slice's power envelope actually drawn while a job holds it.
# This is the weakest constant in the model and it is not measured anywhere in the study.
# The header of selfdistill.sh explicitly flags a suspected dataloader bottleneck, which
# would put real utilisation well below 1.0, so the central value is deliberately mid.
UTIL_LOW, UTIL_MID, UTIL_HIGH = 0.30, 0.50, 0.80

# Grid carbon intensity in gCO2e per kWh. Nibi is hosted in Ontario, whose grid is mostly
# nuclear and hydro. The low/high pair spans the usual range of published Ontario figures,
# which differ mainly over whether they count generation only or include lifecycle terms.
# CONFIRM THIS before it goes in a paper; it is an assumption, not a lookup.
INTENSITY_LOW, INTENSITY_MID, INTENSITY_HIGH = 25.0, 40.0, 90.0

# Power usage effectiveness: the multiplier for cooling and distribution overhead in the
# building. 1.1 to 1.2 is typical of a modern facility; 1.15 is a neutral placeholder.
DEFAULT_PUE = 1.15


# --- reading the run store -------------------------------------------------
def read_meta(meta: Path) -> dict:
    """meta.yaml here is flat key: value, so a full YAML parser is not needed."""
    out = {}
    for line in meta.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def last_metric(run_dir: Path, key: str):
    f = run_dir / "metrics" / key
    if not f.is_file():
        return None
    lines = [l for l in f.read_text().splitlines() if l.strip()]
    if not lines:
        return None
    try:
        return float(lines[-1].split()[1])
    except (IndexError, ValueError):
        return None


def read_param(run_dir: Path, key: str):
    f = run_dir / "params" / key
    return f.read_text().strip() if f.is_file() else None


def collect(mlruns: Path) -> list:
    """One record per distinct run_name. Duplicated experiment dirs are collapsed."""
    seen = {}
    for meta_file in mlruns.rglob("meta.yaml"):
        run_dir = meta_file.parent
        if not (run_dir / "metrics").is_dir() and not (run_dir / "params").is_dir():
            continue
        meta = read_meta(meta_file)
        name = meta.get("run_name")
        if not name or name in seen:
            continue

        # Wall clock, in order of preference. emissions/duration_s is what the tracker
        # actually timed; meta start/end brackets the whole process including staging,
        # and is the only source for the probe runs, whose tracker wrote nothing.
        dur = last_metric(run_dir, "emissions/duration_s")
        source = "tracker"
        if dur is None:
            try:
                st, en = int(meta.get("start_time", 0)), int(meta.get("end_time", 0))
                if st and en and en > st:
                    dur, source = (en - st) / 1000.0, "meta"
            except ValueError:
                pass
        if not dur:
            continue

        epochs = read_param(run_dir, "trainer/max_epochs")
        try:
            epochs = int(epochs) if epochs is not None else None
        except ValueError:
            epochs = None

        seen[name] = {
            "name": name,
            "dir": run_dir,
            "duration_s": dur,
            "duration_source": source,
            "epochs": epochs,
            "kind": "probe" if "probe" in name else "train",
            "cpu_kwh": last_metric(run_dir, "emissions/cpu_energy_kwh") or 0.0,
            "ram_kwh": last_metric(run_dir, "emissions/ram_energy_kwh") or 0.0,
            "gpu_kwh_logged": last_metric(run_dir, "emissions/gpu_energy_kwh"),
            "co2_logged": last_metric(run_dir, "emissions/co2eq_kg"),
        }
    return sorted(seen.values(), key=lambda r: r["name"])


# --- the model -------------------------------------------------------------
def gpu_kwh(hours: float, tdp_w: float, frac: float, util: float) -> float:
    """Attributed GPU energy: a slice-sized share of the board, scaled by utilisation."""
    return tdp_w * frac * util * hours / 1000.0


def price(rec, tdp_w, frac, util, pue, intensity_g):
    """Return (kWh at the wall, kg CO2e) for one run under one set of constants."""
    hours = rec["duration_s"] / 3600.0
    g = gpu_kwh(hours, tdp_w, frac, util)
    # CPU and RAM come from the tracker and are real. Probe runs have neither, so their
    # non-GPU load is invisible here and their totals are GPU-only and therefore low.
    it_kwh = g + rec["cpu_kwh"] + rec["ram_kwh"]
    wall_kwh = it_kwh * pue
    return wall_kwh, wall_kwh * intensity_g / 1000.0


def band(records, tdp_w, frac, pue):
    """Total kWh and kg CO2e across three (utilisation, intensity) corners."""
    out = {}
    for tag, util, inten in (
        ("low", UTIL_LOW, INTENSITY_LOW),
        ("central", UTIL_MID, INTENSITY_MID),
        ("high", UTIL_HIGH, INTENSITY_HIGH),
    ):
        kwh = co2 = 0.0
        for r in records:
            k, c = price(r, tdp_w, frac, util, pue, inten)
            kwh += k
            co2 += c
        out[tag] = (kwh, co2, util, inten)
    return out


def parse_plan(spec):
    """`name:runs:epochs[,name:runs:epochs...]` -> list of (name, n_runs, n_epochs)."""
    plan = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            sys.exit(f"bad --plan entry {item!r}; expected name:runs:epochs")
        plan.append((parts[0], int(parts[1]), int(parts[2])))
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlruns", default="mlruns_nibi", type=Path,
                    help="run store to account for (default: mlruns_nibi)")
    ap.add_argument("--slice", default="h100_3g.40gb", choices=sorted(SLICE_RGU),
                    help="MIG instance the jobs ran on (default: h100_3g.40gb)")
    ap.add_argument("--tdp", type=float, default=DEFAULT_TDP_W,
                    help=f"full-device board power in W (default: {DEFAULT_TDP_W:.0f}, "
                         f"SXM; pass 350 for PCIe)")
    ap.add_argument("--pue", type=float, default=DEFAULT_PUE,
                    help=f"datacentre PUE (default: {DEFAULT_PUE})")
    ap.add_argument("--per-epoch", type=float, default=None,
                    help="MEASURED seconds per epoch, e.g. from a one-epoch calibration "
                         "run. Overrides the value calibrated from the store.")
    ap.add_argument("--plan", default=None,
                    help="planned future work as name:runs:epochs[,...], e.g. "
                         "'birdmae_bg07:2:36'")
    ap.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = ap.parse_args()

    if not args.mlruns.is_dir():
        sys.exit(f"no such run store: {args.mlruns}")

    frac = SLICE_RGU[args.slice] / FULL_RGU
    records = collect(args.mlruns)
    if not records:
        sys.exit(f"no runs with a usable duration found under {args.mlruns}")

    trains = [r for r in records if r["kind"] == "train"]
    probes = [r for r in records if r["kind"] == "probe"]

    # --- calibration: seconds per epoch ------------------------------------
    calib = [r for r in trains if r["epochs"]]
    if args.per_epoch is not None:
        sec_per_epoch, calib_from = args.per_epoch, "measured, passed with --per-epoch"
    elif calib:
        rates = sorted(r["duration_s"] / r["epochs"] for r in calib)
        sec_per_epoch = rates[len(rates) // 2]
        calib_from = f"median of {len(calib)} completed training runs in the store"
    else:
        sec_per_epoch, calib_from = None, "unavailable"

    print("=" * 78)
    print("ENERGY AND CARBON ACCOUNTING — MODELLED GPU TERM, MEASURED CPU AND RAM")
    print("=" * 78)
    print(f"run store          : {args.mlruns}")
    print(f"GPU instance       : {args.slice}  "
          f"({SLICE_RGU[args.slice]}/{FULL_RGU} RGU = {frac:.3f} of a board)")
    print(f"board power        : {args.tdp:.0f} W")
    print(f"PUE                : {args.pue}")
    print(f"runs accounted     : {len(records)}  "
          f"({len(trains)} training, {len(probes)} probe)")

    zeroed = sum(1 for r in trains if (r["gpu_kwh_logged"] or 0.0) == 0.0)
    print(f"\nMIG hole           : {zeroed} of {len(trains)} training runs logged "
          f"gpu_energy_kwh = 0.0")
    print("                     That is the term this script replaces with a model.")

    # --- per-epoch calibration ---------------------------------------------
    print("\n" + "-" * 78)
    print("CALIBRATION: seconds per epoch")
    print("-" * 78)
    if sec_per_epoch is None:
        print("  No epoch count available. Pass --per-epoch from a calibration run.")
    else:
        print(f"  seconds per epoch  : {sec_per_epoch:,.1f}   ({calib_from})")
        kwh_ep, co2_ep = price({"duration_s": sec_per_epoch, "cpu_kwh": 0.0,
                                "ram_kwh": 0.0},
                               args.tdp, frac, UTIL_MID, args.pue, INTENSITY_MID)
        print(f"  GPU-only cost of ONE epoch, central constants: "
              f"{kwh_ep:.4f} kWh, {co2_ep * 1000:.2f} g CO2e")
        if calib and args.per_epoch is None:
            rates = sorted(r["duration_s"] / r["epochs"] for r in calib)
            med = rates[len(rates) // 2]
            print(f"  spread across runs : {rates[0]:,.0f} to {rates[-1]:,.0f} s/epoch, "
                  f"which is {100 * (rates[0] - med) / med:+.0f}% to "
                  f"{100 * (rates[-1] - med) / med:+.0f}% around the median")
            print("  That spread mixes two backbones and two epoch counts, so it is an")
            print("  upper bound on the uncertainty for any ONE arm. Pass --per-epoch")
            print("  from a calibration run of the arm you are pricing to remove it.")

    # --- retrospective total ------------------------------------------------
    print("\n" + "-" * 78)
    print("THE PROJECT SO FAR")
    print("-" * 78)
    gpu_h = sum(r["duration_s"] for r in records) / 3600.0
    print(f"  wall-clock GPU-hours held : {gpu_h:,.1f} h "
          f"({gpu_h * frac:,.1f} board-equivalent hours)")
    measured = sum(r["cpu_kwh"] + r["ram_kwh"] for r in records)
    print(f"  measured CPU + RAM energy : {measured:,.3f} kWh (from the tracker)")

    b = band(records, args.tdp, frac, args.pue)
    print(f"\n  {'scenario':<10}{'util':>7}{'gCO2e/kWh':>12}{'kWh':>12}{'kg CO2e':>12}")
    for tag in ("low", "central", "high"):
        kwh, co2, util, inten = b[tag]
        print(f"  {tag:<10}{util:>7.2f}{inten:>12.0f}{kwh:>12.2f}{co2:>12.3f}")

    logged = sum(r["co2_logged"] or 0.0 for r in records)
    central = b["central"][1]
    print(f"\n  for comparison, the sum of the STORED co2eq_kg is {logged:.3f} kg.")
    print("  Do not quote it. It omits the GPU entirely and it applies a Canadian")
    print("  national-average intensity to a job that ran in Ontario, so it is wrong")
    print(f"  in two directions at once. The central estimate here is {central:.3f} kg.")

    # --- forward projection -------------------------------------------------
    if args.plan:
        if sec_per_epoch is None:
            sys.exit("--plan needs a per-epoch rate; pass --per-epoch")
        print("\n" + "-" * 78)
        print("PLANNED WORK")
        print("-" * 78)
        print(f"  {'cell':<22}{'runs':>6}{'epochs':>8}{'hours':>9}"
              f"{'kWh':>10}{'kg CO2e':>10}")
        tot_kwh = tot_co2 = tot_h = 0.0
        for name, n_runs, n_epochs in parse_plan(args.plan):
            secs = sec_per_epoch * n_epochs * n_runs
            kwh, co2 = price({"duration_s": secs, "cpu_kwh": 0.0, "ram_kwh": 0.0},
                             args.tdp, frac, UTIL_MID, args.pue, INTENSITY_MID)
            tot_kwh, tot_co2, tot_h = tot_kwh + kwh, tot_co2 + co2, tot_h + secs / 3600
            print(f"  {name:<22}{n_runs:>6}{n_epochs:>8}{secs / 3600:>9.2f}"
                  f"{kwh:>10.3f}{co2:>10.4f}")
        print(f"  {'TOTAL':<22}{'':>6}{'':>8}{tot_h:>9.2f}{tot_kwh:>10.3f}"
              f"{tot_co2:>10.4f}")
        print("\n  GPU term only: CPU and RAM are excluded because they are measured")
        print("  per run and there is nothing to measure on a run that has not happened.")
        print(f"  On the completed runs they added {measured / max(gpu_h, 1e-9):.4f} kWh")
        print("  per GPU-hour, so scale up by roughly that if you want a wall figure.")

    print("\n" + "=" * 78)
    print("WHAT IS MEASURED AND WHAT IS ASSUMED")
    print("=" * 78)
    print("  measured : wall-clock duration, CPU energy, RAM energy")
    print("  assumed  : GPU utilisation, grid carbon intensity, PUE, board TDP,")
    print("             and that a slice draws its RGU share of the board's power")
    print("  The last assumption is the load-bearing one. NVML cannot separate a MIG")
    print("  instance's draw from its neighbours', so it cannot be checked directly.")
    print("  Run with POWER_SAMPLE=1 to record whole-device draw, which bounds it above.")

    if args.json:
        print("\n" + json.dumps({
            "runs": len(records),
            "gpu_hours": gpu_h,
            "slice_fraction": frac,
            "seconds_per_epoch": sec_per_epoch,
            "measured_cpu_ram_kwh": measured,
            "scenarios": {k: {"kwh": v[0], "kg_co2e": v[1],
                              "util": v[2], "intensity_g_per_kwh": v[3]}
                          for k, v in b.items()},
        }, indent=2))


if __name__ == "__main__":
    main()
