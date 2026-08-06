#!/bin/bash
# ---------------------------------------------------------------------------
# One-time data prestaging for the self-distillation runs.
#
# RUN THIS ON A LOGIN NODE (or a DTN), NOT via sbatch:
#
#     bash slurm/selfdistill/prestage_selfdistill_data.sh verify    # read-only census
#     bash slurm/selfdistill/prestage_selfdistill_data.sh manifest  # rebuild + tar
#     bash slurm/selfdistill/prestage_selfdistill_data.sh full      # download + tar
#
# `verify` writes nothing and downloads nothing. Run it FIRST on any tree you did
# not stage in this session: the line to read is "clips on disk : n / total". A
# tree staged at another clip_duration reports ~0% there, because the window is
# baked into every filename — which is also the only cheap way to find out what
# duration an inherited tree actually holds.
#
# `manifest` is for clips that are already on disk: it rebuilds the cache manifest
# from a tree walk and packs the tarball, with no network. Use it after an
# interrupted download, or on a tree staged by an older version of this code —
# a manifest that predates per-clip matching lists SOURCE FILES, not clips, and
# renaming it to the current manifest_name does not convert it (prepare_data now
# rejects it by name rather than mismatching silently).
#
# Compute nodes have no internet, so the GPU jobs cannot pull from
# gs://noaa-passive-bioacoustic/... . This downloads the ~206k clips once and
# packs them into a single tar, so they sit on Lustre as ONE file rather than
# 206k small ones (inode quota + metadata server — see the Alliance "Handling
# large collections of files" guidance).
#
# Differs from finetune/prestage_torca_data.sh in two ways:
#   * clip_duration is 3.0, not 5.0. LocalPath encodes the window as
#     {start_ms}-{end_ms}.wav, so the two share ZERO filenames — a 5s prestage
#     is useless to a 3s run, silently (prepare_data skips downloading, every
#     row misses on disk, and you get empty batches with no error).
#   * it calls prestage_clips.py rather than dm.prepare_data(), so it also
#     writes the cached manifest and can be re-run with --verify.
#
# I/O- and CPU-heavy (13k source files -> 206k clips). Run inside tmux/screen.
# ---------------------------------------------------------------------------
set -euo pipefail

MODE="${1:-full}"

# ============================== USER SETTINGS ==============================
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/links/projects/def-XXXX/$USER/torca_root/torca}"
VENV="${VENV:-$HOME/torca_venv}"                 # needs gcsfs, soundfile, polars
STAGE="${STAGE:-$SCRATCH/selfdistill_stage}"     # loose files on scratch (1M inodes)
TARBALL="${TARBALL:-$HOME/links/projects/def-XXXX/$USER/torca_root/data/dclde_clips_3s.tar}"
CLIP_DURATION="${CLIP_DURATION:-3.0}"            # MUST match data.dataset.clip_duration
# MUST match data.dataset.manifest_name in BOTH dclde_selfdistill_*.yaml. Named per
# clip_duration so a 5s cache cannot masquerade as a 3s one.
MANIFEST_NAME="${MANIFEST_NAME:-DCLDE_3secs.parquet}"
WORKERS="${WORKERS:-32}"
# Delete the loose tree once the tarball is verified. DESTRUCTIVE, so off by default
# even though it is the point of tarring — prestage_clips.py refuses when the entry
# count is short, but nothing protects you from having pointed STAGE somewhere else.
# Set REMOVE_LOOSE=1 once the entry count above looks right.
REMOVE_LOOSE="${REMOVE_LOOSE:-0}"
# ==========================================================================

module load StdEnv/2023 python/3.11 gcc arrow/22.0.0
source "$VENV/bin/activate"
export PROJECT_ROOT HYDRA_FULL_ERROR=1

PARQUET="$PROJECT_ROOT/ds/DCLDE_w_Buzzes.parquet"
mkdir -p "$STAGE/data" "$(dirname "$TARBALL")"
cd "$PROJECT_ROOT"

echo "clip_duration = $CLIP_DURATION   (must equal data.dataset.clip_duration)"

# --- download, write the manifest, tar, and drop the loose tree ------------
# One command does all four. Idempotent: existing clips are skipped, so re-run
# freely after an interruption (which also retries any transient GCS failures).
#
# --tar is not optional in practice. 206k loose files burn inode quota and make
# every job start re-stat the tree through the Lustre metadata server; the tarball
# turns that into one sequential read onto node-local NVMe. The manifest is written
# BEFORE the tar, so it travels inside the archive and prepare_data finds it after
# extraction. Its absolute LocalPaths point at THIS node's $STAGE and are meaningless
# on the compute node, which is fine: prepare_data matches on clip_key(), the tail
# of the path (Provider/Dataset/stem/window.wav), which is the same wherever the
# tree is extracted. It matches per CLIP — a manifest listing only source files
# (anything written by the pre-2026-08 inline prepare_data) will NOT do.
#
# --remove-loose refuses if the tar entry count is short, so a truncated archive
# cannot silently become your only copy.
COMMON=(
  --parquet "$PARQUET"
  --dataset-dir "$STAGE/data"
  --clip-duration "$CLIP_DURATION"
  --manifest-name "$MANIFEST_NAME"
)
EXTRA=()
if [ "$REMOVE_LOOSE" = "1" ]; then EXTRA+=(--remove-loose); fi

# verify/manifest read an EXISTING tree, so an empty one means STAGE points somewhere
# wrong. mkdir -p above will happily have created it, and without this check `manifest`
# would write a manifest with zero rows and tar an empty directory — which then fails
# much later, inside a queued job, as "only 0 clips staged".
if [ "$MODE" != "full" ]; then
  if [ -z "$(find "$STAGE/data" -name '*.wav' -print -quit)" ]; then
    echo "ERROR: no .wav under $STAGE/data — set STAGE to the tree you staged" >&2
    exit 1
  fi
fi

case "$MODE" in
  verify)
    echo "MODE=verify — nothing is downloaded, nothing is written"
    python prestage_clips.py "${COMMON[@]}" --verify
    exit 0 ;;
  manifest)
    echo "MODE=manifest — rebuilding from the tree on disk, no network"
    python prestage_clips.py "${COMMON[@]}" --manifest-only \
        --tar "$TARBALL" "${EXTRA[@]}" ;;
  full)
    echo "MODE=full — download, manifest, tar"
    python prestage_clips.py "${COMMON[@]}" --workers "$WORKERS" \
        --tar "$TARBALL" "${EXTRA[@]}" ;;
  *)
    echo "ERROR: unknown mode '$MODE' (verify | manifest | full)" >&2; exit 1 ;;
esac

echo "Done. Point TARBALL in slurm/selfdistill/selfdistill.sh at: $TARBALL"
[ "$REMOVE_LOOSE" = "1" ] || echo "Loose tree kept at $STAGE/data — re-run with REMOVE_LOOSE=1 to drop it."
