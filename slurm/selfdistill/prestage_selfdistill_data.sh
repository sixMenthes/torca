#!/bin/bash
# ---------------------------------------------------------------------------
# One-time data prestaging for the self-distillation runs.
#
# RUN THIS ON A LOGIN NODE (or a DTN), NOT via sbatch:
#     bash slurm/selfdistill/prestage_selfdistill_data.sh
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

# ============================== USER SETTINGS ==============================
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/links/projects/def-XXXX/$USER/torca_root/torca}"
VENV="${VENV:-$HOME/torca_venv}"                 # needs gcsfs, soundfile, polars
STAGE="${STAGE:-$SCRATCH/selfdistill_stage}"     # loose files on scratch (1M inodes)
TARBALL="${TARBALL:-$HOME/links/projects/def-XXXX/$USER/torca_root/data/dclde_clips_3s.tar}"
CLIP_DURATION="${CLIP_DURATION:-3.0}"            # MUST match data.dataset.clip_duration
WORKERS="${WORKERS:-32}"
# ==========================================================================

module load StdEnv/2023 python/3.11 gcc arrow/22.0.0
source "$VENV/bin/activate"
export PROJECT_ROOT HYDRA_FULL_ERROR=1

PARQUET="$PROJECT_ROOT/ds/DCLDE_w_Buzzes.parquet"
mkdir -p "$STAGE/data" "$(dirname "$TARBALL")"
cd "$PROJECT_ROOT"

echo "clip_duration = $CLIP_DURATION   (must equal data.dataset.clip_duration)"

# --- download -------------------------------------------------------------
# Idempotent: existing clips are skipped, so re-run freely after an interruption.
python prestage_clips.py \
    --parquet "$PARQUET" \
    --dataset-dir "$STAGE/data" \
    --clip-duration "$CLIP_DURATION" \
    --workers "$WORKERS"

python prestage_clips.py --parquet "$PARQUET" --dataset-dir "$STAGE/data" \
    --clip-duration "$CLIP_DURATION" --verify

n=$(find "$STAGE/data" -name '*.wav' | wc -l)
echo "Downloaded $n wav clips."
[ "$n" -gt 0 ] || { echo "ERROR: no clips downloaded — check GCS access" >&2; exit 1; }

# --- pack -----------------------------------------------------------------
# The tar includes the DCLDE_no_balance manifest written inside data/. That is
# deliberate: prepare_data reads it to skip downloading, and it only uses the
# Soundfile column — the LocalPaths inside it are ignored, precisely because
# they point at this staging path rather than the compute node's $SLURM_TMPDIR.
echo "Packing tarball -> $TARBALL"
tar -cf "$TARBALL" -C "$STAGE" data
echo "Tarball size: $(du -h "$TARBALL" | cut -f1)"

echo "Removing loose staging files from scratch ..."
rm -rf "$STAGE/data"

echo "Done. Point TARBALL in slurm/selfdistill/selfdistill.sh at: $TARBALL"
