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

# --- download, write the manifest, tar, and drop the loose tree ------------
# One command does all four. Idempotent: existing clips are skipped, so re-run
# freely after an interruption (which also retries any transient GCS failures).
#
# --tar is not optional in practice. 206k loose files burn inode quota and make
# every job start re-stat the tree through the Lustre metadata server; the tarball
# turns that into one sequential read onto node-local NVMe. The manifest is written
# BEFORE the tar, so it travels inside the archive and prepare_data finds it after
# extraction — its LocalPaths are ignored (only Soundfile is read), which is exactly
# why a staging-node path causes no problem on the compute node.
#
# --remove-loose refuses if the tar entry count is short, so a truncated archive
# cannot silently become your only copy.
python prestage_clips.py \
    --parquet "$PARQUET" \
    --dataset-dir "$STAGE/data" \
    --clip-duration "$CLIP_DURATION" \
    --workers "$WORKERS" \
    --tar "$TARBALL" \
    --remove-loose

echo "Done. Point TARBALL in slurm/selfdistill/selfdistill.sh at: $TARBALL"
