#!/bin/bash
# ---------------------------------------------------------------------------
# One-time data prestaging (SquashFS variant) for the Torca finetune on Fir.
#
# RUN THIS ON A LOGIN NODE (or fir-dtn), NOT via sbatch:
#     bash slurm/finetune/prestage_torca_data_squashfs.sh
#
# Same download as prestage_torca_data.sh, but instead of a tar it builds a
# read-only SquashFS image. The job then MOUNTS the .sqfs (one Lustre inode)
# and reads clips on demand -- no 100-200 GB extraction into $SLURM_TMPDIR.
# Use this variant when the extracted dataset is too big to copy node-local.
# ---------------------------------------------------------------------------
set -euo pipefail

# ============================== USER SETTINGS ==============================
PROJECT_ROOT="$HOME/projects/def-XXXX/$USER/torca"   # repo root on Fir
VENV="$HOME/torca_venv"                              # venv (needs gcsfs, soundfile, polars)
STAGE="$SCRATCH/torca_stage"                         # loose-file staging on scratch (1M-inode quota)
SQFS="$HOME/projects/def-XXXX/$USER/torca_data/dclde_clips.sqfs"   # final image on project
# ==========================================================================

module load StdEnv/2023 python/3.10
module load squashfs-tools                           # provides mksquashfs (or: it's in StdEnv)
source "$VENV/bin/activate"
export PROJECT_ROOT
export HYDRA_FULL_ERROR=1

mkdir -p "$STAGE/data" "$(dirname "$SQFS")"
cd "$PROJECT_ROOT"

# --- download every clip into $STAGE/data via TorcaDataModule.prepare_data ---
python - "$STAGE/data" "$PROJECT_ROOT" <<'PY'
import os, sys
from hydra import initialize_config_dir, compose
from torca_datamodule import TorcaDataModule

stage_data_dir, project_root = sys.argv[1], sys.argv[2]
with initialize_config_dir(version_base=None, config_dir=os.path.join(project_root, "configs")):
    cfg = compose(
        config_name="torca",
        overrides=[
            f"data.dataset.dataset_dir={stage_data_dir}",
            f"data.dataset.parquet_path={project_root}/ds/DCLDE_w_Buzzes.parquet",
        ],
    )
dm = TorcaDataModule(cfg.data.dataset, cfg.data.loaders, cfg.data.transform)
dm.prepare_data()
print("Download complete:", stage_data_dir)
PY

n=$(find "$STAGE/data" -name '*.wav' | wc -l)
echo "Downloaded $n wav clips."

# --- build the SquashFS image (contents of $STAGE/data become image root) ---
# wav is ~incompressible, so -noD (store data uncompressed) trades a little
# space for much faster build and read; drop it if you want compression.
echo "Building SquashFS image -> $SQFS"
rm -f "$SQFS"
mksquashfs "$STAGE/data" "$SQFS" -noD -processors "$(nproc)" -quiet
echo "Image size: $(du -h "$SQFS" | cut -f1)"

echo "Removing loose staging files from scratch ..."
rm -rf "$STAGE/data"

echo "Done. Point torca_fir_squashfs.sh's SQFS at: $SQFS"
