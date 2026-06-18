#!/bin/bash
# ---------------------------------------------------------------------------
# One-time data prestaging for the Torca finetune on Fir.
#
# RUN THIS ON A LOGIN NODE (or fir-dtn), NOT via sbatch:
#     bash slurm/finetune/prestage_torca_data.sh
#
# Why: Fir compute nodes have no internet, so the GPU job cannot pull clips
# from gs://noaa-passive-bioacoustic/... . This script downloads the ~206k
# DCLDE clips once, then packs them into a single tar so they live on Lustre
# as ONE file instead of 206k small files (which would blow the inode quota
# and crush the metadata server -- see the Alliance "Handling large
# collections of files" guidance).
#
# It is I/O- and CPU-heavy (decodes 13k source files into 206k clips). Run it
# inside tmux/screen so it survives disconnects; it is a one-time cost.
# ---------------------------------------------------------------------------
set -euo pipefail

# ============================== USER SETTINGS ==============================
PROJECT_ROOT="$HOME/projects/def-XXXX/$USER/torca"   # repo root on Fir
VENV="$HOME/torca_venv"                              # prebuilt virtualenv (needs gcsfs, soundfile, polars)
STAGE="$SCRATCH/torca_stage"                         # loose-file staging on scratch (1M-inode quota)
TARBALL="$HOME/projects/def-XXXX/$USER/torca_data/dclde_clips.tar"   # final single-file archive on project
# ==========================================================================

module load StdEnv/2023 python/3.10
source "$VENV/bin/activate"
export PROJECT_ROOT
export HYDRA_FULL_ERROR=1

mkdir -p "$STAGE/data" "$(dirname "$TARBALL")"
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
dm.prepare_data()   # streams clips from GCS, writes <dataset_dir>/<Provider>/<Dataset>/<stem>/<start>-<end>.wav
print("Download complete:", stage_data_dir)
PY

n=$(find "$STAGE/data" -name '*.wav' | wc -l)
echo "Downloaded $n wav clips."

# --- pack into a single tar on project, then drop the loose files ----------
echo "Packing tarball -> $TARBALL"
tar -cf "$TARBALL" -C "$STAGE" data
echo "Tarball size: $(du -h "$TARBALL" | cut -f1)"

echo "Removing loose staging files from scratch ..."
rm -rf "$STAGE/data"

echo "Done. Point torca_fir.sh's TARBALL at: $TARBALL"
