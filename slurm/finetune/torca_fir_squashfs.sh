#!/bin/bash
# ---------------------------------------------------------------------------
# Torca finetune on Fir, one H100-80GB GPU -- SquashFS staging variant.
#
# Reads the dataset from a read-only SquashFS image (built by
# prestage_torca_data_squashfs.sh) mounted with squashfuse. No extraction,
# so it needs no large $SLURM_TMPDIR space -- use this when the dataset is
# too big to copy node-local.
#
# Submit:  sbatch slurm/finetune/torca_fir_squashfs.sh
# ---------------------------------------------------------------------------
#SBATCH --account=def-XXXX
#SBATCH --job-name=torca_finetune
#SBATCH --gpus-per-node=h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.out

set -euo pipefail

# ============================== USER SETTINGS ==============================
PROJECT_ROOT="$HOME/projects/def-XXXX/$USER/torca"
OUTPUT_DIR="$HOME/scratch/torca"           # scratch BASE; hydra appends /runs/.../<timestamp>
VENV="$HOME/torca_venv"
CKPT="$PROJECT_ROOT/checkpoints/AudioMAE_XCL_epoch=99_mixup.ckpt"
SQFS="$HOME/projects/def-XXXX/$USER/torca_data/dclde_clips.sqfs"   # built by prestage_..._squashfs.sh
PARQUET="$PROJECT_ROOT/ds/DCLDE_w_Buzzes.parquet"
COPY_TO_TMPDIR=0     # 1 = copy the .sqfs to $SLURM_TMPDIR first (hotter random I/O, needs the space)
# ==========================================================================

date; hostname
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"

module load StdEnv/2023 python/3.10 squashfuse
source "$VENV/bin/activate"

export PROJECT_ROOT OUTPUT_DIR HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "$PROJECT_ROOT"
mkdir -p logs/slurm

[ -f "$CKPT" ] || { echo "ERROR: backbone checkpoint not found: $CKPT" >&2; exit 1; }
[ -f "$SQFS" ] || { echo "ERROR: SquashFS image not found: $SQFS  (run prestage_torca_data_squashfs.sh)" >&2; exit 1; }

# --- mount the image read-only ------------------------------------------
IMG="$SQFS"
if [ "$COPY_TO_TMPDIR" = "1" ]; then
    echo "Copying image to \$SLURM_TMPDIR ..."
    cp "$SQFS" "$SLURM_TMPDIR/" && IMG="$SLURM_TMPDIR/$(basename "$SQFS")"
fi

MNT="$SLURM_TMPDIR/dclde_mnt"
mkdir -p "$MNT"
squashfuse "$IMG" "$MNT"
trap 'fusermount -u "$MNT" 2>/dev/null || true' EXIT
DATA_DIR="$MNT"     # image root holds the <Provider>/<Dataset>/... tree
echo "Mounted $(find "$DATA_DIR" -maxdepth 1 | wc -l) top-level entries at $DATA_DIR"

# --- run -----------------------------------------------------------------
srun python finetune.py \
    --config-name=torca \
    trainer=single_gpu \
    trainer.devices=1 \
    trainer.precision=bf16 \
    data.dataset.dataset_dir="$DATA_DIR" \
    data.dataset.parquet_path="$PARQUET" \
    module.network.pretrained_weights_path="$CKPT"

echo "Finished with exit code $?"
