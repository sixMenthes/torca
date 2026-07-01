#!/bin/bash
# ---------------------------------------------------------------------------
# Torca finetune on the Alliance Canada "Rorqual" cluster.
# Runs configs/torca.yaml via finetune.py on ONE H100-80GB GPU.
#
#   Rorqual has H100 GPUs, NOT A100 (the Alliance A100 cluster is Narval).
#   H100-80GB strictly dominates an A100 for this job, so we use h100.
#
# Compute nodes on Rorqual have NO internet, so the dataset is NOT downloaded
# here. Run slurm/finetune/prestage_torca_data.sh on a LOGIN node first to
# build the data tarball; this job copies it to node-local $SLURM_TMPDIR,
# extracts, and trains offline.
#
# Submit:  sbatch slurm/finetune/torca_fir.sh
# ---------------------------------------------------------------------------
#SBATCH --account=def-XXXX               # <-- your Alliance allocation (def-/rrg-)
#SBATCH --job-name=torca_finetune
#SBATCH --gpus-per-node=h100_3g.40gb:1            # one H100-80GB (Fir's gres type is h100)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # <= 24 (= 1/8 of a 192-core Fir GPU node)
#SBATCH --mem=60G
#SBATCH --time=02:00:00                  # 60 epochs; tune to your measured epoch time
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.out
##SBATCH --mail-user=XXXX@gmail.com
##SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================== USER SETTINGS ==============================
PROJECT_ROOT="$HOME/links/projects/def-XXXX/$USER/torca_root/torca"                 # repo root on Fir (has finetune.py)
OUTPUT_DIR="$HOME/links/projects/def-XXXX/$USER/torca_root/runs"           # scratch BASE; hydra appends /runs/.../<timestamp>
VENV="$HOME/torca_venv"                                            # prebuilt virtualenv
CKPT="$HOME/links/projects/def-XXXX/$USER/torca_root/data/MODELNAME"  # pretrained backbone
TARBALL="$HOME/links/projects/def-XXXX/$USER/torca_root/data/dclde_clips.tar" 
PARQUET="$PROJECT_ROOT/ds/DCLDE_w_Buzzes.parquet"                  # absolute (Hydra chdir=True)
# ==========================================================================

date; hostname
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"

# --- environment ---------------------------------------------------------
module load StdEnv/2023 python/3.11 gcc arrow/22.0.0
       # match the modules your venv was built with
source "$VENV/bin/activate"

export PROJECT_ROOT
export OUTPUT_DIR                           # consumed by paths.scratch_dir -> hydra.run.dir
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "$PROJECT_ROOT"
mkdir -p logs/slurm

# --- sanity checks -------------------------------------------------------
[ -f "$CKPT" ]    || { echo "ERROR: backbone checkpoint not found: $CKPT" >&2; exit 1; }
[ -f "$TARBALL" ] || { echo "ERROR: data tarball not found: $TARBALL  (run prestage_torca_data.sh on a login node)" >&2; exit 1; }

# --- stage data to node-local NVMe (avoids 206k small files on Lustre) ---
echo "Staging dataset to \$SLURM_TMPDIR ..."
tar -xf "$TARBALL" -C "$SLURM_TMPDIR"      # extracts a 'data/' tree
DATA_DIR="$SLURM_TMPDIR/data"
echo "Staged $(find "$DATA_DIR" -name '*.wav' | wc -l) wav files to $DATA_DIR"

# --- run -----------------------------------------------------------------
# dataset_dir points at the local copy; every clip already exists there, so
# TorcaDataModule.prepare_data() finds all files present and never hits GCS.
srun python finetune.py \
    --config-name=torca \
    trainer=single_gpu \
    trainer.devices=1 \
    trainer.precision=bf16 \
    paths.dataset_dir="$DATA_DIR" \
    data.dataset.parquet_path="$PARQUET" \
    module.network.pretrained_weights_path="$CKPT"

echo "Finished with exit code $?"
