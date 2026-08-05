#!/bin/bash
# ---------------------------------------------------------------------------
# Self-distillation training cell on the Alliance cluster. ONE script, three
# arms — the arms differ only in backbone and one augmentation flag, and keeping
# them in one file is what guarantees that (if they drifted apart, the ablation
# would be comparing scripts rather than treatments).
#
#   sbatch slurm/selfdistill/selfdistill.sh birdmae        # C4  <- start here
#   sbatch slurm/selfdistill/selfdistill.sh beats          # C3
#   sbatch slurm/selfdistill/selfdistill.sh birdmae_nobg   # C5, the control
#
# C5 is not optional if you want to claim the de-confounding is caused by the
# invariance objective: adapting on in-domain audio reorganises features on its
# own, so frozen -> nobg -> full is what separates the two explanations.
#
# Run slurm/selfdistill/prestage_selfdistill_data.sh on a LOGIN node first —
# compute nodes have no internet.
# ---------------------------------------------------------------------------
#SBATCH --account=def-XXXX
#SBATCH --job-name=selfdistill
#SBATCH --gpus=h100:1                    # full H100; the MIG 3g.40gb slice used for
                                         # finetune is tight for SSL batch sizes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12               # dataloader decodes 206k wavs; keep it fed
#SBATCH --mem=80G
#SBATCH --time=24:00:00                  # SSL over 206k clips; measure, then tune
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.out
##SBATCH --mail-user=XXXX@gmail.com
##SBATCH --mail-type=END,FAIL

set -euo pipefail

ARM="${1:-birdmae}"

# ============================== USER SETTINGS ==============================
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/links/projects/def-XXXX/$USER/torca_root/torca}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/links/projects/def-XXXX/$USER/torca_root/runs}"
VENV="${VENV:-$HOME/torca_venv}"
DATA_ROOT="${DATA_ROOT:-$HOME/links/projects/def-XXXX/$USER/torca_root/data}"
TARBALL="${TARBALL:-$DATA_ROOT/dclde_clips_3s.tar}"
BIRDMAE_CKPT="${BIRDMAE_CKPT:-$DATA_ROOT/Bird-MAE-B}"
BEATS_CKPT="${BEATS_CKPT:-$DATA_ROOT/BEATs_iter3.pt}"
PARQUET="$PROJECT_ROOT/ds/DCLDE_w_Buzzes.parquet"
# ==========================================================================

# --- arm -> overrides -----------------------------------------------------
case "$ARM" in
  birdmae)
    NETWORK="mim_distillation";        CKPT="$BIRDMAE_CKPT"; EXTRA=() ;;
  beats)
    NETWORK="mim_distillation_beats";  CKPT="$BEATS_CKPT";   EXTRA=() ;;
  birdmae_nobg)
    # The attribution control: identical recipe with the cross-hydrophone noise
    # removed. Everything else — masking, EMA, FSQ, steps, lr — is unchanged, so
    # a difference is attributable to the de-confounding lever and nothing else.
    NETWORK="mim_distillation";        CKPT="$BIRDMAE_CKPT"
    EXTRA=(data.transform.augmentations.student.background.p=0.0) ;;
  *)
    echo "ERROR: unknown arm '$ARM' (birdmae | beats | birdmae_nobg)" >&2; exit 1 ;;
esac

date; hostname
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME  |  arm=$ARM  network=$NETWORK"

# --- environment ----------------------------------------------------------
module load StdEnv/2023 python/3.11 gcc arrow/22.0.0
source "$VENV/bin/activate"

export PROJECT_ROOT OUTPUT_DIR
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

cd "$PROJECT_ROOT"
mkdir -p logs/slurm

# --- sanity checks --------------------------------------------------------
[ -e "$CKPT" ]    || { echo "ERROR: backbone checkpoint not found: $CKPT" >&2; exit 1; }
[ -f "$TARBALL" ] || { echo "ERROR: tarball not found: $TARBALL (run prestage on a login node)" >&2; exit 1; }

# --- stage data to node-local NVMe ---------------------------------------
DATA_DIR="$SLURM_TMPDIR/data"
if [ -f "$DATA_DIR/.staged_ok" ]; then
  echo "Dataset already staged, skipping extraction."
else
  rm -rf "$DATA_DIR"
  echo "Staging dataset to \$SLURM_TMPDIR ..."
  tar -xf "$TARBALL" -C "$SLURM_TMPDIR"
  touch "$DATA_DIR/.staged_ok"
fi

NCLIPS=$(find "$DATA_DIR" -name '*.wav' | wc -l)
echo "Staged $NCLIPS wav files to $DATA_DIR"
# A 3s run against a 5s prestage finds nothing and trains on empty batches
# without erroring, so fail loudly here instead.
[ "$NCLIPS" -gt 1000 ] || { echo "ERROR: only $NCLIPS clips staged — wrong tarball or wrong clip_duration?" >&2; exit 1; }

# --- run ------------------------------------------------------------------
srun python train_selfdistill.py \
    module/network="$NETWORK" \
    trainer=single_gpu \
    trainer.devices=1 \
    trainer.precision=bf16 \
    paths.dataset_dir="$DATA_DIR" \
    data.dataset.parquet_path="$PARQUET" \
    module.network.encoder.pretrained_weights_path="$CKPT" \
    task_name="selfdistill_$ARM" \
    "${EXTRA[@]}"

echo "Finished with exit code $?"
