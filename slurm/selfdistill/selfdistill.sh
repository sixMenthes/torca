#!/bin/bash
# ---------------------------------------------------------------------------
# Self-distillation training cell on Nibi. ONE script, three arms — the arms
# differ only in backbone and one augmentation flag, and keeping them in one file
# is what guarantees that (if they drifted apart, the ablation would be comparing
# scripts rather than treatments).
#
# Nibi GPU instance names (--gpus=<name>:<n>):
#   h100            full H100-80GB          (also h100_80gb)
#   h100_3g.40gb    3/8 compute, 40GB       <- default here, see the header below
#   h100_2g.20gb    2/8 compute, 20GB
#   h100_1g.10gb    1/8 compute, 10GB
# Roughly half the GPU nodes are MIG-configured, so a MIG slice usually queues
# sooner than a full card.
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
# --- Nibi GPU instance -----------------------------------------------------
# Cores and memory below are Nibi's RECOMMENDED bundle for the requested instance.
# Asking for more than the bundle makes the job wait for a whole node; asking for
# less wastes allocation you are billed for anyway.
#
#   instance        RGU    recommended
#   h100 (full)     12.2   14 cores, 250 GB
#   h100_3g.40gb     6.1    6 cores, 124 GB   <- default here
#   h100_2g.20gb     3.48   4 cores,  62 GB
#   h100_1g.10gb     1.74   2 cores,  31 GB
#
# Start on 3g.40gb. Nibi bundles cores at a FIXED 1.15 cores per RGU, so a bigger GPU
# brings proportionally more cores and the CPU:GPU balance is identical at every size
# — scaling up does NOT fix a dataloader bottleneck, it scales both sides together.
# So the only question a bigger instance answers is wall-clock vs queue time. Measure
# GPU utilisation on the first run: if it sits low, the loader is the limit and a full
# H100 buys nothing.
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=124G
## full H100 — remember to change ALL THREE lines together:
##SBATCH --gpus=h100:1
##SBATCH --cpus-per-task=14
##SBATCH --mem=250G
#SBATCH --ntasks=1
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

# Checkpoint cadence. model_checkpoint monitors val/loss, so a checkpoint can only
# be written on an epoch that VALIDATES — with the trainer config's
# check_val_every_n_epoch=5 the first one lands after five full epochs (~6.2k steps
# each). Validate every epoch for the first production run: the val set is
# CarmanahPt alone, so a pass is cheap, and a run that dies at hour six having
# saved nothing is not.
MAX_EPOCHS="${MAX_EPOCHS:-5}"
VAL_EVERY="${VAL_EVERY:-1}"
# Two batches through the val loader BEFORE training starts. The trainer config
# disables this, which was fine while every run had limit_val_batches=0 — but that
# means val_ssl_set has never actually been constructed or read. A broken val path
# should cost seconds at job start, not a full epoch.
SANITY_STEPS="${SANITY_STEPS:-2}"
# ==========================================================================

# --- arm -> overrides -----------------------------------------------------
# DATASET must move with NETWORK: the dataset config now carries model_name, which
# picks the fbank applied in the dataloader workers. The datamodule raises if the two
# disagree on sample rate, so a mismatch fails fast rather than training on a wrong
# front-end.
case "$ARM" in
  birdmae)
    NETWORK="mim_distillation";        DATASET="dclde_selfdistill_birdmae"
    CKPT="$BIRDMAE_CKPT"; EXTRA=() ;;
  beats)
    NETWORK="mim_distillation_beats";  DATASET="dclde_selfdistill_beats"
    CKPT="$BEATS_CKPT";   EXTRA=() ;;
  birdmae_nobg)
    # The attribution control: identical recipe with the cross-hydrophone noise
    # removed. Everything else — masking, EMA, FSQ, steps, lr — is unchanged, so
    # a difference is attributable to the de-confounding lever and nothing else.
    NETWORK="mim_distillation";        DATASET="dclde_selfdistill_birdmae"
    CKPT="$BIRDMAE_CKPT"
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

# Dataloader workers track the allocation, so changing --cpus-per-task (or the GPU
# instance) needs no config edit. One core is left for the main process. This is the
# knob that parallelises the fbank front-end, which now runs in the workers.
NWORKERS=$(( ${SLURM_CPUS_PER_TASK:-6} - 1 ))
[ "$NWORKERS" -lt 1 ] && NWORKERS=1
echo "dataloader workers: $NWORKERS (of ${SLURM_CPUS_PER_TASK:-6} cores)"

# Nibi compute nodes have no internet; CodeCarbon must stay offline (it is, by
# config default) or it will block trying to geolocate for a grid-intensity lookup.
export CODECARBON_LOG_LEVEL=error

cd "$PROJECT_ROOT"
mkdir -p logs/slurm

# --- sanity checks --------------------------------------------------------
[ -e "$CKPT" ]    || { echo "ERROR: backbone checkpoint not found: $CKPT" >&2; exit 1; }
[ -f "$TARBALL" ] || { echo "ERROR: tarball not found: $TARBALL (run prestage on a login node)" >&2; exit 1; }

# --- stage data to node-local NVMe ---------------------------------------
# The tarball holds the CONTENTS of the clip tree (packed with `-C dataset_dir .`),
# not a fixed top-level directory, so we choose the destination here and nothing
# depends on what the staging directory was called on the login node.
DATA_DIR="$SLURM_TMPDIR/data"
if [ -f "$DATA_DIR/.staged_ok" ]; then
  echo "Dataset already staged, skipping extraction."
else
  rm -rf "$DATA_DIR"
  mkdir -p "$DATA_DIR"
  echo "Staging dataset to \$SLURM_TMPDIR ..."
  time tar -xf "$TARBALL" -C "$DATA_DIR"
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
    data/dataset="$DATASET" \
    trainer=single_gpu \
    trainer.devices=1 \
    trainer.precision=bf16 \
    trainer.max_epochs="$MAX_EPOCHS" \
    trainer.check_val_every_n_epoch="$VAL_EVERY" \
    trainer.num_sanity_val_steps="$SANITY_STEPS" \
    paths.dataset_dir="$DATA_DIR" \
    data.dataset.parquet_path="$PARQUET" \
    data.loaders.train.num_workers="$NWORKERS" \
    data.loaders.val.num_workers=2 \
    module.network.encoder.pretrained_weights_path="$CKPT" \
    task_name="selfdistill_$ARM" \
    "${EXTRA[@]}"

echo "Finished with exit code $?"
