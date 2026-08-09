#!/bin/bash
# ---------------------------------------------------------------------------
# The SEALED probe, run ON THE CLUSTER against a checkpoint that is already there.
#
#   sbatch slurm/selfdistill/probe_sealed.sh <cell> <ckpt>
#   sbatch slurm/selfdistill/probe_sealed.sh C4 \
#       $HOME/projects/def-XXXX/$USER/runs/selfdistill_birdmae/DCLDE_selfdistill_birdmae/MIMDistillation/2026-08-07_165712/checkpoints/last.ckpt
#
# WHY THIS EXISTS. The reported six-cell table runs on the workstation, because every
# cell has to be probed by one process against one MLflow store or the numbers are not
# comparable. That requires rsyncing a 1.4 GB checkpoint back per adapted cell, which is
# half an hour on this link. While TUNING that is the wrong trade: the question being
# asked is "did the nuisance metric move at all", the answer is a few kilobytes of
# metrics, and the checkpoint is already sitting on the cluster filesystem.
#
# So: run the sealed probe here, copy back the mlruns directory, and leave the
# checkpoint where it is until the recipe is frozen and the reported table is due.
#
# WHAT IT COSTS, AND HOW LONG IT TAKES. The GPU part is one forward pass over 26,735
# train clips with no backward and no optimiser. The eighteen-epoch training run was
# 18 x 51,072 clips with a teacher pass, a student pass and a backward on each, and it
# took 54 minutes, so the extraction here is under one percent of that. GPU time is not
# what this job spends its wall-clock on. Expect roughly:
#
#   staging the tarball to $SLURM_TMPDIR    a few minutes, and the least predictable
#                                           part — it is the same extraction
#                                           selfdistill.sh times, so read that job's
#                                           log for the real figure on this filesystem
#   feature extraction, 836 batches         1-3 minutes
#   ecotype probe, GroupKFold x 5 folds     several minutes; 26,735 x 768 features
#   nuisance probe, Stratified x 5 folds    several minutes; 7,234 Background rows
#                                           over 12 hydrophone classes
#
# So ten to twenty minutes of run time inside a 40-minute request. If it ever times out
# it will be in staging, not in compute.
#
# WHAT SEALED MEANS. seal_test=true drops the test and val rows in probe_pool, BEFORE
# feature extraction, so the backbone never sees a held-out clip. The ecotype probe
# becomes GroupKFold by hydrophone within train and the call-type probe is skipped
# entirely, since its test mask IS the sealed split. The nuisance probe is unchanged,
# because it always ran on train. Metric keys differ from the reported run
# (task_ecotype_cvtrain, not task_ecotype) so the two cannot be overlaid by accident.
# ---------------------------------------------------------------------------
#SBATCH --account=def-XXXX
#SBATCH --job-name=probe_sealed
# Same slice and core count as selfdistill.sh, which is known to be grantable. The
# GPU is oversized for one inference pass, but Nibi bundles cores at a fixed ratio to
# the GPU request, and the CORES are what this job is short of: the probes themselves
# are sklearn LogisticRegression(max_iter=2000) over 26,735 rows of 768 features across
# five folds, and those fits take longer than the forward pass does. Asking for a 1g
# slice to save GPU would cut the core allocation with it and make the job slower.
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --ntasks=1
# Generous against a job whose compute is a few minutes, because the variable parts
# are staging the clip tree out of the tarball and the sklearn fits. See the duration
# breakdown at the bottom of this comment block.
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.out

set -euo pipefail

CELL="${1:?usage: sbatch probe_sealed.sh <cell> [ckpt]   e.g. C2, or C4 /path/last.ckpt}"
CKPT_PATH="${2:-}"

# ============================== USER SETTINGS ==============================
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/def-XXXX/$USER/torca}"
VENV="${VENV:-$HOME/.torca_venv}"
DATA_ROOT="${DATA_ROOT:-$HOME/projects/def-XXXX/$USER}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT}"
TARBALL="${TARBALL:-$DATA_ROOT/dclde_clips_3s.tar}"
PARQUET="$PROJECT_ROOT/ds/DCLDE_w_Buzzes.parquet"
# The CONTROL cells matter as much as the adapted ones here, and more than they did
# before. An adapted cell's sealed numbers are meaningless on their own: the sealed task
# metric is GroupKFold within train, which is a different estimator from the reported
# fit-on-train/score-on-test figure, so comparing an adapted sealed number against a
# frozen REPORTED number measures the change of estimator rather than the effect of
# adaptation. Every adapted cell needs its frozen counterpart run through this same
# script before either number means anything.
#
# C0 needs no encoder at all and C1/C2 need no checkpoint, so the second argument is
# required only for C3/C4/C5.
case "$CELL" in
  C0)    SOURCE="mfcc";    NETWORK="mim_distillation"; DATASET="dclde_selfdistill_birdmae"
         BACKBONE="$DATA_ROOT/Bird-MAE-B" ;;
  C1)    SOURCE="frozen";  NETWORK="mim_distillation_beats"; DATASET="dclde_selfdistill_beats"
         BACKBONE="$DATA_ROOT/BEATs_iter3.pt" ;;
  C2)    SOURCE="frozen";  NETWORK="mim_distillation"; DATASET="dclde_selfdistill_birdmae"
         BACKBONE="$DATA_ROOT/Bird-MAE-B" ;;
  C3)    SOURCE="adapted"; NETWORK="mim_distillation_beats"; DATASET="dclde_selfdistill_beats"
         BACKBONE="$DATA_ROOT/BEATs_iter3.pt" ;;
  # C6 is the birdmae_teacherbg arm: same backbone and same probe path as C4 and C5, so
  # it needs no special handling here beyond having a name of its own. Giving it one
  # matters for the same reason C4 and C5 have separate labels — the three differ only
  # in which checkpoint they read, and an unlabelled run is indistinguishable in the
  # results table from the cell it is supposed to be compared against.
  C4|C5|C6) SOURCE="adapted"; NETWORK="mim_distillation"; DATASET="dclde_selfdistill_birdmae"
         BACKBONE="$DATA_ROOT/Bird-MAE-B" ;;
  *)
    # Any other label is an ADAPTED cell on the arm named by ARM, default birdmae.
    #
    # Tuning runs need labels of their own. Reusing C4 for a second adapted checkpoint
    # would put two different models under the same cell tag writing the same metric
    # keys, distinguishable only by timestamp, which is exactly the confusion
    # configs/probe.yaml warns about for C4 against C5. So `probe_sealed.sh T1 <ckpt>`
    # is allowed and lands under its own label.
    SOURCE="adapted"
    case "${ARM:-birdmae}" in
      birdmae) NETWORK="mim_distillation";       DATASET="dclde_selfdistill_birdmae"
               BACKBONE="$DATA_ROOT/Bird-MAE-B" ;;
      beats)   NETWORK="mim_distillation_beats"; DATASET="dclde_selfdistill_beats"
               BACKBONE="$DATA_ROOT/BEATs_iter3.pt" ;;
      *) echo "unknown ARM '${ARM}' (birdmae | beats)" >&2; exit 1 ;;
    esac
    echo "cell '$CELL' is not one of C0-C6; treating it as an adapted ${ARM:-birdmae} cell"
    ;;
esac
if [ "$SOURCE" = "adapted" ] && [ -z "$CKPT_PATH" ]; then
  echo "ERROR: cell $CELL is an adapted cell and needs a checkpoint as the second argument" >&2
  exit 1
fi
# ==========================================================================

module load StdEnv/2023 python/3.11 gcc arrow/22.0.0
source "$VENV/bin/activate"
export PROJECT_ROOT OUTPUT_DIR HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
# The Alliance mlflow wheel refuses a file-backed tracking store without this. Same
# reason it is in selfdistill.sh and preflight.sh.
export MLFLOW_ALLOW_FILE_STORE=true
# Compute nodes have no internet; CodeCarbon must not try to geolocate.
export CODECARBON_LOG_LEVEL=error

cd "$PROJECT_ROOT"
mkdir -p logs/slurm

if [ -n "$CKPT_PATH" ]; then
  [ -f "$CKPT_PATH" ] || { echo "ERROR: checkpoint not found: $CKPT_PATH" >&2; exit 1; }
fi
[ -e "$BACKBONE" ] || { echo "ERROR: backbone not found: $BACKBONE" >&2; exit 1; }
[ -f "$TARBALL" ]  || { echo "ERROR: tarball not found: $TARBALL" >&2; exit 1; }

# --- stage data to node-local NVMe ---------------------------------------
DATA_DIR="$SLURM_TMPDIR/data"
if [ -f "$DATA_DIR/.staged_ok" ]; then
  echo "Dataset already staged, skipping extraction."
else
  rm -rf "$DATA_DIR"; mkdir -p "$DATA_DIR"
  echo "Staging dataset to \$SLURM_TMPDIR ..."
  time tar -xf "$TARBALL" -C "$DATA_DIR"
  touch "$DATA_DIR/.staged_ok"
fi
NCLIPS=$(find "$DATA_DIR" -name '*.wav' | wc -l)
echo "Staged $NCLIPS wav files to $DATA_DIR"
[ "$NCLIPS" -gt 1000 ] || { echo "ERROR: only $NCLIPS clips staged" >&2; exit 1; }

# --- run ------------------------------------------------------------------
# paths=cluster hydra=cluster are NOT optional: probe.yaml defaults to the workstation
# variants of both, and without these the results land under $PROJECT_ROOT/mlruns
# instead of $OUTPUT_DIR/mlruns, which is not what gets copied back.
CKPT_OVERRIDE=()
if [ -n "$CKPT_PATH" ]; then
  CKPT_OVERRIDE=("ckpt_path=$CKPT_PATH")
fi

srun python probe_selfdistill.py \
    paths=cluster \
    hydra=cluster \
    source="$SOURCE" \
    seal_test=true \
    cell="$CELL" \
    "${CKPT_OVERRIDE[@]}" \
    module/network="$NETWORK" \
    data/dataset="$DATASET" \
    paths.dataset_dir="$DATA_DIR" \
    data.dataset.parquet_path="$PARQUET" \
    module.network.encoder.pretrained_weights_path="$BACKBONE" \
    num_workers=4 \
    probe_n_jobs="${SLURM_CPUS_PER_TASK:-4}" \
    device=cuda \
    task_name="probe_sealed_$CELL"

echo "Finished with exit code $?"
echo
echo "Copy the metrics back (kilobytes, not gigabytes):"
echo "  rsync -av nibi:$OUTPUT_DIR/mlruns/ /home/tundra/claude/torca/mlruns_nibi/mlruns/"
