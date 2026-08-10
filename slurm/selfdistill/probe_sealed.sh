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

# Sealed by default, which is what this script is named for. SEAL=0 runs the REPORTED
# protocol instead: fit the probe on the train hydrophones, choose C by GroupKFold
# within train, and score ONCE on the held-out test hydrophones.
#
# The reported protocol is the number the study actually claims, because the test split
# is held out BY HYDROPHONE (StraitofGeorgia, CarmanahPt, BarkleyCanyon) and therefore
# measures generalisation to recording sites the encoder never adapted on. The sealed
# number is a proxy for it and reads optimistically high, since the backbone adapted on
# every site in its GroupKFold pool. It also unlocks the call-type probe, which is
# skipped entirely under seal because its test mask IS the sealed split.
#
# It can only be spent once. Read the seal_test block in configs/probe.yaml before
# setting this, and decide which cells go in the table BEFORE looking at any of them:
# choosing the winner afterwards is model selection on the test split.
SEAL="${SEAL:-1}"
case "$SEAL" in
  1|true|yes)  SEAL_TEST=true;  PROTOCOL=sealed ;;
  0|false|no)  SEAL_TEST=false; PROTOCOL=reported ;;
  *) echo "SEAL must be 1 or 0, got '$SEAL'" >&2; exit 1 ;;
esac

# Write the pooled features out alongside the metrics, so the UMAP figure can be built
# on the workstation without re-running the backbone pass or rsyncing a 1.4 GB
# checkpoint. SAVE_FEATURES=0 turns it off. Roughly 40 MB per cell as float16.
#
# One directory for the whole batch, and the filenames carry the cell, because a
# projection is only comparable across cells when every cell was projected by the same
# fit — which means one process reading all of them at once.
SAVE_FEATURES="${SAVE_FEATURES:-1}"

# ============================== USER SETTINGS ==============================
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/def-XXXX/$USER/torca}"
VENV="${VENV:-$HOME/.torca_venv}"
DATA_ROOT="${DATA_ROOT:-$HOME/projects/def-XXXX/$USER}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT}"
# Resolved here rather than beside SAVE_FEATURES above, because OUTPUT_DIR is not
# defined until this line and the script runs with `set -u`. An `if` rather than an
# AND-list for the reason given in selfdistill.sh: the AND-list form is exempt from
# `set -e` only by a subclause of its rules.
FEATURE_DIR=null
if [ "$SAVE_FEATURES" != "0" ]; then
  FEATURE_DIR="$OUTPUT_DIR/features/$PROTOCOL"
fi
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
if [ "$SOURCE" = "adapted" ] && [ -z "$CKPT_PATH" ] && [ -z "${CKPT_GLOB:-}" ]; then
  echo "ERROR: cell $CELL is an adapted cell and needs a checkpoint, either as the" >&2
  echo "       second argument or as CKPT_GLOB in the environment (which is resolved" >&2
  echo "       inside the job, so a probe can be chained onto a training run that has" >&2
  echo "       not written its checkpoint yet)." >&2
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

# A bare `cd` into a placeholder path fails with the shell's own terse message four
# seconds into an allocation, and gives no hint that ACCOUNT is what needs setting.
# It has already cost a full batch of eleven jobs, so it says so now.
if [ ! -d "$PROJECT_ROOT" ]; then
  echo "ERROR: PROJECT_ROOT does not exist: $PROJECT_ROOT" >&2
  echo "       The default is built from a PLACEHOLDER allocation name, so it only" >&2
  echo "       resolves when PROJECT_ROOT or ACCOUNT is supplied. Submit through" >&2
  echo "       slurm/selfdistill/launch_probes_renorm.sh, which passes it, or set it:" >&2
  echo "         sbatch --export=ALL,PROJECT_ROOT=\$HOME/projects/def-yourpi/\$USER/torca ..." >&2
  exit 1
fi
cd "$PROJECT_ROOT"
mkdir -p logs/slurm

# CKPT_GLOB resolves the checkpoint HERE, inside the job, rather than at submit time.
# That is what makes a probe chainable onto a training run with --dependency: when the
# pair is submitted together the checkpoint does not exist yet, so a path argument
# would fail the existence check below before training had a chance to write it.
# Unquoted on purpose, so the shell expands the pattern; newest match wins.
if [ -z "$CKPT_PATH" ] && [ -n "${CKPT_GLOB:-}" ]; then
  CKPT_PATH=$(ls -t $CKPT_GLOB 2>/dev/null | head -1)
  if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: CKPT_GLOB matched nothing: $CKPT_GLOB" >&2
    echo "       The training job it was chained to should have written a checkpoint." >&2
    exit 1
  fi
  echo "resolved CKPT_GLOB -> $CKPT_PATH"
fi

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
    seal_test="$SEAL_TEST" \
    save_features="$FEATURE_DIR" \
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
    task_name="probe_${PROTOCOL}_$CELL"

echo "Finished with exit code $?"
echo
echo "Copy the metrics back (kilobytes, not gigabytes):"
echo "  rsync -av nibi:$OUTPUT_DIR/mlruns/ /home/tundra/claude/torca/mlruns_nibi/mlruns/"
