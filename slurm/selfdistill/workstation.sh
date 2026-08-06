#!/bin/bash
# ---------------------------------------------------------------------------
# Self-distillation debug sequence for the WORKSTATION (tundra).
#
# Plain bash, not sbatch — this is the machine you debug on before anything is
# submitted to the cluster. It runs the whole pipeline in dependency order and
# stops at the first failure, so whatever breaks, breaks here rather than in a
# queued job.
#
#   bash slurm/selfdistill/workstation.sh            # everything, in order
#   bash slurm/selfdistill/workstation.sh prestage   # just one step
#   bash slurm/selfdistill/workstation.sh smoke probes
#
# Steps:
#   prestage  download 3s clips + write the cached manifest   (do this FIRST)
#   manifest  rewrite the manifest from what is already on disk, no download
#   ckpt      load Bird-MAE-B at target_length=304
#   smoke     fast_dev_run of both arms on real data
#   probes    MFCC floor + frozen controls  <- the no-training cells live here
#   train     a real Bird-MAE run (only if you want one locally)
# ---------------------------------------------------------------------------
set -euo pipefail

# ============================== USER SETTINGS ==============================
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/torca}"
VENV="${VENV:-$PROJECT_ROOT/.venv}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/ds/clips}"          # == data.dataset.dataset_dir
PARQUET="${PARQUET:-$PROJECT_ROOT/ds/DCLDE_w_Buzzes.parquet}"
BIRDMAE_CKPT="${BIRDMAE_CKPT:-$PROJECT_ROOT/pretrained/Bird-MAE-B}"
BEATS_CKPT="${BEATS_CKPT:-$PROJECT_ROOT/pretrained/BEATs_iter3.pt}"
CLIP_DURATION="${CLIP_DURATION:-3.0}"                   # MUST match data.dataset.clip_duration
# MUST match data.dataset.manifest_name in BOTH dclde_selfdistill_*.yaml. Named per
# clip_duration because clips are {start_ms}-{end_ms}.wav: a 5s stage and a 3s stage
# share no files, so a shared manifest name would let one masquerade as the other.
MANIFEST_NAME="${MANIFEST_NAME:-DCLDE_3secs.parquet}"
WORKERS="${WORKERS:-32}"
# ==========================================================================

source "$VENV/bin/activate"
export PROJECT_ROOT HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"
mkdir -p logs

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(prestage ckpt smoke probes)
runs() { [[ " ${STEPS[*]} " == *" $1 "* ]]; }
banner() { echo; echo "=================== $1 ==================="; date; }

# Shared overrides. Every entry point needs the data location and the parquet, since
# the configs' defaults point at paths that only exist on the original dev box.
COMMON=(
  paths.dataset_dir="$DATA_DIR"
  data.dataset.parquet_path="$PARQUET"
  data.dataset.manifest_name="$MANIFEST_NAME"
)

# --- 1. clips -------------------------------------------------------------
# Everything downstream is silently wrong without this: LocalPath encodes the
# window as {start_ms}-{end_ms}.wav, so clips staged at 5s are INVISIBLE to a 3s
# run (zero filename overlap out of 206k). prepare_data then skips downloading,
# every row misses, collate_fn_skip returns None, and every batch is dropped.
# --verify afterwards is the check that this actually happened; it writes nothing.
if runs prestage; then
  banner "PRESTAGE  clip_duration=$CLIP_DURATION"
  python prestage_clips.py \
      --parquet "$PARQUET" \
      --dataset-dir "$DATA_DIR" \
      --clip-duration "$CLIP_DURATION" \
      --manifest-name "$MANIFEST_NAME" \
      --workers "$WORKERS"
  python prestage_clips.py --parquet "$PARQUET" --dataset-dir "$DATA_DIR" \
      --clip-duration "$CLIP_DURATION" --manifest-name "$MANIFEST_NAME" --verify
fi

# --- 1b. manifest only ----------------------------------------------------
# For when the clips already landed but the manifest did not, or landed under an
# older name: one os.walk over the tree, no downloading, no re-stat storm. This is
# also the cheap repair after an interrupted prestage.
if runs manifest; then
  banner "MANIFEST ONLY  -> $DATA_DIR/$MANIFEST_NAME"
  python prestage_clips.py \
      --parquet "$PARQUET" \
      --dataset-dir "$DATA_DIR" \
      --clip-duration "$CLIP_DURATION" \
      --manifest-name "$MANIFEST_NAME" \
      --manifest-only
fi

# --- 2. does the backbone actually load at 304? ---------------------------
# Watch the [audiomae] lines: pos_embed SHOULD be dropped (it is fixed sincos,
# regenerated at this grid); anything ELSE in the missing list means the backbone
# is partly random, which reads exactly like "self-distillation doesn't work".
if runs ckpt; then
  banner "CHECKPOINT SANITY"
  CLIP=$(find "$DATA_DIR" -name '*.wav' | head -1)
  [ -n "$CLIP" ] || { echo "no clips found in $DATA_DIR — run prestage first" >&2; exit 1; }
  python test_birdmae.py "$BIRDMAE_CKPT" "$CLIP"
fi

# --- 3. does the loop run on REAL data? -----------------------------------
# experiment=... rather than module/network=...: the fbank front-end now runs in the
# dataloader workers, so the backbone is selected by data/dataset AND module/network
# together. The experiment configs bind the pair; a bare network override would leave
# the dataset building the other backbone's fbank (the datamodule raises, but only
# because someone added a cross-check for exactly this).
if runs smoke; then
  banner "FAST_DEV_RUN  Bird-MAE"
  python train_selfdistill.py +trainer.fast_dev_run=true \
      experiment=selfdistill_birdmae \
      module.network.encoder.pretrained_weights_path="$BIRDMAE_CKPT" \
      "${COMMON[@]}"

  banner "FAST_DEV_RUN  BEATs"
  python train_selfdistill.py +trainer.fast_dev_run=true \
      experiment=selfdistill_beats \
      module.network.encoder.pretrained_weights_path="$BEATS_CKPT" \
      "${COMMON[@]}"
fi

# --- 4. the no-training cells --------------------------------------------
# These need no adaptation and no queue, so they establish the floor and the
# frozen controls while the cluster jobs are still pending.
#
# probe.yaml has no `experiment` group, and does not need one: probing never builds
# a front-end from the dataset config (the encoder applies its own), so only
# module/network matters here and the dataset choice is irrelevant.
if runs probes; then
  banner "PROBE  C0: MFCC floor"
  python probe_selfdistill.py source=mfcc "${COMMON[@]}"

  banner "PROBE  C2: Bird-MAE frozen"
  python probe_selfdistill.py source=frozen \
      module/network=mim_distillation \
      module.network.encoder.pretrained_weights_path="$BIRDMAE_CKPT" \
      "${COMMON[@]}"

  banner "PROBE  C1: BEATs frozen"
  python probe_selfdistill.py source=frozen \
      module/network=mim_distillation_beats \
      module.network.encoder.pretrained_weights_path="$BEATS_CKPT" \
      "${COMMON[@]}"
fi

# --- 5. optional: a real local run ---------------------------------------
# task_name has to be set here rather than in the experiment file: _self_ is LAST in
# selfdistill.yaml's defaults, so that file's own task_name would win over one set in
# an experiment config. It also names the MLflow experiment (logger.experiment_name).
if runs train; then
  banner "TRAIN  Bird-MAE (local)"
  python train_selfdistill.py \
      experiment=selfdistill_birdmae \
      task_name=selfdistill_birdmae \
      module.network.encoder.pretrained_weights_path="$BIRDMAE_CKPT" \
      "${COMMON[@]}"
fi

banner "DONE"
