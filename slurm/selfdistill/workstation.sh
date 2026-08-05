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
#   ckpt      load Bird-MAE-B at target_length=304            (never yet run for real)
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

# --- 1. clips -------------------------------------------------------------
# Everything downstream is silently wrong without this: LocalPath encodes the
# window as {start_ms}-{end_ms}.wav, so clips staged at 5s are INVISIBLE to a 3s
# run (zero filename overlap out of 206k). prepare_data then skips downloading,
# every row misses, collate_fn_skip returns None, and you train on empty batches
# with no error. --verify afterwards is the check that this actually happened.
if runs prestage; then
  banner "PRESTAGE  clip_duration=$CLIP_DURATION"
  python prestage_clips.py \
      --parquet "$PARQUET" \
      --dataset-dir "$DATA_DIR" \
      --clip-duration "$CLIP_DURATION" \
      --workers "$WORKERS"
  python prestage_clips.py --parquet "$PARQUET" --dataset-dir "$DATA_DIR" \
      --clip-duration "$CLIP_DURATION" --verify
fi

# --- 2. does the backbone actually load at 304? ---------------------------
# The general-length branch in VIT.load_pretrained_weights has only ever been
# tested against a synthetic 512-length checkpoint. Watch the [audiomae] lines:
# pos_embed SHOULD be dropped (it is fixed sincos, regenerated at this grid);
# anything ELSE in the missing list means the backbone is partly random.
if runs ckpt; then
  banner "CHECKPOINT SANITY"
  CLIP=$(find "$DATA_DIR" -name '*.wav' | head -1)
  [ -n "$CLIP" ] || { echo "no clips found in $DATA_DIR — run prestage first" >&2; exit 1; }
  python test_birdmae.py "$BIRDMAE_CKPT" "$CLIP"
fi

# --- 3. does the loop run on REAL data? -----------------------------------
if runs smoke; then
  banner "FAST_DEV_RUN  Bird-MAE"
  python train_selfdistill.py trainer.fast_dev_run=true \
      paths.dataset_dir="$DATA_DIR" \
      data.dataset.parquet_path="$PARQUET" \
      module.network.encoder.pretrained_weights_path="$BIRDMAE_CKPT"

  banner "FAST_DEV_RUN  BEATs"
  python train_selfdistill.py trainer.fast_dev_run=true \
      module/network=mim_distillation_beats \
      paths.dataset_dir="$DATA_DIR" \
      data.dataset.parquet_path="$PARQUET" \
      module.network.encoder.pretrained_weights_path="$BEATS_CKPT"
fi

# --- 4. the no-training cells --------------------------------------------
# These need no adaptation and no queue, so they establish the floor and the
# frozen controls while the cluster jobs are still pending.
if runs probes; then
  banner "PROBE  C0: MFCC floor"
  python probe_selfdistill.py source=mfcc \
      paths.dataset_dir="$DATA_DIR" data.dataset.parquet_path="$PARQUET"

  banner "PROBE  C2: Bird-MAE frozen"
  python probe_selfdistill.py source=frozen \
      paths.dataset_dir="$DATA_DIR" data.dataset.parquet_path="$PARQUET" \
      module.network.encoder.pretrained_weights_path="$BIRDMAE_CKPT"

  banner "PROBE  C1: BEATs frozen"
  python probe_selfdistill.py source=frozen \
      module/network=mim_distillation_beats \
      paths.dataset_dir="$DATA_DIR" data.dataset.parquet_path="$PARQUET" \
      module.network.encoder.pretrained_weights_path="$BEATS_CKPT"
fi

# --- 5. optional: a real local run ---------------------------------------
if runs train; then
  banner "TRAIN  Bird-MAE (local)"
  python train_selfdistill.py \
      paths.dataset_dir="$DATA_DIR" \
      data.dataset.parquet_path="$PARQUET" \
      module.network.encoder.pretrained_weights_path="$BIRDMAE_CKPT"
fi

banner "DONE"
