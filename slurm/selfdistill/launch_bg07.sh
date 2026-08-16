#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the birdmae_bg07 arm: the attribution control at a high dose.
# RUN THIS ON A LOGIN NODE. It is a submitter, not a job, and it calls sbatch.
#
#   bash slurm/selfdistill/launch_bg07.sh calibrate   # ONE 1-epoch run, prices the rest
#   bash slurm/selfdistill/launch_bg07.sh submit      # the two real runs, seeds 11 and 17
#   bash slurm/selfdistill/launch_bg07.sh estimate    # energy and carbon, no jobs
#   DRY_RUN=1 bash slurm/selfdistill/launch_bg07.sh submit   # print, submit nothing
#
# WHY THIS ARM EXISTS. The recipe mixes noise from a DIFFERENT hydrophone into the
# student view, on the reasoning that a student forced to predict the teacher's codes
# through foreign interference must become invariant to the recording channel. Cell C5
# (birdmae_nobg) is the attribution control: it switches that augmentation off and asks
# whether anything changes.
#
# The control is under-powered. C5 was written on 2026-08-05, when the config carried
# background.p = 0.8. Commit 35bbbeb on 2026-08-09 cut the probability to 0.2, together
# with shift and gain from 0.5 to 0.05, and C5 was never re-scoped. So the control that
# reached the results table is a 0.2-against-0.0 contrast, and at that separation the
# mechanism metric measures nothing: site mutual information is 0.959 and 1.226 bits for
# C5 against 1.126 and 0.696 for C4, fully overlapping across the two seeds.
#
# This arm trains at p = 0.7 so that C5, C4 and C6 form a dose series of 0.0, 0.2 and 0.7
# on ONE lever with everything else held fixed.
#
# A NULL IS A RESULT. You are testing whether a large dose separates what a small one
# cannot. If nothing moves, that is reportable and it is the answer.
#
# WHAT TO READ. Nuisance, meaning hydrophone decodability from Background clips. Not the
# loss, and not ecotype. At p = 0.7 the student sees foreign noise on most views, so the
# pretext task is harder by construction and a higher cross-entropy or a lower masked
# accuracy is expected rather than a failure signal. Ecotype accuracy is expected to
# fall, which confounds de-confounding pressure with task difficulty, and is exactly why
# the arm is read as a trend across three doses instead of as one comparison.
# ---------------------------------------------------------------------------
set -euo pipefail

MODE="${1:-submit}"

# A PLACEHOLDER, matching selfdistill.sh, launch_renorm.sh and probe_sealed.sh, so the
# real allocation name stays out of the checkout. Set it in the environment:
#   ACCOUNT=def-yourpi bash slurm/selfdistill/launch_bg07.sh submit
ACCOUNT="${ACCOUNT:-def-XXXX}"

# 36 and 6 match the post-renormalisation sweep exactly. They are not defaults to be
# tuned here: C4 and C5 trained for 36 epochs, and an arm trained for a different number
# is not a dose series, it is a second treatment. selfdistill.sh itself defaults to 18,
# which is why this is passed explicitly on every submission below.
MAX_EPOCHS="${MAX_EPOCHS:-36}"
VAL_EVERY="${VAL_EVERY:-6}"
SEEDS="${SEEDS:-11 17}"
DRY_RUN="${DRY_RUN:-0}"

# Sample whole-device GPU power during the run. On by default for calibration, off for
# the real runs, because the sampler's value is in pricing the campaign rather than in
# the science and it writes a CSV line every 20 seconds.
POWER_SAMPLE="${POWER_SAMPLE:-auto}"

# The recipe under test, copied from launch_renorm.sh so that the ONLY difference between
# this arm and C4 is the background probability the arm itself sets.
MASK_STRATEGY=bands
MASK_RATIO=0.7
DIVERSITY_WEIGHT=1.0
DIVERSITY_FLOOR=0.7

ARM=birdmae_bg07

cd "$(dirname "$0")/../.."
REPO="$PWD"
echo "repo: $REPO"
echo "mode: $MODE"

# --- estimate mode leaves before any of the submission guards --------------
# It touches no allocation and needs no account, so gating it behind the preflight would
# only make it harder to run.
if [ "$MODE" = "estimate" ]; then
  STORE="${STORE:-mlruns_nibi}"
  [ -d "$STORE" ] || { echo "no run store at $STORE; set STORE=<path>" >&2; exit 1; }
  ARGS=(--mlruns "$STORE")
  [ -n "${PER_EPOCH:-}" ] && ARGS+=(--per-epoch "$PER_EPOCH")
  ARGS+=(--plan "${PLAN:-birdmae_bg07:2:$MAX_EPOCHS}")
  # Stdlib only, so the login node's bare python3 is enough and no module load or venv
  # activation is needed. PYTHON is here for the workstation, where a hook requires the
  # project venv.
  exec "${PYTHON:-python3}" estimate_emissions.py "${ARGS[@]}"
fi

# --- refuse to submit against an un-pulled or misconfigured tree -----------
# Every guard corresponds to a failure that has actually happened in this study. They are
# fatal rather than warnings because the cheapest of them costs two wasted GPU-hours and
# the most expensive silently produces a duplicate of a cell we already have.
fail=0
note() { echo "  BLOCKED: $1" >&2; fail=1; }

[ "$ACCOUNT" != "def-XXXX" ] \
  || note "ACCOUNT is still the placeholder def-XXXX. Re-run as:
             ACCOUNT=def-yourpi bash \$0 $MODE"

grep -q '^  birdmae_bg07)' slurm/selfdistill/selfdistill.sh \
  || note "selfdistill.sh has no birdmae_bg07 arm, so every job would exit 1 on the
           unknown-arm branch. Pull before submitting."

grep -q 'background.p=0.7' slurm/selfdistill/selfdistill.sh \
  || note "the birdmae_bg07 arm does not set background.p=0.7. A silently ignored or
           edited override produces a duplicate of C4 and NOTHING in the metrics would
           look wrong, which makes this the most expensive failure available here."

grep -q 'mean=mean, std=std' selfdistill_datamodule.py \
  || note "selfdistill_datamodule.py does not pass mean/std to make_frontend — this is
           the pre-fix tree, so the run would reproduce the normalisation bug the whole
           sweep exists to undo. Pull before submitting."

grep -qE '^mean: -7\.2$' configs/data/dataset/dclde_selfdistill_birdmae.yaml \
  || note "the Bird-MAE dataset config does not carry mean: -7.2"

if [ $(( MAX_EPOCHS % VAL_EVERY )) -ne 0 ]; then
  note "VAL_EVERY=$VAL_EVERY does not divide MAX_EPOCHS=$MAX_EPOCHS. Lightning only
        checkpoints on an epoch that validates, so the tail of the run would train and
        then be discarded without ever being written."
fi

[ "$fail" -eq 0 ] || { echo "nothing submitted." >&2; exit 1; }
echo "preflight checks passed."

# The SBATCH --output directive is resolved before the job script runs, so selfdistill.sh
# creating this directory itself would be too late on a clean checkout.
mkdir -p logs/slurm

# --- the cells -------------------------------------------------------------
# Fields are seed | epochs | what it is for.
CELLS=()
case "$MODE" in
  calibrate)
    # ONE epoch, ONE seed, purely to time an epoch on this arm so the campaign can be
    # priced before it is spent. VAL_EVERY must equal MAX_EPOCHS here or the single epoch
    # never validates and never checkpoints.
    #
    # This run lands under task_name selfdistill_birdmae_bg07_div1.0_e1_s11_..., because
    # selfdistill.sh appends _e<N> whenever MAX_EPOCHS differs from its own default of
    # 18. The real runs get _e36. The two therefore cannot collide in the output tree and
    # the calibration checkpoint can never be picked up by the probe launcher's glob.
    CELLS+=("11|1|one-epoch calibration, timing only — NOT a scientific run")
    VAL_EVERY=1
    WALLTIME="${WALLTIME:-00:40:00}"
    [ "$POWER_SAMPLE" = "auto" ] && POWER_SAMPLE=1
    ;;
  submit)
    for seed in $SEEDS; do
      CELLS+=("$seed|$MAX_EPOCHS|attribution control at p = 0.7, the high dose")
    done
    WALLTIME="${WALLTIME:-03:00:00}"
    [ "$POWER_SAMPLE" = "auto" ] && POWER_SAMPLE=0
    ;;
  *)
    echo "ERROR: unknown mode '$MODE'" >&2
    echo "       (calibrate | submit | estimate)" >&2
    exit 1 ;;
esac

echo "walltime request: $WALLTIME   power sampling: $POWER_SAMPLE"
echo

n=0
ids=()
for cell in "${CELLS[@]}"; do
  IFS='|' read -r seed epochs why <<< "$cell"
  n=$(( n + 1 ))
  printf '%d/%d  %-14s seed=%-3s epochs=%-3s %s\n' \
    "$n" "${#CELLS[@]}" "$ARM" "$seed" "$epochs" "$why"

  if [ "$DRY_RUN" = "1" ]; then
    continue
  fi

  # Levers go through the environment rather than --export. selfdistill.sh reads them
  # with ${VAR:-default}, and --export takes a comma-separated list that would split a
  # bracketed levels string on its inner commas — a trap this study has already hit.
  out=$(MAX_EPOCHS="$epochs" \
        VAL_EVERY="$VAL_EVERY" \
        SEED="$seed" \
        MASK_STRATEGY="$MASK_STRATEGY" \
        MASK_RATIO="$MASK_RATIO" \
        DIVERSITY_WEIGHT="$DIVERSITY_WEIGHT" \
        DIVERSITY_FLOOR="$DIVERSITY_FLOOR" \
        POWER_SAMPLE="$POWER_SAMPLE" \
        sbatch --account="$ACCOUNT" --time="$WALLTIME" \
               --job-name="sd_bg07_s${seed}" \
               slurm/selfdistill/selfdistill.sh "$ARM")
  echo "      $out"
  ids+=("${out##* }")
done

if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "DRY_RUN=1, nothing submitted."
  exit 0
fi

echo
echo "submitted ${#ids[@]} job(s): ${ids[*]}"
echo
echo "watch the queue:   squeue -u \$USER -o '%.10i %.30j %.2t %.10M %.10L %R'"
echo "follow the first:  tail -f logs/slurm/sd_bg07_s${SEEDS%% *}_${ids[0]}.out"
echo
echo "TWO LINES TO CHECK in the first log before trusting any of this:"
echo "  1. fbank front-end: BirdMAE @ 32000 Hz, standardising with mean=-7.2, std=4.43"
echo "     If it says -45.198, the cluster tree is stale and the runs are wasted."
echo "  2. the resolved data.transform.augmentations.student.background.p, which must"
echo "     read 0.7. A silently ignored override reproduces C4 and looks perfectly fine."
echo
echo "The value is ALSO recorded permanently in the run store, so it can be confirmed"
echo "after the fact rather than only from the log:"
echo "  cat \$OUTPUT_DIR/../mlruns/*/*/params/data/transform/augmentations/student/background/p"

if [ "$MODE" = "calibrate" ]; then
  echo
  echo "WHEN THE CALIBRATION RUN FINISHES, price the campaign from its one epoch:"
  echo "  PER_EPOCH=\$(<seconds for the single epoch>) \\"
  echo "    bash slurm/selfdistill/launch_bg07.sh estimate"
  echo "Take the seconds from emissions/duration_s on that run, or from the elapsed"
  echo "time in sacct: sacct -j ${ids[0]} --format=JobID,Elapsed,State"
fi
