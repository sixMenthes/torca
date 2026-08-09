#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the post-renormalisation sweep. RUN THIS ON A LOGIN NODE; it is a
# submitter, not a job, and it calls sbatch once per cell.
#
#   bash slurm/selfdistill/launch_renorm.sh              # everything, 8 jobs
#   bash slurm/selfdistill/launch_renorm.sh presentation # the 6 that matter, no K240
#   DRY_RUN=1 bash slurm/selfdistill/launch_renorm.sh    # print, submit nothing
#
# WHY THIS SWEEP EXISTS. Until 2026-08-10 the datamodule built its fbank front-end
# without passing mean or std, so every self-distillation run standardised with
# KaldiFbank's hard-coded -45.198 / 14.855 whatever the dataset config said, while the
# OFFLINE probe took a different route: it hands raw waveforms to the encoder, which
# runs its own front-end built from `encoder.fbank_mean`, and that one did interpolate
# the dataset value. Training and probing therefore standardised the same audio
# differently, by 0.83 units on average. Nothing errored, because both halves were
# individually valid. Every Bird-MAE checkpoint from before that date is unusable and
# every cell here is a replacement for one of them.
#
# The pre-fix runs at seeds 59 and 17 are discarded WHOLESALE and results-blind, which
# is worth saying out loud: no seed was dropped for having produced a bad number, so the
# spread across the seeds below is an honest estimate of run-to-run variance rather than
# one conditioned on avoiding known-bad draws.
#
# The BEATs arm is included but was never affected by the bug — its front-end class
# defaults happened to be BEATs' own constants, so the wrong dataset value was inert.
# It is here because it has never been trained at all.
# ---------------------------------------------------------------------------
set -euo pipefail

ACCOUNT="${ACCOUNT:-def-ruthjoy}"
MAX_EPOCHS="${MAX_EPOCHS:-36}"
# `VAL_EVERY` must DIVIDE `MAX_EPOCHS`. Lightning validates when (epoch + 1) % N == 0 and
# model_checkpoint only writes on an epoch that validates, so a value that does not
# divide silently discards the tail of every run. Checked below rather than trusted.
VAL_EVERY="${VAL_EVERY:-6}"
SEEDS="${SEEDS:-11 17}"
DRY_RUN="${DRY_RUN:-0}"
WHICH="${1:-all}"

# The recipe under test, identical across every cell so that arm and codebook size are
# the only things that vary. These are the settings the hinge runs used.
MASK_STRATEGY=bands
MASK_RATIO=0.7
DIVERSITY_WEIGHT=1.0
DIVERSITY_FLOOR=0.7

cd "$(dirname "$0")/../.."
REPO="$PWD"
echo "repo: $REPO"

# --- refuse to submit against an un-pulled tree ---------------------------
# Every guard here corresponds to a failure that has actually happened in this study,
# and each one costs eight wasted jobs rather than one, which is why they are fatal
# instead of warnings.
fail=0
note() { echo "  BLOCKED: $1" >&2; fail=1; }

grep -q 'mean=mean, std=std' selfdistill_datamodule.py \
  || note "selfdistill_datamodule.py does not pass mean/std to make_frontend — this is
           the pre-fix tree, so every run would reproduce the bug the sweep exists to
           undo. Pull before submitting."

grep -qE '^mean: -7\.2$' configs/data/dataset/dclde_selfdistill_birdmae.yaml \
  || note "the Bird-MAE dataset config does not carry mean: -7.2"

grep -qE '^mean: 15\.41663$' configs/data/dataset/dclde_selfdistill_beats.yaml \
  || note "the BEATs dataset config does not carry mean: 15.41663 — with the front-end
           now reading it, a stale -45.198 here would be APPLIED rather than ignored"

for key in mask_strategy diversity_floor max_time_band_frac p_freq_band; do
  grep -q "^  ${key}:" configs/module/network/mim_distillation_beats.yaml \
    || note "mim_distillation_beats.yaml has no '${key}' — selfdistill.sh passes it as a
             plain Hydra override, which fails at composition time against a key the
             config does not define, so the BEATs jobs would die before the first step"
done

if [ $(( MAX_EPOCHS % VAL_EVERY )) -ne 0 ]; then
  note "VAL_EVERY=$VAL_EVERY does not divide MAX_EPOCHS=$MAX_EPOCHS, so the final epoch
        would train and never checkpoint"
fi

[ "$fail" -eq 0 ] || { echo "nothing submitted." >&2; exit 1; }
echo "preflight checks passed."

# The SBATCH --output directive is resolved before the job script runs, so selfdistill.sh
# creating this directory itself is too late on a clean checkout.
mkdir -p logs/slurm

# --- the cells, in priority order -----------------------------------------
# Ordered SEED-MAJOR on purpose. If the queue only gets through part of this, a complete
# set of all three presentation cells at one seed is worth more than two seeds of one
# cell: the first gives a full table with no error bars, the second gives error bars on
# a table you cannot show.
#
# Fields are arm | seed | levels (empty = the config's [8,5,5,5], K=1000) | what it is for.
CELLS=()
for seed in $SEEDS; do
  CELLS+=("birdmae|$seed||the headline adapted cell")
  CELLS+=("birdmae_nobg|$seed||attribution control: cross-hydrophone noise off")
  CELLS+=("beats|$seed||the second backbone")
done
if [ "$WHICH" != "presentation" ]; then
  # Last, and genuinely droppable. This is the one configuration with direct evidence of
  # not reproducing: its ecotype probe moved 0.407 to 0.340 between seeds 59 and 17,
  # a larger gap than the entire spread across all five adapted cells.
  for seed in $SEEDS; do
    CELLS+=("birdmae|$seed|[8,6,5]|smaller codebook, K=240")
  done
fi

# --- submit ---------------------------------------------------------------
n=0
ids=()
for cell in "${CELLS[@]}"; do
  IFS='|' read -r arm seed levels why <<< "$cell"
  n=$(( n + 1 ))
  printf '%d/%d  %-14s seed=%-3s %-9s %s\n' \
    "$n" "${#CELLS[@]}" "$arm" "$seed" "${levels:-K=1000}" "$why"

  if [ "$DRY_RUN" = "1" ]; then
    continue
  fi

  # LEVELS is exported through the environment rather than --export, because --export
  # takes a comma-separated list and would split '[8,6,5]' on its inner commas.
  out=$(LEVELS="$levels" \
        MAX_EPOCHS="$MAX_EPOCHS" \
        VAL_EVERY="$VAL_EVERY" \
        SEED="$seed" \
        MASK_STRATEGY="$MASK_STRATEGY" \
        MASK_RATIO="$MASK_RATIO" \
        DIVERSITY_WEIGHT="$DIVERSITY_WEIGHT" \
        DIVERSITY_FLOOR="$DIVERSITY_FLOOR" \
        sbatch --account="$ACCOUNT" slurm/selfdistill/selfdistill.sh "$arm")
  echo "      $out"
  ids+=("${out##* }")
done

if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "DRY_RUN=1, nothing submitted."
  exit 0
fi

echo
echo "submitted ${#ids[@]} jobs: ${ids[*]}"
echo
echo "watch the queue:   squeue -u \$USER -o '%.10i %.30j %.2t %.10M %.10L %R'"
echo "follow the first:  tail -f logs/slurm/selfdistill_${ids[0]}.out"
echo
echo "CHECK THIS LINE in the first job's log before trusting any of them:"
echo "  fbank front-end: BirdMAE @ 32000 Hz, standardising with mean=-7.2, std=4.43"
echo "If it says -45.198, the tree on the cluster is stale and all ${#ids[@]} are wasted."
