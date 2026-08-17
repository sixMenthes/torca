#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the SEALED probes for the post-renormalisation sweep. RUN THIS ON A LOGIN
# NODE; it is a submitter, not a job, and it calls sbatch once per cell.
#
#   bash slurm/selfdistill/launch_probes_renorm.sh              # everything
#   bash slurm/selfdistill/launch_probes_renorm.sh presentation # the 6 adapted + 3 frozen
#   bash slurm/selfdistill/launch_probes_renorm.sh bg07         # just the p=0.7 arm + C2
#   DRY_RUN=1 bash slurm/selfdistill/launch_probes_renorm.sh    # print, submit nothing
#
# It is the companion to launch_renorm.sh: that script trains the eight cells, this one
# probes them, and the two agree on the recipe strings so that the checkpoint paths can
# be RECONSTRUCTED rather than guessed. If you change the recipe in one, change it in
# the other, or this script will not find the checkpoints and will refuse to submit.
#
# WHY THE FROZEN CELLS ARE IN HERE. The sealed protocol scores ecotype by GroupKFold
# within train, which is a different estimator from the reported fit-on-train,
# score-on-test figure. An adapted sealed number compared against a frozen REPORTED
# number therefore measures the change of estimator rather than the effect of
# adaptation. C2 (frozen Bird-MAE) has been probed sealed and C1 (frozen BEATs) has
# NOT, so before this script existed the BEATs arm had no comparator at all and its
# adapted number would have meant nothing on its own.
#
# The frozen cells were not affected by the normalisation bug, because a frozen cell
# never trains and so only ever has the encoder's own front-end. They are re-run here
# anyway: they are one job each, they are the denominator of every claim in the table,
# and running them in the same batch as the adapted cells removes any question about
# which version of the tree produced them.
# ---------------------------------------------------------------------------
set -euo pipefail

# The allocation, and through it every default path below. A PLACEHOLDER on purpose,
# the same one selfdistill.sh and probe_sealed.sh carry, so the real account name stays
# out of the checkout. Set it in the environment:
#
#   ACCOUNT=def-yourpi SEAL=0 bash slurm/selfdistill/launch_probes_renorm.sh
#
# Leaving it unset is a hard error rather than a silent wrong path — see the guard
# below. That distinction is not hypothetical: an unresolved placeholder is what killed
# the first eleven jobs, and it did so four seconds into an allocation rather than here.
ACCOUNT="${ACCOUNT:-def-XXXX}"
SEEDS="${SEEDS:-11 17}"
DRY_RUN="${DRY_RUN:-0}"
WHICH="${1:-all}"
# Write the pooled features out for project_embeddings.py. On by default; SAVE_FEATURES=0
# turns it off. Passed to the jobs explicitly rather than left to their own default, for
# the same reason the paths are.
SAVE_FEATURES="${SAVE_FEATURES:-1}"
# SEAL=0 spends the test split. See the block at the top of probe_sealed.sh; the short
# version is that it is the number the study claims and it can be run once.
SEAL="${SEAL:-1}"

# The #SBATCH --time in probe_sealed.sh is 40 minutes, and that was sized for the SEALED
# protocol: 10 logistic-regression fits, five for ecotype and five for hydrophone.
#
# The reported protocol asks for 47. The ecotype probe gains a nested C selection, which
# is four C values by five folds, so 21 fits instead of 5; the call-type probe stops
# being skipped and costs another 21; and the pool grows by a third, because sealing
# dropped the val and test rows before extraction and now it does not. That is roughly
# four times the sklearn work and, after six-way parallelism over folds, about three
# times the wall-clock.
#
# A command-line --time overrides the #SBATCH directive, so the request is raised here
# rather than by editing the job script, which would slow the cheap sealed runs too.
WALLTIME="${WALLTIME:-}"
if [ -z "$WALLTIME" ]; then
  if [ "$SEAL" = "1" ]; then WALLTIME=00:40:00; else WALLTIME=01:45:00; fi
fi

# These MUST match launch_renorm.sh. They are not levers here; they are how the
# task_name, and therefore the checkpoint path, is spelled.
MAX_EPOCHS="${MAX_EPOCHS:-36}"
MASK_STRATEGY=bands
MASK_RATIO=0.7
DIVERSITY_WEIGHT=1.0
DIVERSITY_FLOOR=0.7

cd "$(dirname "$0")/../.."
REPO="$PWD"
DATA_ROOT="${DATA_ROOT:-$HOME/projects/$ACCOUNT/$USER}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT}"
# PROJECT_ROOT is the checkout this script is running from, which is also the checkout
# whose configs the guards below inspect. Passing it to the jobs rather than letting
# them work it out closes a gap that has already cost a whole batch: probe_sealed.sh
# defaults PROJECT_ROOT and DATA_ROOT to $HOME/projects/def-XXXX/$USER, a placeholder
# that only ever resolved because the real value arrived from the submitting shell's
# environment. The first eleven jobs were submitted with an explicit --export list,
# that inheritance stopped, and all seven that started died in four seconds at
# `cd: /home/sixmints/projects/def-XXXX/sixmints/torca: No such file or directory`.
#
# It also removes a subtler hazard. The guards below check THIS tree, so a job that
# resolved a different PROJECT_ROOT would be running code nothing had validated.
PROJECT_ROOT="$REPO"
echo "repo:    $REPO"
echo "runs in: $OUTPUT_DIR/runs"

# --- refuse to probe with the wrong normalisation -------------------------
# The probe hands raw waveforms to the encoder, so it standardises with
# encoder.fbank_mean, which interpolates the dataset config. If that config is stale
# the probe reproduces exactly the train/probe mismatch this whole sweep exists to
# undo, and it does so silently.
fail=0
note() { echo "  BLOCKED: $1" >&2; fail=1; }

[ "$ACCOUNT" != "def-XXXX" ] \
  || note "ACCOUNT is still the placeholder def-XXXX. Every default path below is built
           from it, so nothing would resolve. Re-run as:
             ACCOUNT=def-yourpi SEAL=$SEAL bash \$0 ${WHICH#all}"

grep -qE '^mean: -7\.2$' configs/data/dataset/dclde_selfdistill_birdmae.yaml \
  || note "the Bird-MAE dataset config does not carry mean: -7.2, so the probe would
           standardise differently from the training that produced these checkpoints"

grep -qE '^mean: 15\.41663$' configs/data/dataset/dclde_selfdistill_beats.yaml \
  || note "the BEATs dataset config does not carry mean: 15.41663"

# The training runs themselves log which constants they used. If any of them says
# -45.198, that run trained on the pre-fix tree and probing it is pointless. Pre-fix
# runs print no such line at all, so an old log in this directory cannot trip this.
if compgen -G "logs/slurm/*.out" > /dev/null; then
  if grep -h 'fbank front-end:' logs/slurm/*.out 2>/dev/null | grep -q -- '-45.198'; then
    note "a job log in logs/slurm reports 'standardising with mean=-45.198' — at least
          one of these checkpoints was trained on the pre-fix tree"
  fi
fi

# Everything the job itself checks and exits on, checked HERE instead, on the login
# node, where the failure costs nothing. The job's own checks are four seconds into a
# scheduled allocation and report one cell at a time; these report all of them at once
# and before anything is queued.
[ -d "$PROJECT_ROOT" ] \
  || note "PROJECT_ROOT does not exist: $PROJECT_ROOT"
[ -d "$OUTPUT_DIR" ] \
  || note "OUTPUT_DIR does not exist: $OUTPUT_DIR — check that ACCOUNT=$ACCOUNT is right,
           since it is what the default path is built from"
[ -f "$DATA_ROOT/dclde_clips_3s.tar" ] \
  || note "the clip tarball is not at $DATA_ROOT/dclde_clips_3s.tar"
[ -e "$DATA_ROOT/Bird-MAE-B" ] \
  || note "the Bird-MAE backbone is not at $DATA_ROOT/Bird-MAE-B"
[ -e "$DATA_ROOT/BEATs_iter3.pt" ] \
  || note "the BEATs backbone is not at $DATA_ROOT/BEATs_iter3.pt"
[ -f "$PROJECT_ROOT/ds/DCLDE_w_Buzzes.parquet" ] \
  || note "the manifest is not at $PROJECT_ROOT/ds/DCLDE_w_Buzzes.parquet"

[ "$fail" -eq 0 ] || { echo "nothing submitted." >&2; exit 1; }
echo "config checks passed.  walltime request: $WALLTIME"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "DATA_ROOT:    $DATA_ROOT"

# Spending the test split is a one-way door, so it announces itself rather than being
# inferable from an environment variable nobody re-reads.
if [ "$SEAL" != "1" ]; then
  echo
  echo "*** REPORTED PROTOCOL: this batch SCORES ON THE HELD-OUT TEST HYDROPHONES. ***"
  echo "    Every cell below is being spent. The table you present must be the one you"
  echo "    chose BEFORE this ran, because picking the best row afterwards is model"
  echo "    selection on the test split and the number stops being evidence."
  echo
fi

mkdir -p logs/slurm

# --- reconstruct a training run's checkpoint path -------------------------
# selfdistill.sh builds task_name as selfdistill_<arm><variant>, appending each lever
# in a fixed order. Reproducing that order here is what lets us name a checkpoint
# without being told where it is. The output tree below task_name is
# <dataset_name>/<model_name>/<timestamp>/checkpoints, and the timestamp is not
# predictable, so that part is globbed and the newest match wins.
ckpt_for() {
  local arm="$1" seed="$2" levels="$3"
  local variant="_div${DIVERSITY_WEIGHT}"
  [ "$MAX_EPOCHS" != "18" ] && variant="${variant}_e${MAX_EPOCHS}"
  variant="${variant}_s${seed}_floor${DIVERSITY_FLOOR}_${MASK_STRATEGY}_m${MASK_RATIO}"
  if [ -n "$levels" ]; then
    # K = prod(levels), spelled the same way selfdistill.sh spells it.
    local K
    K=$(awk -v s="$levels" 'BEGIN{gsub(/[][ ]/,"",s); n=split(s,a,","); K=1;
        for(i=1;i<=n;i++) K*=a[i]; print K}')
    variant="${variant}_K${K}"
  fi
  local task="selfdistill_${arm}${variant}"
  # Newest last.ckpt under this task_name. `ls -t` over a glob rather than `find`,
  # because a partially written checkpoint from a still-running job would otherwise
  # be indistinguishable from a finished one by name alone.
  #
  # The `|| true` is load-bearing. `ls` exits non-zero when the glob matches nothing, and
  # this script runs with `set -o pipefail` under `set -e`, so without it the empty case
  # killed the script at the `ckpt=$(ckpt_for ...)` assignment — silently, with status 2,
  # before reaching the MISSING report below. That made the all-or-nothing guard
  # unreachable in precisely the situation it exists for.
  ls -t "$OUTPUT_DIR"/runs/"$task"/*/*/*/checkpoints/last.ckpt 2>/dev/null | head -1 || true
}

# --- the cells -------------------------------------------------------------
# Fields are label | arm-for-probe | seed | levels | what it is for.
#
# The label becomes the MLflow cell tag and must be unique per checkpoint: two models
# under one label are distinguishable only by timestamp, which is exactly the confusion
# the C4-against-C5 warning in probe_sealed.sh is about.
#
# `birdmae_nobg` probes as ARM=birdmae. The two differ only in a training-time
# augmentation probability, so they share a backbone, a network config and a front-end,
# and probing the control on the beats path would be a category error.
CELLS=()

if [ "$WHICH" = "bg07" ]; then
  # THE ATTRIBUTION CONTROL AT ITS HIGH DOSE, on its own. The other six adapted cells are
  # already trained and already probed, and re-running them would spend GPU hours
  # reproducing numbers the results table already carries.
  #
  # C2 is kept because it is the denominator of the claim and it is one 40-minute job.
  # Its sealed value is already known to be 0.912259356862241, so the new run doubles as
  # a check that this tree produces the same comparator as the tree that made the table:
  # if it comes back different, something in the front-end path has moved and the p=0.7
  # numbers cannot be compared against the existing rows either. Set WITH_C2=0 to skip it
  # and accept the stored value instead.
  #
  # Note that this puts a THIRD C2_probe_sealed run in the store. Two already exist and
  # they disagree — 0.9019737814832183 from 2026-08-09 08:20 and 0.912259356862241 from
  # 11:33 — because the fbank renormalisation landed between them. The new one should
  # match the later value. Read them by timestamp, never by label alone.
  [ "${WITH_C2:-1}" = "1" ] \
    && CELLS+=("C2|birdmae|||frozen Bird-MAE, the comparator and a tree check")
  for seed in $SEEDS; do
    CELLS+=("R_bg07_s${seed}|birdmae|$seed||attribution control at p = 0.7, the high dose")
  done
else

# The frozen comparators first: they gate the interpretation of everything else, and if
# the queue stalls halfway, a table of adapted numbers with no denominator is worthless.
CELLS+=("C2|birdmae|||frozen Bird-MAE, the comparator for birdmae and nobg")
CELLS+=("C1|beats|||frozen BEATs, the comparator for the beats arm — never run sealed")
CELLS+=("C0|birdmae|||MFCC floor, the same estimator as everything above")

for seed in $SEEDS; do
  CELLS+=("R_birdmae_s${seed}|birdmae|$seed||the headline adapted cell")
  CELLS+=("R_nobg_s${seed}|birdmae|$seed||attribution control: cross-hydrophone noise off")
  CELLS+=("R_beats_s${seed}|beats|$seed||the second backbone")
done

if [ "$WHICH" != "presentation" ]; then
  for seed in $SEEDS; do
    CELLS+=("R_birdmae_K240_s${seed}|birdmae|$seed|[8,6,5]|smaller codebook, K=240")
  done
fi

fi

# --- resolve every checkpoint BEFORE submitting anything ------------------
# All or nothing. A half-submitted batch is the worst outcome here, because the missing
# cell is discovered hours later when the table is being assembled.
PLAN=()
missing=0
for cell in "${CELLS[@]}"; do
  IFS='|' read -r label arm seed levels why <<< "$cell"
  if [ -z "$seed" ]; then
    PLAN+=("$label|$arm||$why")   # a frozen cell needs no checkpoint
    continue
  fi
  # The nobg control trains under its own arm name, so its path spells nobg even
  # though it probes as birdmae.
  train_arm="$arm"
  [[ "$label" == R_nobg_* ]] && train_arm="birdmae_nobg"
  [[ "$label" == R_bg07_* ]] && train_arm="birdmae_bg07"
  ckpt=$(ckpt_for "$train_arm" "$seed" "$levels")
  if [ -z "$ckpt" ]; then
    echo "  MISSING: no last.ckpt for $label (arm=$train_arm seed=$seed levels=${levels:-K1000})" >&2
    missing=1
    continue
  fi
  PLAN+=("$label|$arm|$ckpt|$why")
done

if [ "$missing" -ne 0 ]; then
  echo >&2
  echo "Some checkpoints are absent. Either those jobs have not finished, or the recipe" >&2
  echo "levers at the top of this script no longer match launch_renorm.sh. Check with:" >&2
  echo "  ls -d $OUTPUT_DIR/runs/selfdistill_*" >&2
  echo "nothing submitted." >&2
  exit 1
fi

# --- submit ---------------------------------------------------------------
n=0
ids=()
for row in "${PLAN[@]}"; do
  IFS='|' read -r label arm ckpt why <<< "$row"
  n=$(( n + 1 ))
  printf '%d/%d  %-22s arm=%-8s %s\n' "$n" "${#PLAN[@]}" "$label" "$arm" "$why"
  [ -n "$ckpt" ] && printf '        %s\n' "$ckpt"

  if [ "$DRY_RUN" = "1" ]; then
    continue
  fi

  # Everything the job needs is named here rather than inherited. ARM selects the
  # backbone for any label outside C0-C6; SEAL and SAVE_FEATURES are the two levers;
  # and PROJECT_ROOT, DATA_ROOT and OUTPUT_DIR override placeholder defaults that
  # otherwise resolve to a directory that does not exist.
  #
  # An explicit --export list is what broke this the first time, because the real paths
  # had been arriving from the submitting shell's environment and the list stopped them.
  # The repair is not to go back to inheriting them — that only worked by luck of how
  # the shell happened to be set up — but to state them, so the job's behaviour is a
  # property of this command line and nothing else. --export splits on commas and none
  # of these values contain one.
  #
  # The job name carries the cell. Without it every row of `squeue` reads probe_sealed
  # and the only way to tell which cell a job ID belongs to is to open its log.
  EXPORTS="ALL,ARM=$arm,SEAL=$SEAL,SAVE_FEATURES=$SAVE_FEATURES"
  EXPORTS="$EXPORTS,PROJECT_ROOT=$PROJECT_ROOT,DATA_ROOT=$DATA_ROOT,OUTPUT_DIR=$OUTPUT_DIR"
  if [ -n "$ckpt" ]; then
    out=$(sbatch --account="$ACCOUNT" --time="$WALLTIME" --export="$EXPORTS" \
                 --job-name="pr_$label" \
                 slurm/selfdistill/probe_sealed.sh "$label" "$ckpt")
  else
    out=$(sbatch --account="$ACCOUNT" --time="$WALLTIME" --export="$EXPORTS" \
                 --job-name="pr_$label" \
                 slurm/selfdistill/probe_sealed.sh "$label")
  fi
  echo "        $out"
  ids+=("${out##* }")
done

if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "DRY_RUN=1, nothing submitted."
  exit 0
fi

echo
echo "submitted ${#ids[@]} probes: ${ids[*]}"
echo
echo "watch the queue:  squeue -u \$USER -o '%.10i %.30j %.2t %.10M %.10L %R'"
echo
echo "When they are all done, copy the metrics back — kilobytes, not gigabytes:"
echo "  rsync -av nibi:$OUTPUT_DIR/mlruns/ /home/tundra/claude/torca/mlruns_nibi/mlruns/"
