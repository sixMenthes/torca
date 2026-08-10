#!/bin/bash
# ---------------------------------------------------------------------------
# One exploratory cell: the REGRESSION objective instead of the categorical one.
# RUN THIS ON A LOGIN NODE; it submits two chained jobs and is not itself a job.
#
#   ACCOUNT=def-yourpi bash slurm/selfdistill/launch_mse.sh
#   ACCOUNT=def-yourpi DRY_RUN=1 bash slurm/selfdistill/launch_mse.sh
#
# WHAT IT RUNS. Bird-MAE, K=240 ([8,6,5]), hinge diversity at floor 0.7, band masking
# at ratio 0.7, seed 59, 36 epochs — the standard recipe with ONE thing changed:
# mse_weight=1.0 and ce_weight=0.0, so the student regresses onto the teacher's code
# instead of classifying over the codebook. See masked_mse for what that trades away.
#
# It then chains the REPORTED probe onto it with --dependency=afterok, so the probe
# starts by itself when training finishes and does not run at all if training fails.
#
# TWO THINGS TO KNOW BEFORE READING THE RESULT.
#
# 1. SEED 59 HAS NO MATCHED CROSS-ENTROPY RUN. Every post-renormalisation cell was run
#    at seeds 11 and 17; 59 was only ever used before the fbank fix and those
#    checkpoints are discarded. So this cell can only be compared against K=240
#    cross-entropy cells at DIFFERENT seeds, and the seed spread on the reported
#    ecotype metric is 0.05 to 0.10, which is larger than most effects worth claiming.
#    Treat the result as a smell test, not a comparison. `SEED=11` or `SEED=17` here
#    would buy a matched pair for the same one job.
#
# 2. IT SPENDS THE TEST SPLIT AGAIN. The probe runs with SEAL=0, so this cell is
#    scored on StraitofGeorgia, CarmanahPt and BarkleyCanyon like the other eleven.
#    That is defensible only if it is reported alongside them rather than instead of
#    whichever of them it beats.
# ---------------------------------------------------------------------------
set -euo pipefail

ACCOUNT="${ACCOUNT:-def-XXXX}"
SEED="${SEED:-59}"
MAX_EPOCHS="${MAX_EPOCHS:-36}"
VAL_EVERY="${VAL_EVERY:-6}"
DRY_RUN="${DRY_RUN:-0}"
PROBE_WALLTIME="${PROBE_WALLTIME:-01:45:00}"

# The recipe, matching launch_renorm.sh so that only the objective differs.
LEVELS='[8,6,5]'
MASK_STRATEGY=bands
MASK_RATIO=0.7
DIVERSITY_WEIGHT=1.0
DIVERSITY_FLOOR=0.7

cd "$(dirname "$0")/../.."
REPO="$PWD"
DATA_ROOT="${DATA_ROOT:-$HOME/projects/$ACCOUNT/$USER}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT}"
PROJECT_ROOT="$REPO"

fail=0
note() { echo "  BLOCKED: $1" >&2; fail=1; }

[ "$ACCOUNT" != "def-XXXX" ] \
  || note "ACCOUNT is still the placeholder def-XXXX. Re-run as:
             ACCOUNT=def-yourpi bash \$0"
grep -q "mse_weight" configs/module/network/mim_distillation.yaml \
  || note "mim_distillation.yaml has no mse_weight — selfdistill.sh passes it as a plain
           Hydra override, which fails at composition time against an undefined key"
grep -q "def masked_mse" models/components/selfdistill_loss.py \
  || note "selfdistill_loss.py has no masked_mse; this is a pre-MSE tree, pull first"
grep -qE '^mean: -7\.2$' configs/data/dataset/dclde_selfdistill_birdmae.yaml \
  || note "the Bird-MAE dataset config does not carry mean: -7.2"
[ -d "$OUTPUT_DIR" ] || note "OUTPUT_DIR does not exist: $OUTPUT_DIR"
[ $(( MAX_EPOCHS % VAL_EVERY )) -eq 0 ] \
  || note "VAL_EVERY=$VAL_EVERY does not divide MAX_EPOCHS=$MAX_EPOCHS, so the final
           epoch would train and never checkpoint"

[ "$fail" -eq 0 ] || { echo "nothing submitted." >&2; exit 1; }

# selfdistill.sh spells task_name as selfdistill_<arm><variant>, appending each lever in
# a fixed order. Reproduced here so the probe can find the checkpoint; the _MSE suffix
# lands where the MSE_WEIGHT block puts it, after the seed and before the floor.
TASK="selfdistill_birdmae_div${DIVERSITY_WEIGHT}_e${MAX_EPOCHS}_s${SEED}_MSE"
TASK="${TASK}_floor${DIVERSITY_FLOOR}_${MASK_STRATEGY}_m${MASK_RATIO}_K240"

echo "repo:        $REPO"
echo "task_name:   $TASK"
echo "checkpoint:  $OUTPUT_DIR/runs/$TASK/*/*/*/checkpoints/last.ckpt"
echo "probe cell:  R_mse_K240_s${SEED}   (reported protocol, SEAL=0)"
echo

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1, nothing submitted."
  exit 0
fi

mkdir -p logs/slurm

# --- 1. train -------------------------------------------------------------
train_out=$(LEVELS="$LEVELS" \
      MAX_EPOCHS="$MAX_EPOCHS" VAL_EVERY="$VAL_EVERY" SEED="$SEED" \
      MASK_STRATEGY="$MASK_STRATEGY" MASK_RATIO="$MASK_RATIO" \
      DIVERSITY_WEIGHT="$DIVERSITY_WEIGHT" DIVERSITY_FLOOR="$DIVERSITY_FLOOR" \
      MSE_WEIGHT=1.0 CE_WEIGHT=0.0 \
      sbatch --account="$ACCOUNT" --job-name="mse_train_s${SEED}" \
             slurm/selfdistill/selfdistill.sh birdmae)
train_id="${train_out##* }"
echo "training:  $train_out"

# --- 2. probe, chained ----------------------------------------------------
# The checkpoint does not exist yet, so probe_sealed.sh cannot be handed a path at
# submit time — it validates the file and would refuse. CKPT_GLOB is resolved inside
# the job instead, once training has written it.
#
# afterok, not afterany: a failed training run must not be probed. Slurm cancels a
# dependent job whose dependency can never be satisfied, so a training failure leaves
# the probe in DependencyNeverSatisfied rather than producing a meaningless number.
probe_out=$(sbatch --account="$ACCOUNT" \
      --dependency="afterok:${train_id}" \
      --time="$PROBE_WALLTIME" \
      --job-name="mse_probe_s${SEED}" \
      --export="ALL,ARM=birdmae,SEAL=0,SAVE_FEATURES=1,PROJECT_ROOT=$PROJECT_ROOT,DATA_ROOT=$DATA_ROOT,OUTPUT_DIR=$OUTPUT_DIR,CKPT_GLOB=$OUTPUT_DIR/runs/$TASK/*/*/*/checkpoints/last.ckpt" \
      slurm/selfdistill/probe_sealed.sh "R_mse_K240_s${SEED}")
probe_id="${probe_out##* }"
echo "probe:     $probe_out   (waits for $train_id)"

echo
echo "watch:  squeue -u \$USER -o '%.10i %.20j %.2t %.10M %.10L %R'"
echo "logs:   logs/slurm/mse_train_s${SEED}_${train_id}.out"
echo "        logs/slurm/mse_probe_s${SEED}_${probe_id}.out"
echo
echo "The probe shows as DependencyNeverSatisfied if training fails. That is the"
echo "intended behaviour, not a second bug to chase."
