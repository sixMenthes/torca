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
# 3 hours, not 24. The 24 was a guess made against a 206k-clip pool before anything had
# been measured; capping each hydrophone at 10,000 took the pool to 51,072 and an
# eighteen-epoch run was then TIMED at 54 minutes, so 24 hours over-requests by a factor
# of roughly twenty-five.
#
# That is not free, and it is not about allocation accounting. Slurm backfills short jobs
# into gaps ahead of larger reservations, and a 24-hour job can only be backfilled into a
# 24-hour gap, so the request itself is what makes these sit in PD (Priority). Three hours
# leaves headroom for staging the tarball and for a slower arm while fitting into far more
# scheduling holes.
#
# Jobs ALREADY QUEUED keep the limit they were submitted with. Lower them in place with
#     scontrol update JobId=<id> TimeLimit=03:00:00
# which Slurm permits because it is a reduction.
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.out
##SBATCH --mail-user=XXXX@gmail.com
##SBATCH --mail-type=END,FAIL

set -euo pipefail

ARM="${1:-birdmae}"

# ============================== USER SETTINGS ==============================
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/def-XXXX/$USER/torca}"
VENV="${VENV:-$HOME/.torca_venv}"
# Everything that is not the repo lives directly under the allocation dir: the
# tarball, both backbones, and (via OUTPUT_DIR below) runs/ logs/ mlruns/.
DATA_ROOT="${DATA_ROOT:-$HOME/projects/def-XXXX/$USER}"
# paths/cluster.yaml reads this as `scratch_dir` and hangs runs/, logs/ and mlruns/
# off it — so this is the DIRECTORY THEY GO IN, not one of them. Naming it
# ".../runs" would give runs/runs/<task_name>/... The name `scratch_dir` is a
# misnomer inherited from the finetune configs; project space is deliberate here,
# because these checkpoints have to survive long enough to be rsynced back for the
# probes, and scratch is purged.
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT}"
TARBALL="${TARBALL:-$DATA_ROOT/dclde_clips_3s.tar}"
BIRDMAE_CKPT="${BIRDMAE_CKPT:-$DATA_ROOT/Bird-MAE-B}"
BEATS_CKPT="${BEATS_CKPT:-$DATA_ROOT/BEATs_iter3.pt}"
PARQUET="$PROJECT_ROOT/ds/DCLDE_w_Buzzes.parquet"

# Checkpoint cadence. model_checkpoint monitors val/loss, so a checkpoint can only be
# written on an epoch that VALIDATES.
#
# VAL_EVERY=6, and the 6 is chosen to DIVIDE MAX_EPOCHS. Lightning validates when
# (epoch + 1) % N == 0, so with 18 epochs a value of 5 would validate at 5, 10 and 15
# and never at 18 — the final epoch would train, never checkpoint, and be discarded. Any
# value that does not divide MAX_EPOCHS throws away the tail of the run silently. If you
# change one of these two numbers, check the other.
#
# It used to be 1, which was right when an epoch was ~6,200 steps. Capping the pool made
# it 1,596, so a crash now costs under an hour of redone work instead of several, and
# three validations over the run is enough insurance. It also makes validation cost
# negligible: ~100 s a time (the SSL val pass over 2,594 clips, the ecotype probe, and
# the code-usage probe over 4,243) is five minutes across the whole run, which is why
# code_usage_probe.every_n_epochs stays at 1 rather than being thinned further.
#
# 18, not 5, and the DEFAULT rather than something you pass at submit time. Two reasons.
#
# The number: 5 was chosen against a 189,838-clip pool. Capping each hydrophone at
# 10,000 took that to 51,072, which at batch 32 is 1,596 steps per epoch, so 18 epochs
# is ~28,700 steps against the ~29,700 the old setting gave. Same amount of training,
# and an epoch is now roughly four times faster, so the wall-clock is comparable too.
# Both the warmup (warmup_ratio 0.067) and the EMA ramp derive from
# trainer.estimated_stepping_batches, so they rescale to this automatically.
#
# The default: ALL THREE ARMS MUST TRAIN FOR THE SAME NUMBER OF EPOCHS. The study
# compares C3/C4/C5 against each other, so a run length that differs between them is a
# treatment nobody controlled. Passing MAX_EPOCHS on the sbatch line for the first arm
# and forgetting it on the second is a one-keystroke way to silently ruin the ablation,
# and it would not show up anywhere in the results. Put it here, where the three
# submissions cannot disagree.
MAX_EPOCHS="${MAX_EPOCHS:-18}"
VAL_EVERY="${VAL_EVERY:-6}"
# Two batches through the val loader BEFORE training starts. The trainer config
# disables this, which was fine while every run had limit_val_batches=0 — but that
# means val_ssl_set has never actually been constructed or read. A broken val path
# should cost seconds at job start, not a full epoch.
SANITY_STEPS="${SANITY_STEPS:-2}"
# Objective reweighting, for comparison runs against the network config's default.
# Empty means "whatever mim_distillation.yaml says", which is 1.0 and is the setting the
# first eighteen-epoch run used. Set it to compare:
#
#   sbatch --export=ALL,DIVERSITY_WEIGHT=0.3 --account=def-XXXX \
#          slurm/selfdistill/selfdistill.sh birdmae
#
# Why 0.3 is the comparison worth running, measured on the 1.0 run rather than argued:
# CodeUsageProbe reported val/code_bg_token_bits 9.60 against val/code_call_token_bits
# 9.49, out of a log2(1000)=9.97 ceiling, with all 1000 codes used by Background clips
# alone. The codebook spends as much capacity representing NOISE as representing calls,
# slightly more in fact, and a codebook with nothing held in reserve is the mechanism by
# which spare capacity ends up encoding recording condition. Lowering the entropy penalty
# is the direct lever on that.
#
# RESULT, and it did not work. The full trajectory says the codebook does not "drift"
# open at all: token_bits_frac is 0.751 after epoch 1 and 0.985 after epoch 2, then flat
# for sixteen more. At 0.3 it is 0.696 and 0.958, and it converges to 0.965 against
# 0.976. Cutting the weight by seventy percent moved the endpoint by one percent, so the
# entropy penalty is not what holds the codebook open in this range. Worse for the
# hypothesis, site information in the codes went UP: val/code_bg_site_mi_excess ended at
# 0.727 bits against 0.511 at weight 1.0. The pretext task did get easier (masked_acc
# 0.201 against 0.138, ce 8.86 against 11.44) but ecotype on the held-out site was flat
# at 0.709 against 0.722.
#
# If anything the data points the other way, since the run with MORE entropy pressure
# had less site information in its codes. A weight of 3.0 is the indicated next probe of
# this lever, not a lower one.
#
# Anything set here is appended to the task_name, so the two runs are distinguishable in
# MLflow by NAME and not only by their params column. That matters more than it sounds:
# the runs are otherwise identical down to the seed, which is fixed at 59 in
# configs/selfdistill.yaml, so the run list would otherwise show two entries differing
# only by timestamp.
DIVERSITY_WEIGHT="${DIVERSITY_WEIGHT:-}"
# Codebook size, the other lever on how much capacity there is to spend on channel.
# Empty means the network config's [8, 5, 5, 5], which is K = 1000.
#
#   LEVELS='[8,6,5]' sbatch --account=def-XXXX \
#          slurm/selfdistill/selfdistill.sh birdmae
#
# NOTE the form. --export takes a COMMA-SEPARATED list, so --export=ALL,LEVELS='[8,6,5]'
# is split by sbatch on the commas inside the brackets and the job receives a mangled
# value. Setting the variable in the submitting shell works instead, because sbatch
# defaults to --export=ALL and the job inherits the environment. Any lever whose value
# contains a comma has to be passed this way; the others can use either form.
#
# Why this is a lever on the confound. It is NOT an information cap: mutual information
# between site and code is bounded by log2(K), which is 7.91 bits at K=240, and the
# measured val/code_bg_site_mi_excess is 0.51 bits, so the bound is nowhere near binding.
# The argument is softer than that. With fewer codes each one has to cover more acoustic
# variability, so the tokenizer must spend its vocabulary on whatever varies MOST, and a
# secondary factor like recording chain is the kind of thing that gets merged away. The
# supporting evidence is that K=240 and K=1000 recovered the same 4.1 to 4.5 bits about
# the teacher's code at 2000 steps, so the compression did not cost content. What that
# measurement cannot tell us is the site half, because CodeUsageProbe did not exist yet.
#
# TEMPERATURE MOVES WITH LEVELS AND IS DERIVED HERE. Logits are -d^2/tau, so tau has to
# scale with the squared spacing of the FINEST axis, which is 1/(max_d(levels_d // 2)).
# The reference point is [8,5,5,5], where that maximum is 4 and tau is 0.05, giving
#
#     tau = 0.05 * (4 / max_d(levels_d // 2))^2
#
# Any levels list containing an 8 therefore keeps tau at 0.05, and [8,6,5] is one of
# those. A list like [5,5,5] does not: its maximum is 2 and tau must go to 0.2. Getting
# this wrong does not fail, it just silently rescales the loss — which is exactly the
# trap documented at length in mim_distillation.yaml. Set TEMPERATURE explicitly to
# override the derivation.
LEVELS="${LEVELS:-}"
TEMPERATURE="${TEMPERATURE:-}"
# How the student's hidden positions are chosen, and how many of them there are.
#
#   sbatch --export=ALL,MASK_STRATEGY=bands,MASK_RATIO=0.7 --account=def-XXXX \
#          slurm/selfdistill/selfdistill.sh birdmae
#
# "random" scatters single 16x16 patches over the time-frequency grid, which is what
# every run before 2026-08-09 used. "bands" draws SpecAugment-shaped stripes instead,
# each one randomly a frequency band or a time band, until the ratio is covered.
#
# Why the shape matters and not just the amount: a scattered tile can often be filled in
# by interpolating its immediate neighbours, so the model can solve the task locally,
# whereas a stripe deletes a whole region and forces inference from surrounding context.
# The reason to want that here is the pattern across three runs, where every change that
# made the pretext task easier also raised the site information in the codes. Masking is
# channel-agnostic difficulty, which is the kind we want more of.
#
# Under "bands" the ratio is a TARGET. Measured on the real 19-by-8 patch grid, asking
# for 0.70 gives a mean coverage of 0.753 with a minimum of 0.704 and a maximum of 0.921,
# because stripes are coarse and the last one overshoots. train/mask_frac logs what
# actually happened.
MASK_STRATEGY="${MASK_STRATEGY:-}"
MASK_RATIO="${MASK_RATIO:-}"
# Turns the entropy penalty into a hinge that is zero above this coverage and linear
# below it, so it prevents collapse without pinning the codebook at its ceiling.
# Empty keeps the constant-pull form. See mim_distillation.yaml for why the two are
# different objectives rather than two strengths of one.
DIVERSITY_FLOOR="${DIVERSITY_FLOOR:-}"
# Replicate seed. Empty keeps configs/selfdistill.yaml's 59.
#
# This exists because every configuration in the study is n=1, and the fold standard
# deviation of the sealed probe measures only how much the PROBE wobbles given one
# checkpoint. It says nothing about how much a different initialisation and data order
# would move the checkpoint itself, which is a separate and unmeasured source of
# variance. With the ecotype spread across five adapted cells at 0.062 and the probe's
# own fold standard deviation between 0.06 and 0.10, the ranking of those cells is not
# currently distinguishable from noise, and one replicate per contender is what settles
# whether it is real.
#
# The frozen and MFCC control cells need no replicate: they never train, so there is no
# seed for them to depend on.
SEED="${SEED:-}"
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
  birdmae_teacherbg)
    # Cross-hydrophone noise on the TEACHER view as well, with an independent draw, so
    # the target codes are computed on a channel-mixed clip instead of a clean one.
    #
    # This is a design change, not a hyperparameter, and it is the direct test of why
    # C4's nuisance metric went the wrong way. With a clean teacher the target codes
    # carry site information (val/code_bg_site_mi_excess 0.51 bits against a 0.05 null,
    # on Background clips), so the cross-entropy pays the student for recovering the
    # hydrophone through the interference — channel recovery rather than channel
    # invariance. Mixing noise into the teacher's view removes that reward.
    #
    # Background only, not the student's full block: adding shift and gain here would
    # test three invariances at once and the result would not be attributable.
    #
    # Read the NUISANCE metric on this arm, not the loss. Independent noise draws make
    # part of the target unpredictable, so ce will be higher and masked_acc lower than
    # C4 by construction, and neither is a failure signal.
    NETWORK="mim_distillation";        DATASET="dclde_selfdistill_birdmae"
    CKPT="$BIRDMAE_CKPT"
    EXTRA=(data.transform.augmentations.teacher.background.p=0.8) ;;
  *)
    echo "ERROR: unknown arm '$ARM'" >&2
    echo "       (birdmae | beats | birdmae_nobg | birdmae_teacherbg)" >&2
    exit 1 ;;
esac

# Objective variants ride on top of the arm, not instead of it, so that a reweighted
# birdmae run is still the birdmae recipe in every other respect.
VARIANT=""
if [ -n "$DIVERSITY_WEIGHT" ]; then
  EXTRA+=("module.network.distill.diversity_weight=$DIVERSITY_WEIGHT")
  VARIANT="_div${DIVERSITY_WEIGHT}"
fi

# MAX_EPOCHS has a default rather than being empty, so it is compared against that
# default rather than against "". Without this, two runs differing only in length
# would share a task_name and land in the same output tree.
if [ "$MAX_EPOCHS" != "18" ]; then
  VARIANT="${VARIANT}_e${MAX_EPOCHS}"
fi

if [ -n "$SEED" ]; then
  EXTRA+=("seed=$SEED")
  VARIANT="${VARIANT}_s${SEED}"
fi

if [ -n "$DIVERSITY_FLOOR" ]; then
  EXTRA+=("module.network.distill.diversity_floor=$DIVERSITY_FLOOR")
  VARIANT="${VARIANT}_floor${DIVERSITY_FLOOR}"
fi

if [ -n "$MASK_STRATEGY" ]; then
  EXTRA+=("module.network.distill.mask_strategy=$MASK_STRATEGY")
  VARIANT="${VARIANT}_${MASK_STRATEGY}"
fi

if [ -n "$MASK_RATIO" ]; then
  EXTRA+=("module.network.distill.mask_ratio=$MASK_RATIO")
  VARIANT="${VARIANT}_m${MASK_RATIO}"
fi

if [ -n "$LEVELS" ]; then
  # K = prod(levels) and tau = 0.05 * (4 / max(levels_d // 2))^2, in one awk pass so the
  # two can never be derived from different lists.
  read -r K TAU_DERIVED <<< "$(awk -v s="$LEVELS" 'BEGIN{
      gsub(/[][ ]/, "", s); n = split(s, a, ",");
      K = 1; m = 0;
      for (i = 1; i <= n; i++) { K *= a[i]; h = int(a[i] / 2); if (h > m) m = h }
      if (n == 0 || m == 0) { print "0 0"; exit }
      printf "%d %.6g", K, 0.05 * (4.0 / m) ^ 2
  }')"
  if [ "$K" -le 1 ]; then
    echo "ERROR: could not parse LEVELS='$LEVELS' — expected a form like '[8,6,5]'" >&2
    exit 1
  fi
  TAU="${TEMPERATURE:-$TAU_DERIVED}"
  EXTRA+=("module.network.tokenizer.levels=$LEVELS")
  EXTRA+=("module.network.distill.temperature=$TAU")
  VARIANT="${VARIANT}_K${K}"
  echo "LEVELS=$LEVELS -> K=$K, temperature=$TAU$([ -n "$TEMPERATURE" ] && echo ' (explicit)' || echo ' (derived)')"
fi

date; hostname
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME  |  arm=$ARM  network=$NETWORK"
# An `if` rather than `[ ... ] && echo ...`: the AND-list form is exempt from set -e
# only by a subclause of its rules, and this script runs with -e.
if [ -n "$VARIANT" ]; then
  echo "VARIANT: diversity_weight=$DIVERSITY_WEIGHT -> task_name=selfdistill_${ARM}${VARIANT}"
fi

# --- environment ----------------------------------------------------------
module load StdEnv/2023 python/3.11 gcc arrow/22.0.0
source "$VENV/bin/activate"

export PROJECT_ROOT OUTPUT_DIR
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
# The Alliance mlflow wheel refuses a FILE-BACKED tracking store unless this is set.
# The guard is aimed at metric logging that creates huge numbers of small files on a
# shared parallel filesystem, which is a genuine problem there. Ours does not: the file
# store keeps ONE file per metric key and appends to it, so an 18-epoch run writes a few
# dozen files with a few hundred lines each. paths/cluster.yaml points mlflow_dir at
# ${paths.scratch_dir}/mlruns, i.e. project space, which is where the checkpoints have to
# live anyway so they survive to be rsynced back for the probes.
export MLFLOW_ALLOW_FILE_STORE=true
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
# paths=cluster hydra=cluster are NOT optional. selfdistill.yaml defaults to the
# workstation variants of both, so without these the job silently composes
# workstation paths: hydra writes runs under $PROJECT_ROOT/tests/runs and MLflow
# under $PROJECT_ROOT/mlruns, $OUTPUT_DIR is exported and then ignored, and
# paths/cluster.yaml is never read at all. Nothing errors — it just puts the
# checkpoints somewhere other than where this script says they go.
srun python train_selfdistill.py \
    paths=cluster \
    hydra=cluster \
    module/network="$NETWORK" \
    data/dataset="$DATASET" \
    trainer=single_gpu \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    trainer.max_epochs="$MAX_EPOCHS" \
    trainer.check_val_every_n_epoch="$VAL_EVERY" \
    trainer.num_sanity_val_steps="$SANITY_STEPS" \
    paths.dataset_dir="$DATA_DIR" \
    data.dataset.parquet_path="$PARQUET" \
    data.loaders.train.num_workers="$NWORKERS" \
    data.loaders.val.num_workers=2 \
    module.network.encoder.pretrained_weights_path="$CKPT" \
    task_name="selfdistill_${ARM}${VARIANT}" \
    "${EXTRA[@]}"

echo "Finished with exit code $?"
