#!/bin/bash
# ---------------------------------------------------------------------------
# The six ablation cells, on the WORKSTATION, into ONE MLflow experiment.
#
#   bash slurm/selfdistill/probe_all.sh              # every cell it can run
#   bash slurm/selfdistill/probe_all.sh C0 C1 C2     # the no-training cells only
#   bash slurm/selfdistill/probe_all.sh C4 C5        # after the checkpoints land
#
#   C0  mfcc                                non-neural floor
#   C1  frozen  BEATs                       unadapted control
#   C2  frozen  BirdMAE                     unadapted control
#   C3  adapted BEATs                       needs CKPT_BEATS
#   C4  adapted BirdMAE                     needs CKPT_BIRDMAE
#   C5  adapted BirdMAE, background.p=0.0   needs CKPT_NOBG
#
# C0-C2 need no training and can run right now. C3-C5 need checkpoints rsynced back
# from the cluster; cells whose checkpoint is absent are SKIPPED with a warning
# rather than failing the batch, so this script is re-runnable as they arrive.
#
# All six land in the `selfdistill_probes` experiment and log identical metric keys,
# so the comparison is one sorted table. Read task/ecotype UP against nuisance/bg
# DOWN, jointly — task alone selects cells that adapted deeper into the confound.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="${REPO:-/home/tundra/claude/torca}"
# Adapted-cell checkpoints. Point these at what came back from Nibi; last.ckpt is
# fine, model_checkpoint also keeps the best-by-val/loss.
CKPT_BEATS="${CKPT_BEATS:-}"
CKPT_BIRDMAE="${CKPT_BIRDMAE:-}"
CKPT_NOBG="${CKPT_NOBG:-}"
# Frozen controls: the pretrained backbones, same paths the training arms used.
BIRDMAE_CKPT="${BIRDMAE_CKPT:-/data/torca_backbones/Bird-MAE-B}"
BEATS_CKPT="${BEATS_CKPT:-/data/torca_backbones/BEATs_iter3.pt}"
PY="${PY:-python}"

cd "$REPO"
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false

# Belt and braces alongside probe_selfdistill.py's file_system sharing strategy.
# systemd ships RLIMIT_NOFILE as soft 1024 / hard 524288 and an interactive shell
# inherits the soft one, which the dataloader exhausts partway through a 5.8k-batch
# pass. Raising it costs nothing and is not a privileged operation — any process may
# raise its soft limit up to the hard limit.
HARD=$(ulimit -Hn)
[ "$HARD" = "unlimited" ] && HARD=1048576
if [ "$(ulimit -n)" -lt "$HARD" ]; then
  ulimit -n "$HARD" 2>/dev/null && echo "raised open-file limit to $(ulimit -n)"
fi

CELLS=("$@")
[ ${#CELLS[@]} -eq 0 ] && CELLS=(C0 C1 C2 C3 C4 C5)

RAN=(); SKIPPED=(); FAILED=()

runs() { [[ " ${CELLS[*]} " == *" $1 "* ]]; }

# One place where a cell is turned into overrides. The alternative — six hand-typed
# command lines — is how C4 and C5 end up differing in something other than the
# checkpoint, which would make the attribution control useless.
probe() {
  local cell="$1"; shift
  echo
  echo "=================== $cell ==================="
  date
  if "$PY" probe_selfdistill.py cell="$cell" "$@"; then
    RAN+=("$cell")
  else
    echo "!!! $cell FAILED" >&2
    FAILED+=("$cell")
  fi
}

# Adapted cells: skip cleanly when the checkpoint has not arrived yet.
probe_adapted() {
  local cell="$1" ckpt="$2" var="$3"; shift 3
  if [ -z "$ckpt" ]; then
    echo "--- $cell skipped: \$$var not set"; SKIPPED+=("$cell"); return
  fi
  if [ ! -e "$ckpt" ]; then
    echo "--- $cell skipped: checkpoint not found at $ckpt"; SKIPPED+=("$cell"); return
  fi
  probe "$cell" source=adapted ckpt_path="$ckpt" "$@"
}

# --- C0: the floor --------------------------------------------------------
# task_name and the backbone tag are overridden because probe.yaml derives both from
# module.network, and MFCC has no backbone — left alone it would log itself as
# BirdMAE and sort next to the real Bird-MAE cells.
runs C0 && probe C0 source=mfcc task_name=probe_mfcc logger.tags.backbone=none

# --- C1/C2: frozen controls ----------------------------------------------
runs C1 && probe C1 source=frozen module/network=mim_distillation_beats \
    module.network.encoder.pretrained_weights_path="$BEATS_CKPT"
runs C2 && probe C2 source=frozen module/network=mim_distillation \
    module.network.encoder.pretrained_weights_path="$BIRDMAE_CKPT"

# --- C3/C4/C5: adapted ----------------------------------------------------
# module/network is still passed even though _build_encoder ignores it for
# source=adapted (the encoder comes out of the checkpoint). It sets the `backbone`
# tag and the logged `encoder` hyperparameter, which would otherwise say BirdMAE for
# the BEATs cell.
runs C3 && probe_adapted C3 "$CKPT_BEATS" CKPT_BEATS \
    module/network=mim_distillation_beats
runs C4 && probe_adapted C4 "$CKPT_BIRDMAE" CKPT_BIRDMAE \
    module/network=mim_distillation
runs C5 && probe_adapted C5 "$CKPT_NOBG" CKPT_NOBG \
    module/network=mim_distillation

# --- summary --------------------------------------------------------------
echo
echo "=================== SUMMARY ==================="
echo "ran     : ${RAN[*]:-none}"
echo "skipped : ${SKIPPED[*]:-none}"
echo "failed  : ${FAILED[*]:-none}"
echo
echo "compare all cells:  mlflow ui --backend-store-uri $REPO/mlruns"
echo "  experiment 'selfdistill_probes', sort by final/task_ecotype,"
echo "  and read it against final/nuisance_bg — the win is both at once."

[ ${#FAILED[@]} -eq 0 ] || exit 1
