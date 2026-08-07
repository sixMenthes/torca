#!/bin/bash
# ---------------------------------------------------------------------------
# Everything that can fail WITHOUT a GPU, checked on a LOGIN NODE in a minute.
#
#     bash slurm/selfdistill/preflight.sh            # birdmae arm
#     bash slurm/selfdistill/preflight.sh beats
#
# Run this before every sbatch. A missing dependency or a config typo kills the
# job seconds after it reaches the front of the queue, and the whole wait is paid
# again to find out — twice already for hydra-colorlog and mlflow, each costing
# more wall-clock than the run itself would have.
#
# Three stages, cheapest first:
#   1. imports        every module the training path needs
#   2. composition    hydra resolves the full config with the cluster overrides
#   3. one batch      real data, real model, CPU, fast_dev_run
#
# Stage 3 reads the LOOSE clip tree on $STAGE, not the tarball, so it needs the
# loose copy to still exist — one more reason not to run REMOVE_LOOSE=1 until a
# GPU job has succeeded end to end.
#
# Cheap enough for a login node: fast_dev_run is a single train batch and a single
# val batch, two workers, on CPU. Do not run anything larger here.
# ---------------------------------------------------------------------------
set -uo pipefail

ARM="${1:-birdmae}"

# ============================== USER SETTINGS ==============================
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/def-XXXX/$USER/torca}"
VENV="${VENV:-$HOME/.torca_venv}"
DATA_ROOT="${DATA_ROOT:-$HOME/projects/def-XXXX/$USER}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT}"
STAGE="${STAGE:-$SCRATCH/selfdistill_stage}"
# ==========================================================================

case "$ARM" in
  birdmae) NETWORK="mim_distillation";       DATASET="dclde_selfdistill_birdmae"
           CKPT="$DATA_ROOT/Bird-MAE-B" ;;
  beats)   NETWORK="mim_distillation_beats"; DATASET="dclde_selfdistill_beats"
           CKPT="$DATA_ROOT/BEATs_iter3.pt" ;;
  *) echo "unknown arm '$ARM' (birdmae | beats)" >&2; exit 1 ;;
esac

module load StdEnv/2023 python/3.11 gcc arrow/22.0.0
source "$VENV/bin/activate"
export PROJECT_ROOT OUTPUT_DIR HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
# Same as selfdistill.sh: the Alliance mlflow wheel refuses a file-backed tracking store
# without this. It belongs here too — stage 3 instantiates the real logger, so without it
# the preflight fails on something the job would also have failed on, which is the point,
# but it fails for a reason you then have to fix in two places.
export MLFLOW_ALLOW_FILE_STORE=true
cd "$PROJECT_ROOT"

echo "=== 1. imports ==="
# hydra plugins live under the hydra_plugins NAMESPACE — `import hydra_colorlog`
# fails even when the package is correctly installed, so check the real path.
python - <<'PY' || { echo "FAILED: install what is missing, then re-run" >&2; exit 1; }
import importlib, sys
mods = ["torch", "torchaudio", "lightning", "pytorch_lightning", "mlflow",
        "hydra", "omegaconf", "polars", "soundfile", "numpy", "sklearn", "tqdm"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m}: {e}")
for m in missing:
    print("  MISSING", m)
print(f"  {len(mods) - len(missing)}/{len(mods)} ok")
sys.exit(1 if missing else 0)
PY

echo
echo "=== 2. config composition ($ARM) ==="
# --cfg job resolves the whole config and exits without running anything. Catches
# a bad override, a missing group, an unresolvable interpolation.
python train_selfdistill.py --cfg job --resolve \
    paths=cluster hydra=cluster \
    module/network="$NETWORK" data/dataset="$DATASET" trainer=single_gpu \
    paths.dataset_dir="$STAGE/data" \
    data.dataset.parquet_path="$PROJECT_ROOT/ds/DCLDE_w_Buzzes.parquet" \
    module.network.encoder.pretrained_weights_path="$CKPT" \
    task_name="preflight_$ARM" >/dev/null \
  || { echo "FAILED: config does not compose" >&2; exit 1; }
echo "  composes"

echo
echo "=== 3. one batch on CPU ($ARM) ==="
[ -d "$STAGE/data" ] || { echo "no loose tree at $STAGE/data — skipping" >&2; exit 0; }
python train_selfdistill.py \
    paths=cluster hydra=cluster \
    +trainer.fast_dev_run=true trainer.accelerator=cpu trainer.precision=32 \
    module/network="$NETWORK" data/dataset="$DATASET" trainer=single_gpu \
    paths.dataset_dir="$STAGE/data" \
    data.dataset.parquet_path="$PROJECT_ROOT/ds/DCLDE_w_Buzzes.parquet" \
    module.network.encoder.pretrained_weights_path="$CKPT" \
    data.loaders.train.num_workers=2 data.loaders.val.num_workers=2 \
    task_name="preflight_$ARM" \
  || { echo "FAILED: the run itself is broken, not the queue" >&2; exit 1; }

echo
echo "=== preflight passed — safe to sbatch ==="
