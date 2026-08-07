"""Do we still need the VICReg terms now that the entropy penalty is on? 2x2, one table.

    python ablate_terms.py                  # 5 training steps per cell (plumbing only)
    python ablate_terms.py --steps 500      # enough to see the terms move
    python ablate_terms.py --steps 2000     # the budget the diversity sweep used
    python ablate_terms.py --device cpu

Four cells over the two guards, holding everything else fixed:

    ce            var_weight=0    cov_weight=0     diversity_weight=0
    ce+div        var_weight=0    cov_weight=0     diversity_weight=1
    ce+vic        var_weight=1    cov_weight=0.04  diversity_weight=0
    ce+vic+div    var_weight=1    cov_weight=0.04  diversity_weight=1

WHAT A SHORT RUN CAN AND CANNOT TELL YOU. At 5 steps this checks that every
combination runs and shows the RELATIVE SCALE of the terms, which is worth knowing on
its own: if one term is two orders of magnitude larger than another at initialisation,
its weight is not doing what its name suggests. It cannot answer the question in the
title. Collapse is a dynamic — the tokenizer walks onto a few codes over time — and the
sweep recorded in configs/module/network/mim_distillation.yaml needed 2000 steps to
separate the cells. Read a 5-step table as "nothing is broken", not as evidence.

WHAT TO READ, in order of importance:

    token_bits_frac   the real answer. Fraction of the log2(K) ceiling actually used,
                      so it is comparable across different `levels`, which raw bits and
                      codes_used are not. Collapse looks like this falling toward zero.
    codes_used        how many of the 1000 codes appear at all over the epoch. The
                      documented collapse signature is ~9; healthy was 673.
    ce                LOWER IS NOT BETTER HERE. A collapsed tokenizer scored 0.699
                      against a healthy 2.776, because predicting one of nine codes is
                      easy. Read it only alongside token_bits_frac.
    masked_acc        same trap, same direction: 0.821 collapsed vs 0.378 healthy.
    vic_var/vic_cov   the guards. The point of the ce cell is to see whether they read
                      healthy while the tokenizer collapses, which is what the network
                      config claims happens.

Deliberately writes no MLflow run and no checkpoint: four throwaway diagnostics do not
belong in the experiment store next to the real arms.
"""

import argparse
import os
import sys
from pathlib import Path

import lightning as L
from hydra import compose, initialize_config_dir

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
os.environ["PROJECT_ROOT"] = str(root)

from build_model import build_model                      # noqa: E402
from selfdistill_datamodule import SelfDistillDataModule  # noqa: E402

# var_weight, cov_weight, diversity_weight. cov moves with var because they are two
# halves of one regulariser: variance alone permits perfectly correlated dimensions.
CELLS = {
    "ce":         (0.0, 0.00, 0.0),
    "ce+div":     (0.0, 0.00, 1.0),
    "ce+vic":     (1.0, 0.04, 0.0),
    "ce+vic+div": (1.0, 0.04, 1.0),
}

# Pulled from the training metrics after the run. train/ rather than val/ because a
# short run may never validate.
KEYS = [
    # token_bits_frac, codes_used and token_bits are logged at EPOCH end by
    # _log_epoch_codebook, pooled over the whole epoch rather than per batch, which is
    # why the run must complete an epoch rather than be cut short mid-way.
    ("token_bits_frac", "train/token_bits_frac"),
    ("token_bits", "train/token_bits"),
    ("codes_used", "train/codes_used"),
    ("ce", "train/ce"),
    ("masked_acc", "train/masked_acc"),
    ("vic_var", "train/vic_var"),
    ("vic_cov", "train/vic_cov"),
    ("diversity", "train/diversity"),
    ("loss", "train/loss"),
]


def run_cell(name, var_w, cov_w, div_w, args):
    extra = []
    if args.levels:
        extra.append(f"module.network.tokenizer.levels={args.levels}")
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        cfg = compose(
            config_name="selfdistill.yaml",
            overrides=extra + [
                f"module.network.distill.var_weight={var_w}",
                f"module.network.distill.cov_weight={cov_w}",
                f"module.network.distill.diversity_weight={div_w}",
                f"paths.dataset_dir={args.data_dir}",
                f"data.dataset.parquet_path={args.parquet}",
                f"data.loaders.train.num_workers={args.workers}",
                "data.loaders.val.num_workers=2",
                f"seed={args.seed}",
            ],
        )

    # Same seed for every cell, so the cells differ ONLY in the weights under test. The
    # data order, the masks and the initialisation are otherwise identical.
    L.seed_everything(cfg.seed, workers=True)

    dm = SelfDistillDataModule(
        dataset_configs=cfg.data.dataset,
        loader_configs=cfg.data.loaders,
        transform_configs=cfg.data.transform,
    )
    model = build_model(cfg.module, label_map={})

    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=args.steps,
        limit_val_batches=2,
        num_sanity_val_steps=0,
        accelerator=args.device,
        devices=1,
        precision=cfg.trainer.get("precision", 32) if args.device != "cpu" else 32,
        logger=False,             # no MLflow run for a throwaway diagnostic
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
    )
    trainer.fit(model=model, datamodule=dm)
    return {k: float(v) for k, v in trainer.callback_metrics.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5,
                    help="training batches per cell. 5 checks plumbing and term scale; "
                         "collapse needs ~1000+")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=59)
    ap.add_argument("--data-dir", default=os.environ.get("TORCA_DATA_DIR",
                                                         "/data/dclde_mim"))
    ap.add_argument("--parquet",
                    default="/home/tundra/projects/torca/ds/DCLDE_w_Buzzes.parquet")
    ap.add_argument("--cells", default=None,
                    help="comma-separated subset, e.g. ce,ce+div")
    # Codebook size, for comparing [8,6,5]=240 against [8,5,5,5]=1000. token_bits_frac
    # is normalised by log2(K), so it is the ONE metric that compares across these;
    # token_bits and codes_used are not, since the ceilings differ (7.91 vs 9.97).
    ap.add_argument("--levels", default=None,
                    help="override FSQ levels, e.g. '[8,6,5]' for K=240")
    args = ap.parse_args()

    wanted = args.cells.split(",") if args.cells else list(CELLS)
    results = {}
    for name in wanted:
        if name not in CELLS:
            raise SystemExit(f"unknown cell {name!r}; pick from {list(CELLS)}")
        print(f"\n{'=' * 70}\n{name}  ({args.steps} steps)\n{'=' * 70}")
        results[name] = run_cell(name, *CELLS[name], args)

    w = max(len(n) for n in results) + 2
    print(f"\n\n{args.steps} training steps per cell, seed {args.seed}\n")
    print(f"{'cell':<{w}}" + "".join(f"{lab:>14}" for lab, _ in KEYS))
    print("-" * (w + 14 * len(KEYS)))
    for name, m in results.items():
        row = f"{name:<{w}}"
        for _, key in KEYS:
            v = m.get(key)
            row += f"{'-':>14}" if v is None else f"{v:>14.4f}"
        print(row)

    print("\nRead token_bits_frac and codes FIRST. Low ce and high masked_acc are")
    print("collapse signatures, not success: the documented collapsed cell scored")
    print("ce 0.699 and masked_acc 0.821 against a healthy 2.776 and 0.378.")
    if args.steps < 500:
        print(f"\nNOTE: {args.steps} steps shows term SCALE and that nothing errors. It")
        print("cannot show collapse, which is a dynamic. Rerun with --steps 2000 to")
        print("match the budget the diversity sweep used before drawing a conclusion.")


if __name__ == "__main__":
    main()
