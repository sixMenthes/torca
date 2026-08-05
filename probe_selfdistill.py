"""Evaluation entry point for the self-distillation study — the finetune.py analogue.

finetune.py trains a classifier head; this trains nothing. It extracts frozen
features and measures them with a linear probe, because the study's claim is about
what the REPRESENTATION contains, and any trained readout confounds that with the
readout's own capacity.

    # non-neural floor
    python probe_selfdistill.py source=mfcc

    # frozen control (pretrained, unadapted)
    python probe_selfdistill.py source=frozen

    # adapted cell
    python probe_selfdistill.py source=adapted ckpt_path=/path/to/last.ckpt

    # where does the confound live? sweep depth
    python probe_selfdistill.py source=frozen 'layers=[0,3,7,11]'

Reports the CO-PRIMARY metric for each cell: task balanced accuracy (grouped by
hydrophone, so the probe can't exploit the recording shortcut) AND nuisance
(hydrophone) balanced accuracy. A win is task UP and nuisance DOWN — task alone will
happily select a cell that adapted deeper into the confound.
"""

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

import probe as probe_lib
from probe_features import (
    build_loader,
    encoder_from_checkpoint,
    extract_backbone,
    extract_mfcc,
    labelled_pool,
)
from selfdistill_datamodule import SelfDistillDataModule
from util.pylogger import get_pylogger

log = get_pylogger(__name__)

root = Path(__file__).resolve()
while not (root / ".git").exists():
    if root.parent == root:
        raise FileNotFoundError("could not find project root (no .git found)")
    root = root.parent

sys.path.insert(0, str(root))
os.environ["PROJECT_ROOT"] = str(root)

_HYDRA_PARAMS = {
    "version_base": None,
    "config_path": str(root / "configs"),
    "config_name": "probe.yaml",
}


def _build_encoder(cfg):
    """Encoder for the requested cell — adapted (from checkpoint) or frozen control."""
    if cfg.source == "adapted":
        if not cfg.get("ckpt_path"):
            raise ValueError("source=adapted requires ckpt_path=...")
        log.info(f"Loading adapted encoder from {cfg.ckpt_path}")
        return encoder_from_checkpoint(cfg.ckpt_path)

    enc_cfg = cfg.module.network.encoder
    name = enc_cfg.name
    kwargs = {k: v for k, v in enc_cfg.items() if k != "name"}
    log.info(f"Building frozen {name} control")
    if name == "BEATs":
        from models.components.beats_encoder import BEATsEncoder

        return BEATsEncoder(**kwargs).eval()
    from models.components.birdmae_encoder import BirdMAEEncoder

    return BirdMAEEncoder(**kwargs).eval()


@hydra.main(**_HYDRA_PARAMS)
def run(cfg: DictConfig):
    log.info("Setup datamodule (manifest only — no training)")
    dm = SelfDistillDataModule(
        dataset_configs=cfg.data.dataset,
        loader_configs=cfg.data.loaders,
        transform_configs=cfg.data.transform,
    )

    df = labelled_pool(dm.df, cfg.data.dataset.labels)
    log.info(f"{df.height} labelled clips")

    sr = int(cfg.data.transform.input.sample_rate)
    loader = build_loader(
        df, sr, cfg.data.dataset.clip_duration, dm.label_map, dm.call_map,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )

    # Split tags come from the same split the adaptation used, so the probe's test
    # hydrophones are ones the backbone never adapted on.
    tags = probe_lib.split_tags(
        df.get_column("Dataset").to_list(),
        list(cfg.data.dataset.test_hydros),
        list(cfg.data.dataset.val_hydros),
        list(cfg.data.dataset.low_sr_hydros),
    )

    if cfg.source == "mfcc":
        cells = {"mfcc": (extract_mfcc(loader, sr, n_mfcc=cfg.n_mfcc))}
    else:
        encoder = _build_encoder(cfg)
        layers = list(cfg.layers) if cfg.get("layers") else [None]
        cells = {
            (f"layer{l}" if l is not None else "final"):
                extract_backbone(encoder, loader, layer=l, device=cfg.device)
            for l in layers
        }

    header = (f"{'cell':>10} {'dim':>6} {'task/ecotype':>14} {'nuisance/hydro':>16} "
              f"{'calltype':>10}")
    print("\n" + header)
    print("-" * len(header))
    chance = None

    for name, (X, meta) in cells.items():
        report = probe_lib.diagnose(
            X,
            task_labels={"ecotype": meta["label"]},
            nuisance_labels={"hydro": meta["hydrophone"]},
            hydrophone=meta["hydrophone"],
        )
        task = report["task/ecotype"]
        nuis = report["nuisance/hydro"]

        # Call-type: train-on-train / eval-on-test (decision 1.a), restricted to rows
        # that actually carry a call-type label.
        ct = "n/a"
        has_call = meta["call"] >= 0
        train_m = has_call & (tags == "train")
        test_m = has_call & (tags == "test")
        if train_m.sum() > 0 and test_m.sum() > 0:
            r = probe_lib.probe_fixed_split(X, meta["call"], train_m, test_m)
            ct = f"{r['balanced_acc']:.3f}"

        print(f"{name:>10} {X.shape[1]:>6} "
              f"{task['balanced_acc']:>9.3f}±{task['std']:.2f} "
              f"{nuis['balanced_acc']:>11.3f}±{nuis['std']:.2f} "
              f"{ct:>10}")
        chance = (task["majority_baseline"], nuis["majority_baseline"])

    if chance:
        print(f"\nchance: ecotype {chance[0]:.3f}, hydrophone {chance[1]:.3f}")
    print("Win = task UP and nuisance DOWN, jointly. Task alone selects cells that "
          "adapted deeper into the confound.")


if __name__ == "__main__":
    run()
