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
import numpy as np
import torch
from omegaconf import DictConfig

# Pass worker->main tensors through /dev/shm files instead of file descriptors.
#
# The default 'file_descriptor' strategy costs one open fd per tensor handed across
# the process boundary, and the probe iterates ~5.8k batches over 187k clips in a
# single pass — against a soft RLIMIT_NOFILE of 1024 (systemd's default, which an
# interactive shell usually inherits) that exhausts partway through and surfaces as
# "Too many open files. Communication with the workers is no longer possible".
# It hit both the MFCC and the frozen-backbone cells, i.e. it is the loader, not the
# feature extractor.
#
# Must run before any DataLoader worker starts, hence module scope. The tradeoff is
# that a hard kill (SIGKILL, OOM) can leave files behind in /dev/shm, where the fd
# strategy would have had the kernel reap them — acceptable for a batch job that
# reads a fixed dataset once, and the alternative is a run that dies two hours in.
torch.multiprocessing.set_sharing_strategy("file_system")

import probe as probe_lib
from emissions import track_emissions
from probe_features import (
    build_loader,
    encoder_from_checkpoint,
    extract_backbone,
    extract_mfcc,
    probe_pool,
    split_report,
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
    # NOT optional, and it is not about downloading. prepare_data is what filters
    # self.df down to clips that exist on disk. Without it, rows whose clip never
    # landed stay in df — and then `tags` below is built from df while X is built
    # from the loader, which SKIPS those rows (_collate drops them, extract_backbone
    # drops empty batches). X ends up shorter than tags and every split assignment
    # is silently shifted. With ~7.3k absent clips in the current stage that is not
    # hypothetical.
    dm.prepare_data()

    # Rows and split tags in one call, so they cannot drift apart — see probe_pool.
    # Split tags come from the same split the adaptation used, so the probe's test
    # hydrophones are ones the backbone never adapted on.
    # seal_test drops the held-out rows HERE, before the backbone runs, so the model
    # never sees a test clip rather than seeing it and having the result masked away.
    df, tags = probe_pool(dm.df, cfg.data.dataset,
                          include_low_sr=cfg.include_low_sr,
                          seal_test=cfg.seal_test)
    log.info(f"{df.height} clips in the probe pool"
             + (" (TRAIN ONLY — test split sealed)" if cfg.seal_test else ""))
    print("\n" + split_report(df, tags, cfg.data.dataset, source_df=dm.df) + "\n")

    sr = int(cfg.data.transform.input.sample_rate)
    loader = build_loader(
        df, sr, cfg.data.dataset.clip_duration, dm.label_map, dm.call_map,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )

    # Feature extraction is the expensive part (a backbone forward over every
    # labelled clip), so that is what the tracker wraps; the probe fits themselves
    # are sklearn on a few thousand rows.
    emissions_cfg = cfg.get("emissions", {}) or {}
    with track_emissions(logger=None, project_name=cfg.task_name,
                         output_dir=str(cfg.paths.get("log_dir", ".")),
                         **emissions_cfg) as emissions:
        if cfg.source == "mfcc":
            # "final", NOT "mfcc". This key becomes the metric prefix, and MLflow can
            # only overlay or tabulate runs that share a metric name — naming the
            # floor's cell after itself put C0 on keys no other cell has, so it
            # silently dropped out of every cross-cell chart and comparison. The name
            # means "the pooled representation", which is exactly what this is.
            cells = {"final": (extract_mfcc(loader, sr, n_mfcc=cfg.n_mfcc))}
        else:
            encoder = _build_encoder(cfg)
            layers = list(cfg.layers) if cfg.get("layers") else [None]
            cells = {
                (f"layer{l}" if l is not None else "final"):
                    extract_backbone(encoder, loader, layer=l, device=cfg.device)
                for l in layers
            }

    # Belt and braces on the alignment above. prepare_data should have made every row
    # in df loadable, but a clip that exists and fails to DECODE is also dropped by
    # _collate, and that one no manifest can predict. A length mismatch here means the
    # split tags no longer correspond row-for-row to the features, which does not raise
    # on its own — it just quietly reports numbers for the wrong split.
    for name, (X, _) in cells.items():
        if X.shape[0] != len(tags):
            raise RuntimeError(
                f"feature/tag misalignment in cell '{name}': {X.shape[0]} feature rows "
                f"vs {len(tags)} split tags. {len(tags) - X.shape[0]} clips were "
                f"dropped during extraction (missing or undecodable). Re-run "
                f"prestage_clips.py --manifest-only so the manifest matches the tree."
            )

    # Optional: write the features out for offline projection. This happens BEFORE the
    # probes run, so a crash in sklearn does not cost the backbone pass that produced
    # them — that pass is the expensive part of this script and the projection figure
    # needs nothing else from it.
    if cfg.get("save_features"):
        out_dir = Path(cfg.save_features)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Integer codes are meaningless without their maps, and the maps live on the
        # datamodule rather than in the config, so they travel WITH the features. A
        # features file that needs this script re-run to be interpreted is not a
        # portable artifact.
        label_names = [n for n, _ in sorted(dm.label_map.items(), key=lambda kv: kv[1])]
        call_names = [n for n, _ in sorted((dm.call_map or {}).items(),
                                           key=lambda kv: kv[1])]
        for name, (X, meta) in cells.items():
            path = out_dir / f"{cfg.cell}_{name}.npz"
            np.savez_compressed(
                path,
                X=X.astype(np.float16),
                label=meta["label"], call=meta["call"],
                hydrophone=meta["hydrophone"].astype(str),
                tags=np.asarray(tags).astype(str),
                label_names=np.array(label_names),
                call_names=np.array(call_names),
                cell=str(cfg.cell), layer=str(name), source=str(cfg.source),
                encoder=("none (MFCC)" if cfg.source == "mfcc"
                         else str(cfg.module.network.encoder.name)),
                ckpt_path=str(cfg.get("ckpt_path")),
                seal_test=bool(cfg.seal_test),
            )
            log.info(f"wrote features for cell '{cfg.cell}' layer '{name}' to {path} "
                     f"({path.stat().st_size / 1e6:.1f} MB, {X.shape[0]} clips)")

    # Background index drives the nuisance probe's channel-only restriction.
    bg_index = dm.label_map.get("Background", -1)
    if bg_index < 0:
        log.warning("no 'Background' class in label_map — nuisance probe will fail")

    # Results go to MLflow as well as stdout: six cells plus a levels sweep is too many
    # numbers to transcribe from terminal output reliably.
    logger = None
    if cfg.get("logger") and not sys.gettrace():
        logger = hydra.utils.instantiate(cfg.logger)
        logger.log_hyperparams({
            "source": cfg.source,
            "ckpt_path": str(cfg.get("ckpt_path")),
            "layers": str(list(cfg.layers) if cfg.get("layers") else ["final"]),
            "c_selection": cfg.c_selection,
            # The MFCC floor has no encoder. module/network is still composed for it
            # (probe.yaml needs a default), so reading the name straight off the
            # config labelled C0 as BirdMAE and made it look like a Bird-MAE variant
            # in every params column.
            "encoder": "none (MFCC)" if cfg.source == "mfcc"
                       else cfg.module.network.encoder.name,
            "clip_duration": cfg.data.dataset.clip_duration,
            # Not cosmetic. transform.input.sample_rate interpolates
            # module.network.sampling_rate, so it is 16 kHz for BEATs and 32 kHz for
            # Bird-MAE — the frozen cells are NOT bandwidth-matched to each other, and
            # nothing in the results table said so. Surface it as a params column.
            "sample_rate": sr,
            "include_low_sr": bool(cfg.include_low_sr),
            "test_hydros": str(list(cfg.data.dataset.test_hydros)),
            "n_labelled_clips": df.height,
        })

    header = (f"{'cell':>10} {'dim':>6} {'task/ecotype':>14} {'nuisance/bg':>16} "
              f"{'calltype':>10}")
    print("\n" + header)
    print("-" * len(header))
    chance = None

    for name, (X, meta) in cells.items():
        if cfg.seal_test:
            # TASK, sealed variant: GroupKFold BY HYDROPHONE within the train split.
            # Each fold is scored on train sites its own fit never saw, which is the
            # structure of the reported protocol without spending the test split. The
            # same quantity the online TrainCVProbe logs as val/probe_eco_cvtrain, so
            # the trajectory during training and this endpoint are directly comparable.
            m = tags == "train"
            task = probe_lib.probe_cv(
                X[m], meta["label"][m], groups=meta["hydrophone"][m], cv="group",
                seed=cfg.seed, n_jobs=cfg.probe_n_jobs,
            )
            # probe_split_protocol's extra keys, so the printing and logging below do
            # not need a branch for every field.
            task.update({"C": 1.0, "per_hydrophone": {},
                         "n_train": int(m.sum()), "n_test": 0})
        else:
            # TASK: fit on train hydros, pick C on val, evaluate once on the held-out
            # test hydros. NOT GroupKFold over everything — that leaks adaptation-train
            # hydrophones into the probe's test folds and hands the adapted cell an
            # advantage the frozen control never gets.
            task = probe_lib.probe_split_protocol(
                X, meta["label"], tags, hydrophone=meta["hydrophone"],
                c_selection=cfg.c_selection, include_low_sr=cfg.include_low_sr,
                n_jobs=cfg.probe_n_jobs,
            )
        # NUISANCE: hydrophone decodability from BACKGROUND clips only, on train.
        # Background-only is what makes it a channel measurement rather than a content
        # one — see probe.probe_nuisance_background.
        nuis = probe_lib.probe_nuisance_background(
            X, meta["hydrophone"], tags, meta["label"] == bg_index,
            n_jobs=cfg.probe_n_jobs,
        )

        # CALL-TYPE: train-on-train / eval-on-test, C chosen by GroupKFold over the
        # train call-type hydrophones (val carries no call-type labels at all).
        ct = "n/a"
        has_call = meta["call"] >= 0
        train_m = has_call & (tags == "train")
        test_m = has_call & (tags == "test")
        if cfg.seal_test:
            # Its test mask IS the sealed split, so there is no sealed variant of this
            # one to fall back to. Skipped rather than approximated.
            train_m = test_m = np.zeros(len(tags), dtype=bool)
        if train_m.sum() > 0 and test_m.sum() > 0:
            ct_C, _ = probe_lib.select_C(
                X[train_m], meta["call"][train_m], meta["hydrophone"][train_m],
                n_jobs=cfg.probe_n_jobs,
            )
            r = probe_lib.probe_fixed_split(X, meta["call"], train_m, test_m, C=ct_C)
            ct = f"{r['balanced_acc']:.3f}"

        print(f"{name:>10} {X.shape[1]:>6} "
              f"{task['balanced_acc']:>14.3f} "
              f"{nuis['balanced_acc']:>11.3f}±{nuis['std']:.2f} "
              f"{ct:>10}")
        if task["per_hydrophone"]:
            per = "  ".join(f"{k}={v:.3f}" for k, v in task["per_hydrophone"].items())
            print(f"{'':>10} per test hydrophone: {per}  (C={task['C']})")
        chance = (task["majority_baseline"], nuis["majority_baseline"])

        if logger:
            # The metric KEY carries the protocol. A sealed run and a reported run are
            # different quantities and must never overlay each other in the UI, which
            # is exactly what would happen if both wrote {name}/task_ecotype.
            task_key = "task_ecotype_cvtrain" if cfg.seal_test else "task_ecotype"
            metrics = {
                f"{name}/{task_key}": task["balanced_acc"],
                f"{name}/{task_key}_chance": task["majority_baseline"],
                # Sealed only, and not decoration. Ecotype support is wildly uneven
                # across the train sites — SwanChan holds 1304 of the 1608 HW clips —
                # so a GroupKFold fold that holds SwanChan out trains on almost no HW
                # and tests on almost nothing else. sklearn says as much, via
                # "y_pred contains classes not in y_true". The headline mean can move
                # several points on which sites landed together, so the spread has to
                # be visible next to it or the number reads far more precise than it is.
                **({f"{name}/{task_key}_std": task["std"]} if "std" in task else {}),
                f"{name}/nuisance_bg": nuis["balanced_acc"],
                f"{name}/nuisance_bg_std": nuis["std"],
                f"{name}/nuisance_bg_chance": nuis["majority_baseline"],
                f"{name}/nuisance_bg_hydros": float(nuis["n_hydrophones"]),
                f"{name}/feature_dim": float(X.shape[1]),
                f"{name}/C": float(task["C"]),
            }
            if ct != "n/a":
                metrics[f"{name}/task_calltype"] = float(ct)
            # per-hydrophone breakdown matters here: with two test hydrophones the
            # headline can be carried by one of them, and that is worth seeing.
            #
            # Underscore, NOT a second slash. MLflow's file store writes each metric
            # to metrics/<key>, so a slash is a directory separator: with
            # "{name}/task_ecotype/{hydro}" it first creates metrics/{name}/task_ecotype
            # as a FILE for the headline metric, then tries to open that same path as a
            # directory and dies with NotADirectoryError. Any key that is a strict
            # prefix of another key breaks the same way. One level of slash is what
            # groups the cell in the UI; below that, keep it flat.
            for hydro, val in task["per_hydrophone"].items():
                metrics[f"{name}/task_ecotype_{str(hydro).replace('/', '_')}"] = val
            logger.log_metrics(metrics)

    if chance:
        where = "train, grouped CV" if cfg.seal_test else "test split"
        print(f"\nchance: ecotype {chance[0]:.3f} ({where}), "
              f"hydrophone {chance[1]:.3f} "
              f"({nuis['n_hydrophones']} hydros, {nuis['n_clips']} Background clips)")
        if cfg.seal_test:
            per = "  ".join(f"{v:.3f}" for v in task.get("per_fold", []))
            print(f"ecotype per fold: {per}   (std {task.get('std', float('nan')):.3f})")
            print(f"n_train={task['n_train']} over {task['n_folds']} hydrophone folds; "
                  f"TEST SPLIT NOT TOUCHED, call-type skipped. The ecotype number here "
                  f"is optimistic against the reported one — the probe's folds are "
                  f"grouped by site but the backbone adapted on all of them.")
        else:
            print(f"n_train={task['n_train']} n_test={task['n_test']} "
                  f"(C selected by {cfg.c_selection})")
        print("nuisance = hydrophone decodability from BACKGROUND clips only, so it "
              "measures channel, not content.\nCompare frozen vs adapted; the drop is "
              "the result. The background.p=0.0 run is the control that attributes it.")
    print("Win = task UP and nuisance DOWN, jointly. Task alone selects cells that "
          "adapted deeper into the confound.")

    if logger:
        # emissions is populated on exit from the context manager above, which happens
        # before this point — logged here so it lands in the same MLflow run
        if emissions:
            logger.log_metrics(emissions)
            print(f"\nemissions: {emissions.get('emissions/co2eq_kg', 0):.6f} kg CO2eq, "
                  f"{emissions.get('emissions/energy_kwh', 0):.4f} kWh")
        # finalize() flushes and closes the MLflow run. There is no Trainer here to do
        # it, so without this the run is left in RUNNING state in the UI.
        logger.finalize("success")
        log.info(f"logged to MLflow experiment '{cfg.logger.experiment_name}'")


if __name__ == "__main__":
    run()
