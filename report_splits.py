"""Print exactly what every probe is computed on. No backbone, no GPU, seconds.

    python report_splits.py                          # default (birdmae) dataset config
    python report_splits.py data/dataset=dclde_selfdistill_beats
    python report_splits.py include_low_sr=true      # what folding them in would give

Reads the same config and the same cached manifest the probe does, and calls the same
probe_pool/split_report, so it cannot report a split the probe does not use. That is
the point: the three probes each build their own mask over the tag array in a different
file, and before this the only way to answer "how many clips, from which hydrophones,
went into this number" was to read all three call sites and hold the tag cascade in
your head.

Write it to a file for the writeup:

    python report_splits.py > splits.txt
"""

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from probe_features import probe_pool, split_report
from selfdistill_datamodule import SelfDistillDataModule

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
os.environ["PROJECT_ROOT"] = str(root)


@hydra.main(version_base=None, config_path=str(root / "configs"),
            config_name="probe.yaml")
def run(cfg: DictConfig):
    dm = SelfDistillDataModule(
        dataset_configs=cfg.data.dataset,
        loader_configs=cfg.data.loaders,
        transform_configs=cfg.data.transform,
    )
    # Same gate the probe uses: this is what filters df down to clips that actually
    # landed on disk. Reporting the split off the un-materialised manifest would
    # overcount by however many clips failed to fetch (~7.3k in the current stage).
    dm.prepare_data()

    df, tags = probe_pool(dm.df, cfg.data.dataset, include_low_sr=cfg.include_low_sr)

    print(f"\ndataset config : {cfg.data.dataset.name}")
    print(f"manifest       : {cfg.paths.dataset_dir}/{cfg.data.dataset.manifest_name}")
    print(f"clip_duration  : {cfg.data.dataset.clip_duration} s")
    # The frozen cells are NOT bandwidth-matched: this interpolates
    # module.network.sampling_rate, so it is 16 kHz under BEATs and 32 kHz under
    # Bird-MAE. Print it next to the split so the two are read together.
    print(f"probe sample_rate : {int(cfg.data.transform.input.sample_rate)} Hz "
          f"(follows module.network.sampling_rate)")
    print(f"include_low_sr : {bool(cfg.include_low_sr)}")
    print(f"c_selection    : {cfg.c_selection}\n")
    print(split_report(df, tags, cfg.data.dataset, source_df=dm.df))


if __name__ == "__main__":
    run()
