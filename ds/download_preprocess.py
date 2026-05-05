"""Download DCLDE 2027 killer-whale audio from the NOAA public bucket.

Reads the audited parquet, filters to rows whose remote file was found
(NewFileOk == True), and bulk-downloads via gcsfs into a local mirror
rooted at `dclde_2027_killer_whales/`.
"""

from pathlib import Path

import gcsfs
import polars as pl
from fsspec.callbacks import TqdmCallback

PARQUET_PATH = "DCLDE_w_Buzzes.parquet"
LOCAL_ROOT = Path("dclde_2027_killer_whales")
GS_ROOT = "gs://noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales/"


def gs_to_local(gs_path: str, local_root: Path = LOCAL_ROOT) -> Path:
    # gs://noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales/dfo_crp/audio/x.wav
    # → dclde_2027_killer_whales/dfo_crp/audio/x.wav
    suffix = gs_path.removeprefix(GS_ROOT)
    return local_root / suffix


def plan_downloads(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    remote = (
        df.filter(pl.col("NewFileOk"))
          .select(pl.col("NewPath").unique())
          .to_series()
          .to_list()
    )
    local = [str(gs_to_local(p)) for p in remote]
    return remote, local


def ensure_parents(paths: list[str]) -> None:
    for p in paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)


def download(remote: list[str], local: list[str], fs: gcsfs.GCSFileSystem) -> None:
    with TqdmCallback(desc="downloading") as cb:
        fs.get(remote, local, on_error="ignore", callback=cb)


def main() -> None:
    df = pl.read_parquet(PARQUET_PATH)
    remote, local = plan_downloads(df)
    print(f"{len(remote)} unique files to download → {LOCAL_ROOT}/")

    ensure_parents(local)
    fs = gcsfs.GCSFileSystem(token="anon")
    download(remote, local, fs)


if __name__ == "__main__":
    main()
