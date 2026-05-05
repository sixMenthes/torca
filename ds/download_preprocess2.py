"""Slice DCLDE 2027 annotations directly from GCS via byte-range reads.

For each unique NewPath we open the remote wav as a seekable file-like,
read only the frames covered by each (FileBeginSec, FileEndSec) pair,
resample to 16 kHz, and write the clip locally. The full source file is
never downloaded.

Files are processed concurrently; groups are submitted in DataFrame
order. Per-file/per-clip failures are logged and skipped.
"""

from __future__ import annotations

import logging
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil, floor

import librosa
import polars as pl
import soundfile as sf
from gcsfs.core import GCSFileSystem

DF_PATH = "./ds/DCLDE_w_Buzzes.parquet"
LOCAL_ROOT = pathlib.Path("./ds/try")
GS_ROOT = "gs://noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales/"
SAMPLE_RATE = 16_000
MAX_WORKERS = 8

GCLIENT = GCSFileSystem(token="anon")
log = logging.getLogger("download_preprocess")

SF_FORMATS = {".wav": "WAV", ".flac": "FLAC", ".aif": "AIFF", ".aiff": "AIFF", ".ogg": "OGG"}


def sf_format(remote_path: str) -> str | None:
    return SF_FORMATS.get(pathlib.Path(remote_path).suffix.lower())


def local_mirror(remote_path: str) -> pathlib.Path:
    return LOCAL_ROOT / remote_path.removeprefix(GS_ROOT)


def chunk_path(remote_path: str, st: int, et: int) -> pathlib.Path:
    base = local_mirror(remote_path)
    return base.with_name(f"{base.stem}_{st}-{et}.wav")


def normalize_intervals(start_times, end_times) -> list[tuple[int, int]]:
    out = []
    for st, et in zip(start_times, end_times):
        if st is None or et is None:
            continue
        s = max(0, floor(float(st)))
        e = ceil(float(et))
        if e > s:
            out.append((s, e))
    return out


def write_clip(
    snd: sf.SoundFile, sr: int, st: int, et: int, out: pathlib.Path
) -> bool:
    start_frame = st * sr
    n_frames = (et - st) * sr
    total = snd.frames
    if start_frame >= total:
        return False
    n_frames = min(n_frames, total - start_frame)
    snd.seek(start_frame)
    chunk = snd.read(frames=n_frames, dtype="float32", always_2d=False)
    if chunk.ndim > 1:
        chunk = chunk.mean(axis=1)
    if chunk.size == 0:
        return False
    if sr != SAMPLE_RATE:
        chunk = librosa.resample(chunk, orig_sr=sr, target_sr=SAMPLE_RATE)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    sf.write(str(tmp), chunk, SAMPLE_RATE)
    tmp.rename(out)
    return True


def process(remote_path: str, start_times, end_times) -> tuple[str, int]:
    intervals = normalize_intervals(start_times, end_times)
    pending = [
        (st, et, chunk_path(remote_path, st, et))
        for st, et in intervals
        if not (
            (p := chunk_path(remote_path, st, et)).exists() and p.stat().st_size > 0
        )
    ]
    if not pending:
        return remote_path, 0

    written = 0
    fmt = sf_format(remote_path)
    with GCLIENT.open(remote_path, "rb") as fobj, sf.SoundFile(fobj, format=fmt) as snd:
        sr = snd.samplerate
        for st, et, out in pending:
            try:
                if write_clip(snd, sr, st, et, out):
                    written += 1
            except Exception as e:
                log.warning("clip fail %s [%d-%d]: %s", remote_path, st, et, e)
    return remote_path, written


def grouped_jobs(df_path: str) -> list[tuple[str, list, list]]:
    df = (
        pl.read_parquet(df_path)
        .filter(pl.col("NewFileOk") & pl.col("NewPath").is_not_null())
        .group_by("NewPath", maintain_order=True)
        .agg("FileBeginSec", "FileEndSec")
    )
    return [
        (row["NewPath"], row["FileBeginSec"], row["FileEndSec"])
        for row in df.iter_rows(named=True)
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    jobs = grouped_jobs(DF_PATH)
    log.info("planned %d source files", len(jobs))
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

    n_done = n_fail = n_clips = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process, remote, st, et): remote
            for remote, st, et in jobs
        }
        for fut in as_completed(futures):
            remote = futures[fut]
            try:
                _, k = fut.result()
                n_done += 1
                n_clips += k
                log.info("ok [%d/%d] %s (+%d clips)", n_done + n_fail, len(jobs), remote, k)
            except Exception as e:
                n_fail += 1
                log.warning("fail [%d/%d] %s: %s", n_done + n_fail, len(jobs), remote, e)

    log.info("done: %d ok, %d failed, %d clips written", n_done, n_fail, n_clips)


if __name__ == "__main__":
    main()
