"""Download DCLDE clips at a fixed clip_duration and write the cached manifest.

Run this ONCE per machine before any training or probing:

    python prestage_clips.py --dataset-dir /path/to/clips --clip-duration 3.0
    python prestage_clips.py --dataset-dir /path/to/clips --verify   # count, no download

Why it exists: `LocalPath` encodes the clip window as `{start_ms}-{end_ms}.wav`, so
clips staged at one `clip_duration` are invisible at another — 3 s and 5 s share zero
filenames out of 206k. Worse, the failure is silent: `LabelDataModule.prepare_data`
skips downloading whenever the cached manifest exists, every row then misses on disk,
`SelfDistillDataset.__getitem__` returns None, `collate_fn_skip` returns None, and you
get empty batches instead of an error.

Paths come from `build_clip_manifest`, the same function the datamodule uses, so the
two cannot drift. Clips are written at the SOURCE sample rate (no resampling here) —
the dataset resamples per backbone at load time.

Idempotent and resumable: existing clips are skipped, so re-running after an
interruption only fetches what is missing.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gcsfs
import polars as pl
import soundfile as sf
from tqdm import tqdm

from torca_datamodule import build_clip_manifest

GCS = gcsfs.core.GCSFileSystem(token="anon")


def fetch_source_file(gcs_path, starts, ends, paths):
    """Extract every requested window from ONE source file.

    Grouped per source file on purpose: opening a remote file is the expensive part,
    and a single file typically supplies many annotations. Returns (n_written, failed).
    """
    todo = [(s, e, p) for s, e, p in zip(starts, ends, paths) if not Path(p).exists()]
    if not todo:
        return 0, None

    written = 0
    try:
        with GCS.open(gcs_path, "rb", block_size=2**20) as fobj:
            with sf.SoundFile(fobj) as snd:
                sr = snd.samplerate
                for start_s, end_s, out in todo:
                    start_frame = int(round(start_s * sr))
                    n_frames = int(round(end_s * sr)) - start_frame
                    snd.seek(start_frame)
                    audio = snd.read(n_frames, dtype="float32")
                    if audio.size == 0:
                        continue
                    Path(out).parent.mkdir(parents=True, exist_ok=True)
                    # write to a temp name then rename, so an interrupted run never
                    # leaves a half-written clip that a later run would skip as "done".
                    # format="WAV" is REQUIRED: soundfile infers format from the
                    # extension, and ".part" is not one it knows.
                    tmp = Path(str(out) + ".part")
                    sf.write(str(tmp), audio, sr, format="WAV")
                    tmp.rename(out)
                    written += 1
    except Exception as e:
        return written, f"{gcs_path}: {e}"
    return written, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default="./ds/DCLDE_w_Buzzes.parquet",
                    help="source annotation parquet")
    ap.add_argument("--dataset-dir", required=True,
                    help="must equal data.dataset.dataset_dir in the run config")
    ap.add_argument("--clip-duration", type=float, default=3.0,
                    help="must equal data.dataset.clip_duration in the run config")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--manifest-name", default="DCLDE_no_balance",
                    help="cache filename LabelDataModule.prepare_data looks for")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N source files (smoke test)")
    ap.add_argument("--verify", action="store_true",
                    help="report how many clips already exist; download nothing")
    args = ap.parse_args()

    print(f"parquet       : {args.parquet}")
    print(f"dataset-dir   : {args.dataset_dir}")
    print(f"clip-duration : {args.clip_duration}  <-- must match the run config")

    manifest = build_clip_manifest(
        pl.read_parquet(args.parquet), args.clip_duration, args.dataset_dir
    )
    print(f"clips planned : {manifest.height} over "
          f"{manifest.get_column('GCSPath').n_unique()} source files")

    if args.verify:
        exists = [Path(p).exists() for p in manifest.get_column("LocalPath")]
        n = sum(exists)
        print(f"\non disk       : {n} / {manifest.height} ({100*n/manifest.height:.1f}%)")
        if n < manifest.height:
            missing = manifest.filter(~pl.Series(exists))
            print("missing by hydrophone:")
            print(missing.group_by("Dataset").len().sort("len", descending=True).head(10))
        return

    jobs = manifest.group_by("GCSPath", maintain_order=True).agg(
        "new_start_time", "new_end_time", "LocalPath"
    )
    if args.limit:
        jobs = jobs.head(args.limit)

    written, failures = 0, []
    with tqdm(total=jobs.height, desc="source files") as bar:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(fetch_source_file, r["GCSPath"], r["new_start_time"],
                          r["new_end_time"], r["LocalPath"])
                for r in jobs.iter_rows(named=True)
            ]
            for fut in as_completed(futures):
                n, err = fut.result()
                written += n
                if err:
                    failures.append(err)
                bar.update(1)

    print(f"\nclips written : {written}")
    print(f"source files failed: {len(failures)}")
    for f in failures[:10]:
        print(f"  {f}")

    # Cache manifest: prepare_data filters self.df by the Soundfile values in here, so
    # source files that failed entirely are dropped from the run. Note the filter is
    # per SOURCE FILE, not per clip (Soundfile is shared by ~16 annotations on
    # average), so a partially-fetched file keeps all its rows; the stragglers return
    # None at load time and collate_fn_skip drops them.
    on_disk = [Path(p).exists() for p in manifest.get_column("LocalPath")]
    survived = manifest.filter(pl.Series(on_disk))
    out = Path(args.dataset_dir) / args.manifest_name
    out.parent.mkdir(parents=True, exist_ok=True)
    survived.write_parquet(out)

    print(f"clips on disk : {survived.height} / {manifest.height}")
    print(f"manifest      : {out}")
    print("\nprepare_data will now skip downloading. If you change clip_duration, "
          "DELETE this manifest and re-run — otherwise the run silently sees no data.")


if __name__ == "__main__":
    main()
