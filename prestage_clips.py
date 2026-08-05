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
import os
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


def existing_clips(dataset_dir):
    """Every .wav under dataset_dir, as a set of absolute paths — via ONE tree walk.

    Emphatically not `Path(p).exists()` per row. On Lustre each stat is a round trip
    to the metadata server, so 206k of them serially takes tens of minutes and hammers
    the MDS — the exact access pattern the Alliance "large collections of files"
    guidance tells you to avoid. A single os.walk gets the same answer from sequential
    readdir calls in seconds.
    """
    found = set()
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith(".wav"):
                found.add(os.path.join(root, f))
    return found


def make_tarball(dataset_dir, tarball):
    """Pack the clip tree into ONE file, and verify the entry count before trusting it.

    206k loose files is the thing Lustre is worst at: it burns inode quota and every
    job start re-stats the tree through the metadata server. One tarball copied to
    node-local NVMe and extracted there turns that into a single sequential read.

    Packed as `-C dataset_dir .` so the archive holds the CONTENTS, not the directory
    name — the job then extracts into a directory of its own choosing and nothing
    depends on what the staging directory happened to be called.

    Uses system tar rather than the tarfile module: for this many small entries the
    difference is minutes versus much longer.
    """
    import subprocess

    tarball = Path(tarball)
    tarball.parent.mkdir(parents=True, exist_ok=True)
    tmp = tarball.with_suffix(tarball.suffix + ".part")

    print(f"\npacking {dataset_dir} -> {tarball}")
    subprocess.run(["tar", "-cf", str(tmp), "-C", str(dataset_dir), "."], check=True)

    # Verify before the tarball is trusted (and certainly before anyone deletes the
    # loose copy): a truncated archive that is never checked is worse than no archive.
    n = int(subprocess.run(["bash", "-c", f"tar -tf {tmp} | wc -l"],
                           capture_output=True, text=True, check=True).stdout.strip())
    tmp.rename(tarball)
    size = subprocess.run(["du", "-h", str(tarball)], capture_output=True,
                          text=True).stdout.split()[0]
    print(f"  {n} entries, {size}")
    return n


def write_manifest(manifest, args):
    """Report what landed on disk and write the cache manifest.

    prepare_data filters self.df by the Soundfile values in here, so source files that
    failed entirely drop out of the run. The filter is per SOURCE FILE, not per clip
    (Soundfile is shared by ~16 annotations on average), so a partially-fetched file
    keeps all its rows; the stragglers return None at load time and collate_fn_skip
    drops them.
    """
    print("scanning dataset dir (one tree walk, not 206k stats) ...")
    found = existing_clips(args.dataset_dir)
    n_files = len(found)
    print(f"  {n_files} .wav files found on disk")

    survived = manifest.filter(pl.col("LocalPath").is_in(list(found)))
    n, total = survived.height, manifest.height
    print(f"clips on disk : {n} / {total} ({100*n/total:.1f}%)")

    if n < total:
        missing = manifest.filter(~pl.col("LocalPath").is_in(list(found)))
        print("\nmissing clips by hydrophone:")
        print(missing.group_by("Dataset").len().sort("len", descending=True).head(12))
        print("missing clips by provider:")
        print(missing.group_by("Provider").len().sort("len", descending=True).head(8))
        print("call-type labels lost:",
              missing.get_column("CalltypeCategory").drop_nulls().len())

    if args.verify:
        print("\n(--verify: nothing written)")
        return

    out = Path(args.dataset_dir) / args.manifest_name
    out.parent.mkdir(parents=True, exist_ok=True)
    survived.write_parquet(out)
    print(f"\nmanifest      : {out}")
    print("prepare_data will now skip downloading. If you change clip_duration, "
          "DELETE this manifest and re-run — otherwise the run silently sees no data.")

    # Tar AFTER the manifest is written, so the archive contains it. The job script
    # extracts both together and prepare_data finds the cache immediately.
    if args.tar:
        n = make_tarball(args.dataset_dir, args.tar)
        if args.remove_loose:
            if n < n_files:
                print(f"REFUSING to delete loose files: tar has {n} entries but the "
                      f"tree had {n_files} .wav files")
            else:
                import shutil
                print(f"removing loose tree {args.dataset_dir} ...")
                shutil.rmtree(args.dataset_dir)
                print("done — the tarball is now the only copy")


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
    ap.add_argument("--manifest-name", default="DCLDE_3secs.parquet",
                    help="cache filename prepare_data looks for; must match "
                         "data.dataset.manifest_name in the run config")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N source files (smoke test)")
    ap.add_argument("--verify", action="store_true",
                    help="report how many clips exist; download nothing, write nothing")
    ap.add_argument("--manifest-only", action="store_true",
                    help="skip downloading, just scan the dir and write the manifest "
                         "(use after an interrupted run whose clips already landed)")
    ap.add_argument("--tar", default=None, metavar="PATH",
                    help="pack the clip tree into this tarball after writing the "
                         "manifest. Do this: 206k loose files burn inode quota and "
                         "hammer the Lustre metadata server on every job start")
    ap.add_argument("--remove-loose", action="store_true",
                    help="delete the loose tree once --tar has been verified. "
                         "DESTRUCTIVE; refuses if the tar entry count looks short")
    args = ap.parse_args()

    print(f"parquet       : {args.parquet}")
    print(f"dataset-dir   : {args.dataset_dir}")
    print(f"clip-duration : {args.clip_duration}  <-- must match the run config")

    manifest = build_clip_manifest(
        pl.read_parquet(args.parquet), args.clip_duration, args.dataset_dir
    )
    print(f"clips planned : {manifest.height} over "
          f"{manifest.get_column('GCSPath').n_unique()} source files")

    if args.verify or args.manifest_only:
        write_manifest(manifest, args)
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

    write_manifest(manifest, args)


if __name__ == "__main__":
    main()
