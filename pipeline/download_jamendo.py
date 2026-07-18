"""Download a subset of MTG-Jamendo (CC-licensed, so the demo can legally stream audio).

MTG-Jamendo ships track-level metadata TSVs and audio split into 100 tar
archives on MTG's servers / Zenodo. This script pulls the metadata first
(cheap), lets you pick a random or tag-stratified subset of N tracks, then
downloads just those audio files.

Usage:
    python -m pipeline.download_jamendo --subset autotagging_top50tags \
        --n-tracks 5000 --out data/audio

Reference: https://github.com/MTG/mtg-jamendo-dataset
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

from pipeline.config import PATHS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_30S_BASE = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio"
METADATA_URLS = {
    "autotagging": (
        "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/"
        "autotagging.tsv"
    ),
    "autotagging_top50tags": (
        "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/"
        "autotagging_top50tags.tsv"
    ),
}


def fetch_metadata(subset: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info("Metadata already present at %s", dest)
        return dest
    url = METADATA_URLS[subset]
    log.info("Downloading metadata from %s", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def read_track_ids(metadata_tsv: Path) -> list[dict]:
    """MTG-Jamendo TSVs have a *variable* number of tab-separated tag fields
    after the DURATION column, so DictReader/pandas silently keep only the
    first tag. Parse manually: fixed columns, then everything else is TAGS."""
    rows = []
    with metadata_tsv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        n_fixed = header.index("TAGS") if "TAGS" in header else len(header) - 1
        for fields in reader:
            if not fields:
                continue
            row = dict(zip(header[:n_fixed], fields[:n_fixed]))
            row["TAGS"] = [t for t in fields[n_fixed:] if t]
            rows.append(row)
    return rows


def stratified_subset(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample tracks spread across genre tags rather than pure random, so the
    subset still covers the tag space used later for precision@k eval."""
    import random

    rng = random.Random(seed)
    by_tag: dict[str, list[dict]] = {}
    for row in rows:
        tags = [t for t in row.get("TAGS", []) if t.startswith("genre---")]
        key = tags[0] if tags else "genre---unknown"
        by_tag.setdefault(key, []).append(row)

    per_tag_quota = max(1, n // max(1, len(by_tag)))
    chosen: list[dict] = []
    for _tag, group in by_tag.items():
        rng.shuffle(group)
        chosen.extend(group[:per_tag_quota])
    rng.shuffle(chosen)
    return chosen[:n]


def track_audio_url(track_row: dict) -> str:
    # Prefer the dataset's own PATH column over reconstructing the layout.
    path = track_row.get("PATH")
    if path:
        return f"{RAW_30S_BASE}/{path}"
    track_id = track_row["TRACK_ID"].replace("track_", "")
    folder = f"{int(track_id) % 100:02d}"
    return f"{RAW_30S_BASE}/{folder}/{track_id}.mp3"


SEGMENT_SECONDS = 30  # keep only excerpt(s); full Jamendo tracks are ~20MB each


def _probe_duration(src: Path) -> float | None:
    import subprocess

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(src)],
            capture_output=True, text=True, timeout=30,
        )
        return float(probe.stdout.strip())
    except Exception:
        return None


def _crop(src: Path, dest: Path, start: float, seconds: int = SEGMENT_SECONDS) -> bool:
    """Crop with ffmpeg stream copy (no re-encode, ~instant)."""
    import subprocess

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-i", str(src),
             "-t", str(seconds), "-c", "copy", str(dest)],
            capture_output=True, timeout=60,
        )
        return result.returncode == 0 and dest.exists() and dest.stat().st_size > 0
    except Exception:
        return False


def _crop_center(src: Path, dest: Path, seconds: int = SEGMENT_SECONDS) -> bool:
    duration = _probe_duration(src)
    if duration is None:
        return False
    return _crop(src, dest, max(0.0, (duration - seconds) / 2), seconds)


def _crop_two_segments(src: Path, dest_a: Path, dest_b: Path, seconds: int = SEGMENT_SECONDS) -> bool:
    """Two *disjoint* segments (centers of the 1st and 2nd half) for
    cross-section positive pairs. Falls back to a single center crop
    (dest_a only) when the track is too short for disjoint segments."""
    duration = _probe_duration(src)
    if duration is None:
        return False
    if duration < 2 * seconds + 5:
        return _crop(src, dest_a, max(0.0, (duration - seconds) / 2), seconds)
    start_a = max(0.0, duration / 3 - seconds / 2)
    start_b = max(start_a + seconds, 2 * duration / 3 - seconds / 2)
    ok_a = _crop(src, dest_a, start_a, seconds)
    ok_b = _crop(src, dest_b, start_b, seconds)
    return ok_a and ok_b


def _download_one(row: dict, out_dir: Path, two_segments: bool = False) -> bool:
    """Download the (full-length, ~20MB) file to a temp path, crop 30s
    excerpt(s) (~1.2MB each), delete the temp. `two_segments=True` writes
    {id}__a.mp3 and {id}__b.mp3 (disjoint sections, for cross-section
    positive pairs); otherwise a single center-crop {id}.mp3."""
    track_id = row["TRACK_ID"].replace("track_", "")
    # Temp full-length file goes to LOCAL disk, never out_dir: when out_dir is
    # on a Drive mount, FUSE deletions move files to Drive's *trash*, which
    # still counts against quota -- ~22MB x thousands of tracks silently
    # filled the account and blocked all further Drive writes.
    import tempfile

    tmp = Path(tempfile.gettempdir()) / f"{track_id}.full.tmp.mp3"
    url = track_audio_url(row)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp.write_bytes(resp.content)
    except requests.RequestException as exc:
        log.warning("Failed to download %s: %s", track_id, exc)
        tmp.unlink(missing_ok=True)
        return False

    if two_segments:
        ok = _crop_two_segments(tmp, out_dir / f"{track_id}__a.mp3", out_dir / f"{track_id}__b.mp3")
    else:
        ok = _crop_center(tmp, out_dir / f"{track_id}.mp3")

    if ok:
        keep_dir = os.getenv("MS_KEEP_FULL_AUDIO_DIR")
        if keep_dir:
            full_dest = Path(keep_dir) / f"{track_id}.mp3"
            full_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp), str(full_dest))
        else:
            tmp.unlink(missing_ok=True)
    else:
        log.warning("ffmpeg crop failed for %s -- keeping full-length file", track_id)
        tmp.rename(out_dir / (f"{track_id}__a.mp3" if two_segments else f"{track_id}.mp3"))
    return True


def download_audio(
    rows: list[dict],
    out_dir: Path,
    skip_existing: bool = True,
    workers: int = 8,
    two_segments: bool = False,
) -> None:
    """Network-bound -- doesn't need a GPU.

    First 10 downloads run sequentially as a fail-fast canary (if all 10
    fail, the URL scheme is wrong -- raise instead of logging 5,000
    warnings); the rest run in a thread pool (`workers`).
    `two_segments=True` (pair subset) writes {id}__a.mp3 + {id}__b.mp3."""
    from concurrent.futures import ThreadPoolExecutor

    out_dir.mkdir(parents=True, exist_ok=True)

    def _done(row: dict) -> bool:
        tid = row["TRACK_ID"].replace("track_", "")
        marker = f"{tid}__a.mp3" if two_segments else f"{tid}.mp3"
        return (out_dir / marker).exists()

    todo = [row for row in rows if not (skip_existing and _done(row))]
    if not todo:
        log.info("All %d tracks already downloaded", len(rows))
        return

    canary, rest = todo[:10], todo[10:]
    canary_ok = sum(
        _download_one(row, out_dir, two_segments) for row in tqdm(canary, desc="canary downloads")
    )
    if canary_ok == 0:
        raise RuntimeError(
            f"First {len(canary)} downloads all failed (e.g. {track_audio_url(canary[-1])}) -- "
            "the audio URL scheme is probably wrong; verify one URL in a browser "
            "before bulk-downloading."
        )

    if rest:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(
                tqdm(pool.map(lambda r: _download_one(r, out_dir, two_segments), rest),
                     total=len(rest), desc="downloading audio")
            )
        log.info("Downloaded %d/%d (plus %d/10 canary)", sum(results), len(rest), canary_ok)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="autotagging_top50tags", choices=list(METADATA_URLS))
    parser.add_argument("--n-tracks", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=PATHS.audio_dir)
    parser.add_argument("--metadata-out", type=Path, default=PATHS.metadata_dir / "jamendo.tsv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_path = fetch_metadata(args.subset, args.metadata_out)
    rows = read_track_ids(metadata_path)
    log.info("Loaded metadata for %d tracks", len(rows))

    subset_rows = stratified_subset(rows, args.n_tracks, seed=args.seed)
    log.info("Selected %d tracks (tag-stratified)", len(subset_rows))

    download_audio(subset_rows, args.out)
    log.info("Done. Audio saved to %s", args.out)


if __name__ == "__main__":
    main()
