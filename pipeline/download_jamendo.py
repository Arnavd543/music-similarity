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


def download_audio(rows: list[dict], out_dir: Path, skip_existing: bool = True) -> None:
    """Fail fast if the URL scheme is wrong: if the first 10 attempted
    downloads all fail, raise instead of logging 5,000 warnings and leaving
    the ingest loop nothing to process."""
    out_dir.mkdir(parents=True, exist_ok=True)
    attempted = succeeded = 0
    for row in tqdm(rows, desc="downloading audio"):
        track_id = row["TRACK_ID"].replace("track_", "")
        dest = out_dir / f"{track_id}.mp3"
        if skip_existing and dest.exists():
            continue
        url = track_audio_url(row)
        attempted += 1
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            succeeded += 1
        except requests.RequestException as exc:
            log.warning("Failed to download %s: %s", track_id, exc)
        if attempted == 10 and succeeded == 0:
            raise RuntimeError(
                f"First {attempted} downloads all failed (last URL: {url}) -- "
                "the audio URL scheme is probably wrong; verify one URL in a "
                "browser before bulk-downloading."
            )


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
