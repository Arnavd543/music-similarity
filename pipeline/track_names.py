"""One-time enrichment: stamp track/artist names onto Qdrant payloads.

MTG-Jamendo's tag TSVs carry no names; `raw.meta.tsv` in the dataset repo
maps TRACK_ID -> TRACK_NAME/ARTIST_NAME. This script downloads it, then
updates the payload of every point already in the `tracks` collection.

Usage (anywhere with Qdrant access -- Colab or laptop):
    QDRANT_URL=... QDRANT_API_KEY=... python -m pipeline.track_names
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import requests

from pipeline.config import PATHS, QDRANT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_META_URL = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/raw.meta.tsv"


def fetch_name_map(cache_path: Path | None = None) -> dict[str, dict[str, str]]:
    """{track_id (no 'track_' prefix, zero-padded as in TSV): {title, artist}}"""
    cache_path = cache_path or PATHS.metadata_dir / "raw.meta.tsv"
    if not cache_path.exists():
        log.info("Downloading %s", RAW_META_URL)
        resp = requests.get(RAW_META_URL, timeout=120)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)

    names: dict[str, dict[str, str]] = {}
    with cache_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tid = row["TRACK_ID"].replace("track_", "")
            names[tid] = {"title": row.get("TRACK_NAME", ""), "artist": row.get("ARTIST_NAME", "")}
    log.info("Loaded names for %d tracks", len(names))
    return names


def stamp_names(client, names: dict[str, dict[str, str]], batch_size: int = 256) -> int:
    """Scroll every point in the tracks collection and set title/artist payload."""
    from qdrant_client.http import models as qmodels

    updated, offset = 0, None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT.collection, limit=batch_size,
            offset=offset, with_payload=True, with_vectors=False,
        )
        for p in points:
            tid = p.payload.get("track_id", "")
            meta = names.get(tid)
            if meta and meta["title"]:
                client.set_payload(
                    collection_name=QDRANT.collection,
                    payload={"title": meta["title"], "artist": meta["artist"]},
                    points=[p.id],
                )
                updated += 1
        if offset is None:
            break
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-cache", type=Path, default=None)
    args = parser.parse_args()

    from pipeline.qdrant_index import get_client

    names = fetch_name_map(args.meta_cache)
    client = get_client()
    updated = stamp_names(client, names)
    log.info("Stamped names on %d points in '%s'", updated, QDRANT.collection)


if __name__ == "__main__":
    main()
