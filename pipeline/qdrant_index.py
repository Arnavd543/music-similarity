"""Build/update the Qdrant collection with one named vector per aspect.

score(q, c) = sum_a w_a * cos(q_a, c_a) is computed server-side at query
time by api/search.py; this module is only responsible for getting
aspect vectors (from trained heads if available, else raw pooled stem
embeddings as the ablation baseline) into Qdrant.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from pipeline.config import ASPECTS, MODEL, PATHS, QDRANT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def stable_point_id(track_id: str) -> int:
    """Deterministic across processes. Python's built-in hash() is salted per
    interpreter session (PYTHONHASHSEED), so using it for point IDs makes
    non-numeric track IDs unfindable in any later session and creates
    duplicates on re-upsert."""
    if str(track_id).isdigit():
        return int(track_id)
    import hashlib

    return int.from_bytes(hashlib.md5(str(track_id).encode()).digest()[:8], "big") >> 1


def get_client() -> QdrantClient:
    """Local (embedded) mode is what Colab notebooks should use --
    QdrantClient(path=...) needs no server process, just a directory,
    which you point at mounted Drive so the index survives a runtime
    recycle. Cloud/server modes are for the deployed app."""
    if QDRANT.local_path:
        log.info("Using local Qdrant at %s (no server)", QDRANT.local_path)
        return QdrantClient(path=QDRANT.local_path)
    if QDRANT.url:
        return QdrantClient(url=QDRANT.url, api_key=QDRANT.api_key)
    return QdrantClient(host=QDRANT.host, port=QDRANT.port)


def ensure_collection(client: QdrantClient, dim: int = MODEL.head_output_dim) -> None:
    # The lyric vector comes from the sentence-transformer (768-d), not the
    # 128-d aspect heads -- declaring it at head dim would make every lyric
    # upsert fail with a dimension mismatch.
    vectors_config = {
        aspect: qmodels.VectorParams(
            size=MODEL.lyric_embed_dim if aspect == "lyric" else dim,
            distance=qmodels.Distance.COSINE,
        )
        for aspect in ASPECTS
    }
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT.collection in existing:
        log.info("Collection %s already exists", QDRANT.collection)
        return
    client.create_collection(
        collection_name=QDRANT.collection,
        vectors_config=vectors_config,
        # Scalar quantization keeps a 30k-track x 5-aspect index inside the
        # 1GB Qdrant Cloud free tier, per the budget plan.
        quantization_config=qmodels.ScalarQuantization(
            scalar=qmodels.ScalarQuantizationConfig(
                type=qmodels.ScalarType.INT8, quantile=0.99, always_ram=True
            )
        ),
    )
    log.info("Created collection %s with aspects %s", QDRANT.collection, ASPECTS)


def load_aspect_table(embeddings_path: Path) -> pd.DataFrame:
    """Expects a parquet with columns: track_id, rhythm, melody, timbre, vocal, lyric
    (each an array of floats) -- i.e. already-projected aspect vectors, produced by
    training/aspect_heads.py's `project_all` + pipeline/lyrics.py for the lyric column."""
    return pd.read_parquet(embeddings_path)


def upsert_tracks(client: QdrantClient, df: pd.DataFrame, batch_size: int = 256) -> None:
    n = len(df)
    for start in range(0, n, batch_size):
        chunk = df.iloc[start : start + batch_size]
        points = []
        for i, row in chunk.iterrows():
            vectors = {}
            for aspect in ASPECTS:
                if aspect in row and row[aspect] is not None:
                    vec = np.asarray(row[aspect], dtype=np.float32)
                    vectors[aspect] = vec.tolist()
            if not vectors:
                continue
            points.append(
                qmodels.PointStruct(
                    id=stable_point_id(row["track_id"]),
                    vector=vectors,
                    payload={"track_id": str(row["track_id"])},
                )
            )
        if points:
            client.upsert(collection_name=QDRANT.collection, points=points)
        log.info("Upserted %d/%d", min(start + batch_size, n), n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aspect-embeddings", type=Path, default=PATHS.embeddings_dir / "aspect_vectors.parquet"
    )
    parser.add_argument("--dim", type=int, default=MODEL.head_output_dim)
    args = parser.parse_args()

    client = get_client()
    ensure_collection(client, dim=args.dim)
    df = load_aspect_table(args.aspect_embeddings)
    upsert_tracks(client, df)
    log.info("Indexed %d tracks into Qdrant collection %s", len(df), QDRANT.collection)


if __name__ == "__main__":
    main()
