"""Server-side weighted multi-vector fusion search.

score(q, c) = sum_a w_a * cos(q_a, c_a)

Qdrant's Query API supports fusion natively (prefetch per named vector +
a fusion stage), so this hits one round trip rather than 5 separate
per-aspect queries + client-side merge.
"""

from __future__ import annotations

import time

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from pipeline.config import ASPECTS, QDRANT
from pipeline.qdrant_index import get_client, stable_point_id


def fetch_seed_vectors(client: QdrantClient, seed_track_id: str) -> dict[str, list[float]]:
    points = client.retrieve(
        collection_name=QDRANT.collection,
        ids=[stable_point_id(seed_track_id)],
        with_vectors=True,
    )
    if not points:
        raise ValueError(f"Seed track {seed_track_id} not found in index")
    return points[0].vector  # {aspect: [floats]}


def weighted_fusion_search(
    client: QdrantClient,
    seed_vectors: dict[str, list[float]],
    weights: dict[str, float],
    top_k: int = 20,
    exclude_id: str | None = None,
) -> list[dict]:
    """Prefetch top candidates per aspect (over-fetch to give the fusion
    stage enough recall), then re-score with the weighted sum client-side.
    Simpler and more transparent than Qdrant's server fusion, at the cost
    of one extra round trip -- fine at this index size (tens of thousands
    of points)."""
    start = time.perf_counter()
    prefetch_k = max(top_k * 5, 100)

    candidate_scores: dict[str, dict[str, float]] = {}
    for aspect in ASPECTS:
        w = weights.get(aspect, 0.0)
        if w <= 0 or aspect not in seed_vectors:
            continue
        hits = client.search(
            collection_name=QDRANT.collection,
            query_vector=qmodels.NamedVector(name=aspect, vector=seed_vectors[aspect]),
            limit=prefetch_k,
            with_payload=True,
        )
        for hit in hits:
            track_id = hit.payload["track_id"]
            candidate_scores.setdefault(track_id, {})[aspect] = hit.score

    fused = []
    for track_id, per_aspect in candidate_scores.items():
        if exclude_id is not None and track_id == exclude_id:
            continue
        total = sum(weights.get(a, 0.0) * s for a, s in per_aspect.items())
        fused.append({"track_id": track_id, "score": total, "per_aspect_score": per_aspect})

    fused.sort(key=lambda r: -r["score"])
    latency_ms = (time.perf_counter() - start) * 1000
    return fused[:top_k], latency_ms


def search_similar(
    seed_track_id: str, weights: dict[str, float], top_k: int = 20
) -> tuple[list[dict], float]:
    client = get_client()
    seed_vectors = fetch_seed_vectors(client, seed_track_id)
    return weighted_fusion_search(
        client, seed_vectors, weights, top_k=top_k, exclude_id=seed_track_id
    )
