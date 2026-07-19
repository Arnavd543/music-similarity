"""Durable storage for pairwise feedback judgments, backed by Qdrant.

The API host (e.g. a free-tier container) loses its memory on every idle
spin-down, so judgments live in a dedicated Qdrant collection instead --
the cluster is already the system's one durable store. Points carry a
1-d dummy vector (Qdrant requires a vector) and the judgment as payload.
"""

from __future__ import annotations

import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from training.personalization import PairwiseJudgment

FEEDBACK_COLLECTION = "feedback"


def ensure_feedback_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if FEEDBACK_COLLECTION in existing:
        return
    client.create_collection(
        collection_name=FEEDBACK_COLLECTION,
        vectors_config=qmodels.VectorParams(size=1, distance=qmodels.Distance.DOT),
    )


def save_judgment(
    client: QdrantClient,
    user_id: str,
    seed_track_id: str,
    judgment: PairwiseJudgment,
) -> None:
    ensure_feedback_collection(client)
    client.upsert(
        collection_name=FEEDBACK_COLLECTION,
        points=[
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.0],
                payload={
                    "user_id": user_id,
                    "seed_track_id": seed_track_id,
                    "seed_sims_a": judgment.seed_sims_a,
                    "seed_sims_b": judgment.seed_sims_b,
                    "chose_a": judgment.chose_a,
                    "ts": time.time(),
                },
            )
        ],
    )


def load_user_judgments(client: QdrantClient, user_id: str, limit: int = 1000) -> list[PairwiseJudgment]:
    ensure_feedback_collection(client)
    points, _ = client.scroll(
        collection_name=FEEDBACK_COLLECTION,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))]
        ),
        limit=limit,
        with_payload=True,
    )
    return [
        PairwiseJudgment(
            seed_sims_a=p.payload["seed_sims_a"],
            seed_sims_b=p.payload["seed_sims_b"],
            chose_a=p.payload["chose_a"],
        )
        for p in points
    ]
