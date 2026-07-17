"""Retrieval precision@k against MTG-Jamendo tags: instrument tags validate
the timbre axis, genre/mood tags validate the others. No human labels
needed -- this is a proxy metric that's cheap to run at scale and is what
the ablation table (raw whole-mix MERT vs per-stem MERT vs trained heads)
is built from.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build_tag_index(metadata_tsv: Path) -> dict[str, set[str]]:
    """Manual TSV parse: MTG-Jamendo rows have a variable number of tag
    fields after DURATION, which pandas/DictReader silently truncate to the
    first tag -- that would quietly corrupt every precision@k number."""
    import csv

    tag_index: dict[str, set[str]] = {}
    with metadata_tsv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        n_fixed = header.index("TAGS") if "TAGS" in header else len(header) - 1
        id_col = header.index("TRACK_ID")
        for fields in reader:
            if not fields:
                continue
            track_id = fields[id_col].replace("track_", "")
            tag_index[track_id] = {t for t in fields[n_fixed:] if t}
    return tag_index


def knn(query_id: str, embeddings: dict[str, np.ndarray], k: int) -> list[str]:
    q = embeddings[query_id]
    q = q / (np.linalg.norm(q) + 1e-8)
    scored = []
    for track_id, vec in embeddings.items():
        if track_id == query_id:
            continue
        v = vec / (np.linalg.norm(vec) + 1e-8)
        scored.append((track_id, float(q @ v)))
    scored.sort(key=lambda x: -x[1])
    return [tid for tid, _ in scored[:k]]


def precision_at_k(
    embeddings: dict[str, np.ndarray],
    tag_index: dict[str, set[str]],
    k: int = 10,
    n_queries: int | None = None,
    seed: int = 42,
) -> float:
    """A retrieved track counts as relevant if it shares >=1 tag with the
    query. This is a coarse proxy (exact match on tag sets would be too
    strict) but scales to thousands of queries with zero human labeling."""
    rng = np.random.default_rng(seed)
    query_ids = [tid for tid in embeddings if tid in tag_index and tag_index[tid]]
    if n_queries:
        query_ids = list(rng.choice(query_ids, size=min(n_queries, len(query_ids)), replace=False))

    precisions = []
    for qid in query_ids:
        neighbors = knn(qid, embeddings, k)
        q_tags = tag_index[qid]
        relevant = sum(1 for nid in neighbors if tag_index.get(nid, set()) & q_tags)
        precisions.append(relevant / k)

    return float(np.mean(precisions)) if precisions else float("nan")


def ablation_table(
    variants: dict[str, dict[str, np.ndarray]], tag_index: dict[str, set[str]], k: int = 10
) -> pd.DataFrame:
    """variants: {"raw_whole_mix_mert": {track_id: vec}, "per_stem_mert": {...},
    "trained_heads": {...}} -> the ablation table from the plan (§Phase 2 eval item 3)."""
    rows = []
    for name, embeddings in variants.items():
        p = precision_at_k(embeddings, tag_index, k=k)
        rows.append({"variant": name, f"precision@{k}": p})
    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=500)
    args = parser.parse_args()

    tag_index = build_tag_index(args.metadata)
    df = pd.read_parquet(args.embeddings)
    embeddings = {str(row["track_id"]): np.asarray(row["embedding"]) for _, row in df.iterrows()}

    p = precision_at_k(embeddings, tag_index, k=args.k, n_queries=args.n_queries)
    print(f"precision@{args.k}: {p:.4f}")


if __name__ == "__main__":
    main()
