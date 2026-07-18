"""Triplet accuracy: for held-out (anchor, positive, negative) triplets,
does the model rank the positive closer to the anchor than the negative,
under the aspect-specific embedding?

The simplest sanity check that a head learned the intended invariance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(a @ b)


def triplet_accuracy(
    triplets: list[dict], embeddings: dict[str, np.ndarray]
) -> dict[str, float]:
    """triplets: list of {anchor_track_id, positive_track_id, negative_track_id, factor, difficulty}
    embeddings: track_id -> embedding vector (already the correct aspect's projection)

    Returns accuracy overall and broken out per factor and per difficulty tier.
    """
    results = {"overall": [], "by_factor": {}, "by_difficulty": {}}
    correct_flags = []

    for t in triplets:
        aid, pid, nid = t["anchor_track_id"], t["positive_track_id"], t["negative_track_id"]
        if aid not in embeddings or pid not in embeddings or nid not in embeddings:
            continue
        sim_pos = cosine_sim(embeddings[aid], embeddings[pid])
        sim_neg = cosine_sim(embeddings[aid], embeddings[nid])
        correct = sim_pos > sim_neg
        correct_flags.append(correct)

        results["by_factor"].setdefault(t["factor"], []).append(correct)
        results["by_difficulty"].setdefault(t["difficulty"], []).append(correct)

    summary = {"overall": float(np.mean(correct_flags)) if correct_flags else float("nan")}
    for factor, flags in results["by_factor"].items():
        summary[f"factor:{factor}"] = float(np.mean(flags))
    for diff, flags in results["by_difficulty"].items():
        summary[f"difficulty:{diff}"] = float(np.mean(flags))

    return summary


def load_triplets(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", type=Path, required=True)
    parser.add_argument(
        "--embeddings", type=Path, required=True,
        help="parquet with columns track_id, embedding (post-head aspect vectors)",
    )
    args = parser.parse_args()

    import pandas as pd

    triplets = load_triplets(args.triplets)
    df = pd.read_parquet(args.embeddings)
    embeddings = {
        str(row["track_id"]): np.asarray(row["embedding"]) for _, row in df.iterrows()
    }

    summary = triplet_accuracy(triplets, embeddings)
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
