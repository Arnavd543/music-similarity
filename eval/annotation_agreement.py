"""Inter-annotator agreement (Krippendorff's alpha) for the human similarity
ratings collected in Phase 2's annotation campaign. Reviewers will look for
this when the paper reports the gold-set filtering criteria.

Expects a long-format dataframe: triplet_id | annotator_id | rating
(rating is an ordinal similarity judgment, e.g. 1-5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _reliability_matrix(df: pd.DataFrame) -> np.ndarray:
    """Rows = items (triplets), columns = annotators, NaN where unrated."""
    pivot = df.pivot_table(
        index="triplet_id", columns="annotator_id", values="rating", aggfunc="first"
    )
    return pivot.to_numpy(dtype=float)


def krippendorff_alpha(df: pd.DataFrame, level: str = "ordinal") -> float:
    """Minimal Krippendorff's alpha implementation for ordinal/interval data
    (no external dependency). For nominal data pass level='nominal'."""
    matrix = _reliability_matrix(df)
    n_items, n_annotators = matrix.shape

    def delta(a: float, b: float) -> float:
        if level == "nominal":
            return 0.0 if a == b else 1.0
        return (a - b) ** 2  # interval/ordinal approx

    # Observed disagreement
    pairs = []
    for i in range(n_items):
        row = matrix[i]
        vals = row[~np.isnan(row)]
        if len(vals) < 2:
            continue
        for a in range(len(vals)):
            for b in range(a + 1, len(vals)):
                pairs.append(delta(vals[a], vals[b]))
    if not pairs:
        return float("nan")
    d_o = np.mean(pairs)

    # Expected disagreement over all rated values, pooled
    all_vals = matrix[~np.isnan(matrix)]
    exp_pairs = []
    for a in range(len(all_vals)):
        for b in range(a + 1, len(all_vals)):
            exp_pairs.append(delta(all_vals[a], all_vals[b]))
    d_e = np.mean(exp_pairs) if exp_pairs else float("nan")

    if d_e == 0:
        return 1.0
    return 1 - d_o / d_e


def filter_high_agreement(
    df: pd.DataFrame, min_annotators: int = 3, max_std: float = 1.0
) -> pd.DataFrame:
    """Gold-set filter: keep triplets with enough raters and low rating
    spread (annotators broadly agree)."""
    stats = df.groupby("triplet_id")["rating"].agg(["count", "std"])
    keep_ids = stats[(stats["count"] >= min_annotators) & (stats["std"] <= max_std)].index
    return df[df["triplet_id"].isin(keep_ids)]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True, help="csv: triplet_id,annotator_id,rating")
    parser.add_argument("--level", default="ordinal", choices=["ordinal", "nominal"])
    args = parser.parse_args()

    df = pd.read_csv(args.ratings)
    alpha = krippendorff_alpha(df, level=args.level)
    print(f"Krippendorff's alpha ({args.level}): {alpha:.4f}")

    gold = filter_high_agreement(df)
    print(f"Gold-set triplets (>=3 raters, std<=1.0): {gold['triplet_id'].nunique()}")


if __name__ == "__main__":
    main()
