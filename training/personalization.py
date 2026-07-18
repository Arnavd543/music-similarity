"""Phase 4: learn per-user slider weights from pairwise feedback.

UI shows the user two candidates for a seed song and asks "which is more
similar?"; each answer is one pairwise judgment. We fit a per-user weight
vector w over the aspect similarities via Bradley-Terry (equivalently,
logistic regression on the *difference* of per-aspect cosine similarities),
so P(user picks A over B) = sigmoid(w . (sim(seed,A) - sim(seed,B))).

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from pipeline.config import ASPECTS


@dataclass
class PairwiseJudgment:
    seed_sims_a: dict[str, float]  # per-aspect cosine sim of candidate A to seed
    seed_sims_b: dict[str, float]  # per-aspect cosine sim of candidate B to seed
    chose_a: bool  # True if the user judged A more similar than B


def _vectorize(sims: dict[str, float]) -> np.ndarray:
    return np.array([sims.get(a, 0.0) for a in ASPECTS], dtype=np.float64)


def fit_user_weights(
    judgments: list[PairwiseJudgment], prior_weights: np.ndarray | None = None
) -> dict[str, float]:
    """Returns a non-negative, sum-to-1 weight per aspect (so it plugs
    directly into the slider UI / server-side fusion score)."""
    if len(judgments) < 5:
        # Not enough signal yet -- fall back to uniform weights (equivalent
        # to "sliders all centered") rather than overfitting to a handful
        # of clicks.
        n = len(ASPECTS)
        return {a: 1.0 / n for a in ASPECTS}

    X = np.stack([_vectorize(j.seed_sims_a) - _vectorize(j.seed_sims_b) for j in judgments])
    y = np.array([1 if j.chose_a else 0 for j in judgments])

    # If everyone always says "A" or always "B" (degenerate), logistic
    # regression on a single class fails -- guard with uniform fallback.
    if len(set(y.tolist())) < 2:
        n = len(ASPECTS)
        return {a: 1.0 / n for a in ASPECTS}

    clf = LogisticRegression(fit_intercept=False, C=1.0)
    clf.fit(X, y)
    raw = clf.coef_[0]

    # Bradley-Terry weights should be non-negative (a "more similar on this
    # axis -> more likely chosen" direction); clip and renormalize to sum 1
    # so they map onto slider positions.
    weights = np.clip(raw, a_min=0.0, a_max=None)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    return dict(zip(ASPECTS, weights.tolist(), strict=True))
