import numpy as np

from pipeline.config import ASPECTS
from training.personalization import PairwiseJudgment, fit_user_weights


def _judgment(winning_aspect: str, chose_a: bool = True) -> PairwiseJudgment:
    sims_a = {a: 0.5 for a in ASPECTS}
    sims_b = {a: 0.5 for a in ASPECTS}
    if chose_a:
        sims_a[winning_aspect] = 0.9
    else:
        sims_b[winning_aspect] = 0.9
    return PairwiseJudgment(seed_sims_a=sims_a, seed_sims_b=sims_b, chose_a=chose_a)


def test_uniform_fallback_with_few_judgments():
    weights = fit_user_weights([_judgment("rhythm")])
    assert set(weights) == set(ASPECTS)
    assert all(abs(w - 1 / len(ASPECTS)) < 1e-9 for w in weights.values())


def test_weights_sum_to_one():
    judgments = [_judgment("rhythm", chose_a=(i % 2 == 0)) for i in range(20)]
    weights = fit_user_weights(judgments)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(w >= 0 for w in weights.values())


def test_learns_dominant_axis():
    # Every judgment: user picks whichever candidate has higher rhythm sim.
    rng = np.random.default_rng(0)
    judgments = []
    for _ in range(200):
        sims_a = {a: rng.uniform(0.3, 0.7) for a in ASPECTS}
        sims_b = {a: rng.uniform(0.3, 0.7) for a in ASPECTS}
        chose_a = sims_a["rhythm"] > sims_b["rhythm"]
        judgments.append(PairwiseJudgment(seed_sims_a=sims_a, seed_sims_b=sims_b, chose_a=chose_a))

    weights = fit_user_weights(judgments)
    assert weights["rhythm"] == max(weights.values())
