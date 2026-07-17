import numpy as np
import pandas as pd

from eval.annotation_agreement import filter_high_agreement, krippendorff_alpha
from eval.precision_at_k import precision_at_k
from eval.triplet_accuracy import triplet_accuracy


def test_triplet_accuracy_perfect_case():
    # anchor closer to positive than negative in embedding space -> accuracy 1.0
    embeddings = {
        "a": np.array([1.0, 0.0]),
        "p": np.array([0.9, 0.1]),
        "n": np.array([-1.0, 0.0]),
    }
    triplets = [
        {"anchor_track_id": "a", "positive_track_id": "p", "negative_track_id": "n",
         "factor": "rhythm", "difficulty": "easy"}
    ]
    summary = triplet_accuracy(triplets, embeddings)
    assert summary["overall"] == 1.0
    assert summary["factor:rhythm"] == 1.0
    assert summary["difficulty:easy"] == 1.0


def test_triplet_accuracy_failure_case():
    embeddings = {
        "a": np.array([1.0, 0.0]),
        "p": np.array([-1.0, 0.0]),  # positive is actually far
        "n": np.array([0.9, 0.1]),  # negative is actually close
    }
    triplets = [
        {"anchor_track_id": "a", "positive_track_id": "p", "negative_track_id": "n",
         "factor": "melody", "difficulty": "hard"}
    ]
    summary = triplet_accuracy(triplets, embeddings)
    assert summary["overall"] == 0.0


def test_precision_at_k_perfect_overlap():
    embeddings = {str(i): np.array([1.0, 0.0]) + np.random.default_rng(i).normal(0, 0.001, 2) for i in range(10)}
    tag_index = {str(i): {"genre---rock"} for i in range(10)}
    p = precision_at_k(embeddings, tag_index, k=5, n_queries=5)
    assert p == 1.0  # every track shares the tag, so every neighbor is relevant


def test_precision_at_k_no_overlap():
    embeddings = {str(i): np.random.default_rng(i).normal(size=2) for i in range(10)}
    tag_index = {str(i): {f"genre---{i}"} for i in range(10)}  # unique tag per track
    p = precision_at_k(embeddings, tag_index, k=5, n_queries=5)
    assert p == 0.0


def test_krippendorff_alpha_perfect_agreement():
    df = pd.DataFrame(
        {
            "triplet_id": ["t1", "t1", "t1", "t2", "t2", "t2"],
            "annotator_id": ["r1", "r2", "r3", "r1", "r2", "r3"],
            "rating": [5, 5, 5, 1, 1, 1],
        }
    )
    alpha = krippendorff_alpha(df)
    assert alpha == 1.0


def test_filter_high_agreement_drops_noisy_triplets():
    df = pd.DataFrame(
        {
            "triplet_id": ["good", "good", "good", "bad", "bad", "bad"],
            "annotator_id": ["r1", "r2", "r3", "r1", "r2", "r3"],
            "rating": [5, 5, 5, 1, 5, 3],
        }
    )
    gold = filter_high_agreement(df, min_annotators=3, max_std=0.5)
    assert set(gold["triplet_id"].unique()) == {"good"}
