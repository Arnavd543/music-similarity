from training.triplet_engine import DIFFICULTY_TIERS, FACTORS, generate_triplet_pool


def test_generate_triplet_pool_covers_all_factors_and_tiers():
    song_ids = [f"song_{i}" for i in range(50)]
    triplets = generate_triplet_pool(song_ids, n_per_factor=30, seed=0)

    assert len(triplets) == 30 * len(FACTORS)
    factors_seen = {t.factor for t in triplets}
    assert factors_seen == set(FACTORS)

    difficulties_seen = {t.difficulty for t in triplets}
    assert difficulties_seen <= set(DIFFICULTY_TIERS)


def test_triplets_reference_distinct_songs():
    song_ids = [f"song_{i}" for i in range(20)]
    triplets = generate_triplet_pool(song_ids, n_per_factor=10, seed=1)
    for t in triplets:
        assert t.anchor_track_id != t.negative_track_id


def test_deterministic_given_seed():
    song_ids = [f"song_{i}" for i in range(20)]
    a = generate_triplet_pool(song_ids, n_per_factor=10, seed=7)
    b = generate_triplet_pool(song_ids, n_per_factor=10, seed=7)
    assert [t.triplet_id for t in a] == [t.triplet_id for t in b]
