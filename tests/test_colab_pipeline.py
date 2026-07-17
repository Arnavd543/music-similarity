import json

from pipeline.colab_pipeline import (
    _append_row,
    _load_completed_ids,
    estimate_feasible_tracks,
    estimate_hours,
    manifest_to_parquet,
)


def test_estimate_hours_known_gpu():
    hours = estimate_hours(100, "T4")
    assert hours is not None
    assert abs(hours - 100 / 1.76) < 1e-6


def test_estimate_hours_unknown_gpu():
    assert estimate_hours(100, "Tesla K80") is None


def test_estimate_feasible_tracks_positive():
    n = estimate_feasible_tracks(100, "T4")
    assert n is not None
    assert n > 0


def test_estimate_feasible_tracks_scales_with_budget():
    small = estimate_feasible_tracks(50, "T4")
    large = estimate_feasible_tracks(200, "T4")
    assert large > small


def test_manifest_resume_roundtrip(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    assert _load_completed_ids(manifest_path) == set()

    _append_row(manifest_path, {"track_id": "a1", "embeddings": {"drums": [0.1, 0.2]}})
    _append_row(manifest_path, {"track_id": "a2", "embeddings": {"drums": [0.3, 0.4]}})

    done = _load_completed_ids(manifest_path)
    assert done == {"a1", "a2"}


def test_manifest_resume_ignores_corrupt_lines(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text('{"track_id": "a1", "embeddings": {}}\nnot json\n')
    assert _load_completed_ids(manifest_path) == {"a1"}


def test_manifest_to_parquet_flattens_rows(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w") as f:
        f.write(json.dumps({"track_id": "t1", "embeddings": {"drums": [1.0, 2.0], "bass": [3.0, 4.0]}}) + "\n")
        f.write(json.dumps({"track_id": "t2", "embeddings": {"drums": [5.0, 6.0]}}) + "\n")

    out_parquet = tmp_path / "out.parquet"
    manifest_to_parquet(manifest_path, out_parquet)

    import pandas as pd

    df = pd.read_parquet(out_parquet)
    assert len(df) == 3  # t1 has 2 stems, t2 has 1
    assert set(df["track_id"]) == {"t1", "t2"}
    assert set(df.columns) == {"track_id", "stem", "embedding"}
