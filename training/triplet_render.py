"""Render triplet recipes (training/triplet_engine.py output) to actual audio.

Reads a pool.jsonl of Triplet recipes and writes, per triplet:

    <out_dir>/<triplet_id>/anchor.wav
    <out_dir>/<triplet_id>/positive.wav
    <out_dir>/<triplet_id>/negative.wav

Rendering per factor:
  rhythm  -- anchor's drums + donor song's pitched stems, donor gain scaled
             by the difficulty tier (stronger donor = harder discrimination).
  melody  -- anchor mix tempo-warped (ffmpeg atempo: tempo changes, pitch and
             melodic content preserved).
  timbre  -- same song, a different section; section gap scales with tier.

CPU-only: numpy + soundfile for audio, ffmpeg for tempo warping. No torch.

Usage:
    python -m training.triplet_render --pool data/triplets/pool.jsonl \
        --corpus-dir data/musdb18hq/train --corpus musdb --out data/triplets/audio
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CANONICAL_STEMS = ("drums", "bass", "vocals", "other")
PITCHED_STEMS = ("bass", "vocals", "other")
TIER_STRENGTH = {"easy": 0.3, "medium": 0.6, "hard": 0.9}
TARGET_RMS = 0.1
SEGMENT_SECONDS = 30.0

# MoisesDB folder names -> canonical stem buckets. MoisesDB splits by
# instrument (per-folder wavs); everything pitched that isn't bass or vocals
# lands in `other`, mirroring the htdemucs convention used by the app pipeline.
_MOISES_MAP = {
    "drums": "drums", "percussion": "drums",
    "bass": "bass",
    "vocals": "vocals", "vox": "vocals",
}


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr  # mono


def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    n_out = int(round(len(x) * sr_to / sr_from))
    return np.interp(
        np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x
    ).astype(np.float32)


def load_stems_musdb(song_dir: Path, sr: int = 44100) -> dict[str, np.ndarray]:
    """MUSDB18-HQ layout: <song>/{drums,bass,vocals,other}.wav"""
    stems = {}
    for name in CANONICAL_STEMS:
        p = song_dir / f"{name}.wav"
        if p.exists():
            x, file_sr = _read_wav(p)
            stems[name] = _resample(x, file_sr, sr)
    return stems


def load_stems_moisesdb(song_dir: Path, sr: int = 44100) -> dict[str, np.ndarray]:
    """MoisesDB layout: <song>/<instrument_folder>/*.wav, one folder per
    instrument type. Folders are bucketed into the canonical 4 stems and
    summed (a song can have e.g. two guitar folders -> both into `other`)."""
    buckets: dict[str, list[np.ndarray]] = {}
    for sub in sorted(p for p in song_dir.iterdir() if p.is_dir()):
        bucket = _MOISES_MAP.get(sub.name.lower(), "other")
        for wav in sorted(sub.glob("*.wav")):
            x, file_sr = _read_wav(wav)
            buckets.setdefault(bucket, []).append(_resample(x, file_sr, sr))
    stems = {}
    for bucket, parts in buckets.items():
        n = max(len(p) for p in parts)
        acc = np.zeros(n, dtype=np.float32)
        for p in parts:
            acc[: len(p)] += p
        stems[bucket] = acc
    return stems


LOADERS = {"musdb": load_stems_musdb, "moisesdb": load_stems_moisesdb}


def _mix(stems: dict[str, np.ndarray], gains: dict[str, float] | None = None) -> np.ndarray:
    n = max(len(x) for x in stems.values())
    acc = np.zeros(n, dtype=np.float32)
    for name, x in stems.items():
        g = (gains or {}).get(name, 1.0)
        acc[: len(x)] += g * x
    return acc


def _segment(x: np.ndarray, sr: int, start_frac: float, seconds: float = SEGMENT_SECONDS) -> np.ndarray:
    n = int(seconds * sr)
    if len(x) <= n:
        return x
    start = int(min(start_frac, 1.0 - n / len(x)) * len(x))
    return x[start : start + n]


def _normalize(x: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    rms = float(np.sqrt((x**2).mean()) + 1e-8)
    y = x * (target_rms / rms)
    peak = float(np.abs(y).max() + 1e-8)
    return y / peak * 0.99 if peak > 0.99 else y


def _tempo_warp_ffmpeg(x: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """ffmpeg atempo: changes tempo without changing pitch. atempo only
    accepts 0.5-2.0 per instance, which covers the recipe range."""
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(str(src), x, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src),
             "-filter:a", f"atempo={rate:.4f}", str(dst)],
            check=True, capture_output=True, timeout=120,
        )
        y, _ = _read_wav(dst)
    return y


def render_triplet(
    t: dict, corpus_dir: Path, corpus: str, out_dir: Path, sr: int = 44100
) -> bool:
    """Render one recipe dict (a Triplet as JSON). Returns False (and logs)
    on unrenderable songs, e.g. missing stems."""
    rng = random.Random(t["triplet_id"])  # reproducible segment choices
    load = LOADERS[corpus]
    strength = TIER_STRENGTH[t["difficulty"]]
    tdir = out_dir / t["triplet_id"]
    if (tdir / "negative.wav").exists():
        return True  # resumable

    anchor_stems = load(corpus_dir / t["anchor_track_id"], sr)
    negative_stems = load(corpus_dir / t["negative_track_id"], sr)
    if "drums" not in anchor_stems or not negative_stems:
        log.warning("Skipping %s: missing required stems", t["triplet_id"])
        return False

    start = rng.uniform(0.2, 0.5)
    anchor = _segment(_mix(anchor_stems), sr, start)

    factor = t["factor"]
    if factor == "rhythm":
        donor_id = t.get("donor_track_id") or t["positive_track_id"].split("__")[-1].removesuffix("+pitched")
        donor_stems = load(corpus_dir / donor_id, sr)
        pitched = {k: v for k, v in donor_stems.items() if k in PITCHED_STEMS}
        positive = _segment(
            _mix({"drums": anchor_stems["drums"], **pitched},
                 gains={k: strength for k in pitched}),
            sr, start,
        )
    elif factor == "melody":
        rate = 1.0 + strength * rng.choice([-0.3, 0.3])
        rate = min(2.0, max(0.5, rate))
        positive = _tempo_warp_ffmpeg(anchor, sr, rate)
    else:  # timbre: same song, different section; gap grows with strength
        pos_start = start + 0.15 + 0.3 * strength
        positive = _segment(_mix(anchor_stems), sr, pos_start % 0.85)

    negative = _segment(_mix(negative_stems), sr, rng.uniform(0.2, 0.5))

    tdir.mkdir(parents=True, exist_ok=True)
    for name, audio in (("anchor", anchor), ("positive", positive), ("negative", negative)):
        sf.write(str(tdir / f"{name}.wav"), _normalize(audio), sr)
    return True


def render_pool(pool_path: Path, corpus_dir: Path, corpus: str, out_dir: Path) -> tuple[int, int]:
    from tqdm import tqdm

    triplets = [json.loads(line) for line in pool_path.open() if line.strip()]
    ok = 0
    for t in tqdm(triplets, desc="rendering", unit="triplet"):
        try:
            ok += render_triplet(t, corpus_dir, corpus, out_dir)
        except Exception:
            log.exception("Failed rendering %s", t.get("triplet_id"))
    return ok, len(triplets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--corpus", choices=list(LOADERS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ok, total = render_pool(args.pool, args.corpus_dir, args.corpus, args.out)
    log.info("Rendered %d/%d triplets to %s", ok, total, args.out)


if __name__ == "__main__":
    main()
