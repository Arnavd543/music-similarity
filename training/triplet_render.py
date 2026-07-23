"""Render triplet recipes (training/triplet_engine.py output) to actual audio.

Reads a pool.jsonl of Triplet recipes and writes, per triplet:

    <out_dir>/<triplet_id>/anchor.wav
    <out_dir>/<triplet_id>/positive.wav
    <out_dir>/<triplet_id>/negative.wav

Rendering per factor:
  rhythm  -- anchor's drums + bass (rhythm section) kept intact, donor
             song's vocals/other swapped in at gain scaled by the difficulty
             tier (stronger donor = harder discrimination).
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
# Rhythm-positive swap set: vocals/other only. Bass stays with the anchor's
# drums (rhythm section) -- swapping a donor's bass in independently clashes
# against the anchor's actual groove (kick-bass lock, walking bass, etc.)
# regardless of tempo match or gain, which is what made "easy" rhythm
# triplets read as subtly off and "hard" ones read as barely related.
RHYTHM_SWAP_STEMS = ("vocals", "other")
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


class SongSource:
    """Resolves song IDs to on-disk stem directories, from either an
    extracted corpus directory or a .zip archive. Zips are read song-by-song
    into a bounded extraction cache, so a 100GB corpus never needs to be
    fully extracted (only `cache_songs` songs' stems exist on disk at once)."""

    def __init__(self, corpus_path: Path, zip_root: str = "", cache_songs: int = 4):
        self.corpus_path = corpus_path
        self.zip_root = zip_root.strip("/")
        self.cache_songs = cache_songs
        self._zip = None
        self._cache_dir: Path | None = None
        self._lru: list[str] = []
        if corpus_path.suffix.lower() == ".zip":
            import zipfile

            self._zip = zipfile.ZipFile(corpus_path)
            self._cache_dir = Path(tempfile.mkdtemp(prefix="triplet_zip_cache_"))

    def _prefix(self, song_id: str) -> str:
        return f"{self.zip_root}/{song_id}/" if self.zip_root else f"{song_id}/"

    def list_songs(self) -> list[str]:
        if self._zip is None:
            return sorted(p.name for p in self.corpus_path.iterdir() if p.is_dir())
        depth = self.zip_root.count("/") + 1 if self.zip_root else 0
        songs = set()
        for n in self._zip.namelist():
            parts = n.strip("/").split("/")
            if len(parts) > depth and (not self.zip_root or n.startswith(self.zip_root + "/")):
                songs.add(parts[depth])
        return sorted(s for s in songs if s)

    def song_dir(self, song_id: str) -> Path:
        if self._zip is None:
            return self.corpus_path / song_id
        dest = self._cache_dir / self._prefix(song_id)
        if not dest.exists():
            members = [n for n in self._zip.namelist() if n.startswith(self._prefix(song_id))]
            self._zip.extractall(self._cache_dir, members=members)
            self._lru.append(song_id)
            while len(self._lru) > self.cache_songs:
                evict = self._lru.pop(0)
                import shutil

                shutil.rmtree(self._cache_dir / self._prefix(evict), ignore_errors=True)
        elif song_id in self._lru:
            self._lru.remove(song_id)
            self._lru.append(song_id)
        return dest


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
    t: dict, source: SongSource, corpus: str, out_dir: Path, sr: int = 44100
) -> bool:
    """Render one recipe dict (a Triplet as JSON). Returns False (and logs)
    on unrenderable songs, e.g. missing stems."""
    rng = random.Random(t["triplet_id"])  # reproducible segment choices
    load = LOADERS[corpus]
    strength = TIER_STRENGTH[t["difficulty"]]
    tdir = out_dir / t["triplet_id"]
    if (tdir / "negative.wav").exists():
        return True  # resumable

    anchor_stems = load(source.song_dir(t["anchor_track_id"]), sr)
    negative_stems = load(source.song_dir(t["negative_track_id"]), sr)
    if "drums" not in anchor_stems or not negative_stems:
        log.warning("Skipping %s: missing required stems", t["triplet_id"])
        return False

    start = rng.uniform(0.2, 0.5)
    anchor = _segment(_mix(anchor_stems), sr, start)

    factor = t["factor"]
    if factor == "rhythm":
        donor_id = t.get("donor_track_id") or t["positive_track_id"].split("__")[-1].removesuffix("+vocals_other")
        donor_stems = load(source.song_dir(donor_id), sr)
        swapped = {k: v for k, v in donor_stems.items() if k in RHYTHM_SWAP_STEMS}
        kept = {k: v for k, v in anchor_stems.items() if k in ("drums", "bass")}
        positive = _segment(
            _mix({**kept, **swapped},
                 gains={k: strength for k in swapped}),
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


def render_pool(
    pool_path: Path, corpus_dir: Path, corpus: str, out_dir: Path, zip_root: str = ""
) -> tuple[int, int]:
    from tqdm import tqdm

    source = SongSource(corpus_dir, zip_root=zip_root)
    triplets = [json.loads(line) for line in pool_path.open() if line.strip()]
    ok = 0
    for t in tqdm(triplets, desc="rendering", unit="triplet"):
        try:
            ok += render_triplet(t, source, corpus, out_dir)
        except Exception:
            log.exception("Failed rendering %s", t.get("triplet_id"))
    return ok, len(triplets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True,
                        help="extracted corpus dir, or a .zip archive (read song-by-song)")
    parser.add_argument("--corpus", choices=list(LOADERS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zip-root", default="",
                        help="path inside the zip to the song dirs, e.g. 'train' (MUSDB18-HQ) or 'moisesdb/moisesdb_v0.1' (MoisesDB)")
    args = parser.parse_args()

    ok, total = render_pool(args.pool, args.corpus_dir, args.corpus, args.out, zip_root=args.zip_root)
    log.info("Rendered %d/%d triplets to %s", ok, total, args.out)


if __name__ == "__main__":
    main()
