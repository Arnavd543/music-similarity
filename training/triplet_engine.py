"""Paper-track: factor-controlled triplet generation from real multitrack
stems (MoisesDB / MUSDB18-HQ / Slakh2100), not generated audio.

This is the benchmark's core artifact and doubles as the head-training
data generator for Phase 2. Each triplet is (anchor, positive, negative)
for one factor, built by *real* stem manipulation:

  - rhythm-positive:   swap vocals/other for a different song's vocals/other,
                        keep the anchor's drums+bass (rhythm section) intact
                        -> positive should still sound rhythmically similar
                        to the anchor. Bass stays with the drums because it's
                        rhythmically locked to them in real mixes (kick-bass
                        interlock, walking bass); swapping it independently
                        clashes against the anchor's actual groove regardless
                        of tempo match.
  - melody-positive:   re-timbre via stem substitution (same MIDI/melodic
                        line, different instrument sample) or tempo-warp,
                        keeping melodic content -> positive.
  - timbre-positive:   same instrumentation/sample set, different song
                        section (so pitch/rhythm differ but timbre doesn't).

Manipulation strength is parameterized (`difficulty`) so the benchmark has
graded tiers -- a differentiator over MERIT's binary triplets.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline.config import PATHS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FACTORS = ["rhythm", "melody", "timbre"]
DIFFICULTY_TIERS = ["easy", "medium", "hard"]

# How strongly each tier perturbs the non-target stems -- larger = more
# aggressive substitution/warping, i.e. a *harder* discrimination task.
TIER_STRENGTH = {"easy": 0.3, "medium": 0.6, "hard": 0.9}


@dataclass
class Triplet:
    triplet_id: str
    factor: str
    difficulty: str
    anchor_track_id: str
    positive_track_id: str
    positive_recipe: str  # how the positive was constructed, for provenance
    negative_track_id: str
    source_corpus: str  # moisesdb | musdb18hq | slakh2100
    donor_track_id: str | None = None  # rhythm triplets: source of the swapped pitched stems


def list_multitrack_songs(corpus_dir: Path, zip_root: str = "") -> list[str]:
    """Each subdirectory of corpus_dir is treated as one song with its own
    stem files (corpus-specific stem-folder conventions are normalized
    upstream by a per-corpus loader; this module only needs song IDs).
    Accepts either an extracted directory or a .zip archive."""
    if corpus_dir.suffix.lower() == ".zip":
        from training.triplet_render import SongSource

        return SongSource(corpus_dir, zip_root=zip_root).list_songs()
    return sorted(p.name for p in corpus_dir.iterdir() if p.is_dir())


def build_rhythm_triplet(anchor_id: str, donor_id: str, negative_id: str, difficulty: str) -> Triplet:
    recipe = (
        f"keep anchor drums+bass (rhythm section); replace vocals/other with "
        f"donor={donor_id}'s vocals/other at strength={TIER_STRENGTH[difficulty]}"
    )
    return Triplet(
        triplet_id=f"rhythm_{anchor_id}_{donor_id}_{difficulty}",
        factor="rhythm",
        difficulty=difficulty,
        anchor_track_id=anchor_id,
        positive_track_id=f"{anchor_id}+drums_bass__{donor_id}+vocals_other",
        positive_recipe=recipe,
        negative_track_id=negative_id,
        source_corpus="moisesdb",
        donor_track_id=donor_id,
    )


def build_melody_triplet(anchor_id: str, negative_id: str, difficulty: str) -> Triplet:
    tempo_delta = 1.0 + TIER_STRENGTH[difficulty] * random.choice([-0.3, 0.3])
    recipe = f"tempo-warp anchor's melodic stems by rate={tempo_delta:.2f}, keep melodic content"
    return Triplet(
        triplet_id=f"melody_{anchor_id}_{negative_id}_{difficulty}",
        factor="melody",
        difficulty=difficulty,
        anchor_track_id=anchor_id,
        positive_track_id=f"{anchor_id}+tempowarp{tempo_delta:.2f}",
        positive_recipe=recipe,
        negative_track_id=negative_id,
        source_corpus="moisesdb",
    )


def build_timbre_triplet(anchor_id: str, negative_id: str, difficulty: str) -> Triplet:
    recipe = f"same instrumentation, disjoint section, section-gap strength={TIER_STRENGTH[difficulty]}"
    return Triplet(
        triplet_id=f"timbre_{anchor_id}_{negative_id}_{difficulty}",
        factor="timbre",
        difficulty=difficulty,
        anchor_track_id=anchor_id,
        positive_track_id=f"{anchor_id}+section2",
        positive_recipe=recipe,
        negative_track_id=negative_id,
        source_corpus="moisesdb",
    )


def generate_triplet_pool(
    song_ids: list[str], n_per_factor: int = 2000, seed: int = 42
) -> list[Triplet]:
    rng = random.Random(seed)
    triplets: list[Triplet] = []

    for factor in FACTORS:
        for _ in range(n_per_factor):
            anchor_id, donor_id, negative_id = rng.sample(song_ids, 3)
            difficulty = rng.choice(DIFFICULTY_TIERS)
            if factor == "rhythm":
                t = build_rhythm_triplet(anchor_id, donor_id, negative_id, difficulty)
            elif factor == "melody":
                t = build_melody_triplet(anchor_id, negative_id, difficulty)
            else:
                t = build_timbre_triplet(anchor_id, negative_id, difficulty)
            triplets.append(t)

    return triplets


def save_pool(triplets: list[Triplet], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for t in triplets:
            f.write(json.dumps(asdict(t)) + "\n")
    log.info("Wrote %d triplets to %s", len(triplets), out_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True,
                        help="extracted corpus dir or .zip archive")
    parser.add_argument("--n-per-factor", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=PATHS.data_dir / "triplets" / "pool.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zip-root", default="",
                        help="'train' for MUSDB18-HQ zip, 'moisesdb/moisesdb_v0.1' for MoisesDB zip")
    args = parser.parse_args()

    song_ids = list_multitrack_songs(args.corpus_dir, zip_root=args.zip_root)
    log.info("Found %d multitrack songs in %s", len(song_ids), args.corpus_dir)

    triplets = generate_triplet_pool(song_ids, n_per_factor=args.n_per_factor, seed=args.seed)
    save_pool(triplets, args.out)


if __name__ == "__main__":
    main()
