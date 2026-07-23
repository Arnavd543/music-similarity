"""Embed rendered triplet audio (training/triplet_render.py output) through
frozen MERT + the trained AspectHeads, producing the (track_id, embedding)
parquet that eval/triplet_accuracy.py scores machine baselines against.

GPU-bound (Demucs + MERT per clip, unlike triplet_render.py which is
CPU-only) -- meant to run on Colab, not laptop CPU.

Anchor/positive/negative each get a synthetic track_id
(f"{triplet_id}::{role}") since the pool's own track_id fields aren't 1:1
with rendered audio (positive_track_id is a recipe description, not a real
song id). A remapped copy of the pool jsonl carries those same synthetic
ids as anchor/positive/negative_track_id, so eval/triplet_accuracy.py's
plain track_id join works against it unmodified.

Prerequisite on Colab: rhythm_head.pt / melody_head.pt / timbre_head.pt are
gitignored (*.pt) and were never committed, so `git pull` will NOT bring
them over -- copy them from the laptop (e.g. via Drive) before running this.

Usage (Colab):
    python -m eval.embed_triplets \
        --pool data/triplets/pool_pilot.jsonl \
        --audio-dir data/triplets/audio_pilot \
        --checkpoints-dir . \
        --out-embeddings data/triplets/embeddings_pilot.parquet \
        --out-triplets data/triplets/pool_pilot_remapped.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import pandas as pd
import torch

from pipeline.colab_pipeline import process_one_track
from pipeline.config import ASPECT_STEM_MAP
from training.aspect_heads import AspectHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROLES = ("anchor", "positive", "negative")


def load_heads(checkpoints_dir: Path, factors: set[str]) -> dict[str, AspectHead]:
    heads = {}
    for factor in factors:
        head = AspectHead()
        head.load_state_dict(torch.load(checkpoints_dir / f"{factor}_head.pt", map_location="cpu"))
        head.eval()
        heads[factor] = head
    return heads


def project(stem_embeddings: dict[str, list[float]], factor: str, head: AspectHead) -> list[float]:
    stems = ASPECT_STEM_MAP[factor]
    available = [torch.tensor(stem_embeddings[s]) for s in stems if s in stem_embeddings]
    if not available:
        raise ValueError(f"none of {stems} present for factor={factor}")
    pooled = torch.stack(available, dim=0).mean(dim=0)
    with torch.no_grad():
        return head(pooled.unsqueeze(0)).squeeze(0).tolist()


def embed_pool(
    triplets: list[dict],
    audio_dir: Path,
    heads: dict[str, AspectHead],
    demucs_model,
    embedder,
    device: str,
) -> tuple[list[dict], list[dict]]:
    scratch = Path(tempfile.mkdtemp(prefix="embed_triplets_"))
    emb_rows: list[dict] = []
    remapped: list[dict] = []

    for t in triplets:
        head = heads[t["factor"]]
        tdir = audio_dir / t["triplet_id"]
        ids = {}
        ok = True
        for role in ROLES:
            wav_path = tdir / f"{role}.wav"
            if not wav_path.exists():
                log.warning("Missing %s, skipping triplet %s", wav_path, t["triplet_id"])
                ok = False
                break
            row = process_one_track(wav_path, demucs_model, embedder, scratch, device)
            try:
                emb = project(row["embeddings"], t["factor"], head)
            except ValueError as e:
                log.warning("Skipping triplet %s: %s", t["triplet_id"], e)
                ok = False
                break
            key = f"{t['triplet_id']}::{role}"
            emb_rows.append({"track_id": key, "embedding": emb})
            ids[role] = key
        if not ok:
            continue
        nt = dict(t)
        nt["anchor_track_id"] = ids["anchor"]
        nt["positive_track_id"] = ids["positive"]
        nt["negative_track_id"] = ids["negative"]
        remapped.append(nt)

    return emb_rows, remapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("."))
    parser.add_argument("--out-embeddings", type=Path, required=True)
    parser.add_argument("--out-triplets", type=Path, required=True)
    parser.add_argument("--demucs-model", default="htdemucs")
    args = parser.parse_args()

    from demucs.pretrained import get_model

    from pipeline.config import MODEL
    from pipeline.mert_embed import MertEmbedder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading Demucs + MERT on %s", device)
    demucs_model = get_model(args.demucs_model)
    demucs_model.to(device)
    demucs_model.eval()
    embedder = MertEmbedder(MODEL.mert_model, device=device)

    triplets = [json.loads(line) for line in args.pool.open() if line.strip()]
    heads = load_heads(args.checkpoints_dir, {t["factor"] for t in triplets})

    emb_rows, remapped = embed_pool(triplets, args.audio_dir, heads, demucs_model, embedder, device)

    args.out_embeddings.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(emb_rows).to_parquet(args.out_embeddings)
    log.info("Wrote %d embeddings to %s", len(emb_rows), args.out_embeddings)

    with args.out_triplets.open("w") as f:
        for nt in remapped:
            f.write(json.dumps(nt) + "\n")
    log.info("Wrote %d remapped triplets to %s", len(remapped), args.out_triplets)


if __name__ == "__main__":
    main()
