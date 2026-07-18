"""Phase 0 walking skeleton: whole-mix MERT embedding -> FAISS -> nearest
neighbors. No stems, no heads. The comparison baseline every later
ablation is measured against.

Usage:
    python -m pipeline.baseline_faiss build --audio-dir data/audio --out data/embeddings/baseline.index
    python -m pipeline.baseline_faiss query --index data/embeddings/baseline.index --track-id 12345 --k 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torchaudio

from pipeline.config import PATHS
from pipeline.mert_embed import MertEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_index(audio_dir: Path, out_index: Path) -> None:
    import faiss

    embedder = MertEmbedder()
    files = sorted(p for p in audio_dir.glob("*") if p.suffix.lower() in (".mp3", ".wav", ".flac"))
    log.info("Embedding %d whole-mix tracks for the baseline index", len(files))

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    for f in files:
        try:
            wav, sr = torchaudio.load(str(f))
            vec = embedder.embed(wav, sr)
            vectors.append(vec)
            ids.append(f.stem)
        except Exception:
            log.exception("Failed embedding %s", f)

    matrix = np.stack(vectors).astype("float32")
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(matrix.shape[1])  # cosine sim via normalized inner product
    index.add(matrix)

    out_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_index))
    with out_index.with_suffix(".ids.json").open("w") as fh:
        json.dump(ids, fh)
    log.info("Wrote FAISS index (%d vectors) to %s", len(ids), out_index)


def query_index(index_path: Path, track_id: str, k: int = 10) -> list[tuple[str, float]]:
    import faiss

    index = faiss.read_index(str(index_path))
    ids = json.loads(index_path.with_suffix(".ids.json").read_text())

    if track_id not in ids:
        raise ValueError(f"{track_id} not in baseline index")
    row = ids.index(track_id)
    query_vec = index.reconstruct(row).reshape(1, -1)

    scores, neighbors = index.search(query_vec, k + 1)  # +1 to drop self-match
    results = [
        (ids[i], float(s)) for i, s in zip(neighbors[0], scores[0], strict=True) if ids[i] != track_id
    ]
    return results[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build")
    build_p.add_argument("--audio-dir", type=Path, default=PATHS.audio_dir)
    build_p.add_argument("--out", type=Path, default=PATHS.embeddings_dir / "baseline.index")

    query_p = sub.add_parser("query")
    query_p.add_argument("--index", type=Path, default=PATHS.embeddings_dir / "baseline.index")
    query_p.add_argument("--track-id", required=True)
    query_p.add_argument("--k", type=int, default=10)

    args = parser.parse_args()
    if args.command == "build":
        build_index(args.audio_dir, args.out)
    else:
        for track_id, score in query_index(args.index, args.track_id, args.k):
            print(f"{track_id}\t{score:.4f}")


if __name__ == "__main__":
    main()
