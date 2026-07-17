"""Extract frozen MERT embeddings per stem and write parquet shards.

For each track that has separated stems, embed each of the 4 stems (plus
the original mix, kept as a whole-mix baseline vector) with frozen
MERT-v1-95M, mean-pool over time, and store to parquet:

    track_id | stem | embedding (list[float768])

This parquet is the input to both training/aspect_heads.py (frozen
features -> projection heads) and pipeline/qdrant_index.py (raw per-stem
vectors, useful as an ablation baseline against the trained heads).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

from pipeline.config import MODEL, PATHS, STEM_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class MertEmbedder:
    def __init__(self, model_name: str = MODEL.mert_model, device: str | None = None):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
        self.sample_rate = self.processor.sampling_rate

    @torch.no_grad()
    def embed(self, wav: torch.Tensor, sr: int) -> np.ndarray:
        """wav: mono float tensor. Returns a single (768,) mean-pooled embedding
        (averaged across MERT's 13 hidden-state layers, then over time)."""
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        inputs = self.processor(
            wav.numpy(), sampling_rate=self.sample_rate, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        # (num_layers=13, time, 768) -> mean over layers, then over time
        hidden = torch.stack(outputs.hidden_states, dim=1).squeeze(0)  # (13, T, 768)
        layer_mean = hidden.mean(dim=0)  # (T, 768)
        pooled = layer_mean.mean(dim=0)  # (768,)
        return pooled.cpu().numpy()


def iter_track_stems(stems_dir: Path):
    for track_dir in sorted(p for p in stems_dir.iterdir() if p.is_dir()):
        track_id = track_dir.name
        for stem in STEM_NAMES:
            wav_path = track_dir / f"{stem}.wav"
            if wav_path.exists():
                yield track_id, stem, wav_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stems-dir", type=Path, default=PATHS.stems_dir)
    parser.add_argument("--out", type=Path, default=PATHS.embeddings_dir / "mert_embeddings.parquet")
    parser.add_argument("--model", default=MODEL.mert_model)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    embedder = MertEmbedder(args.model)

    items = list(iter_track_stems(args.stems_dir))
    items = [item for i, item in enumerate(items) if i % args.num_shards == args.shard_index]
    log.info("Embedding %d (track, stem) pairs", len(items))

    rows = []
    for track_id, stem, wav_path in items:
        try:
            wav, sr = torchaudio.load(str(wav_path))
            emb = embedder.embed(wav, sr)
            rows.append({"track_id": track_id, "stem": stem, "embedding": emb.tolist()})
        except Exception:
            log.exception("Failed embedding %s/%s", track_id, stem)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shard_out = args.out.with_name(f"{args.out.stem}_shard{args.shard_index}{args.out.suffix}")
    pd.DataFrame(rows).to_parquet(shard_out)
    log.info("Wrote %d rows to %s", len(rows), shard_out)


if __name__ == "__main__":
    main()
