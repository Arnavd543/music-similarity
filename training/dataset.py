"""Dataset for aspect-head training.

Expects two parquet files produced by an offline job that runs
training/augmentations.py over stem waveforms and then pipeline/mert_embed.py
over both the original and augmented audio:

  anchor_embeddings.parquet:    track_id | stem | embedding
  positive_embeddings.parquet:  track_id | stem | embedding   (post-augmentation)

Both are keyed the same way so row i of one dataset is the positive pair
for row i of the other, per aspect (rhythm uses the `drums` stem rows,
melody uses `bass`/`other`, etc. -- see config.ASPECT_STEM_MAP).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pipeline.config import ASPECT_STEM_MAP


class AspectPairDataset(Dataset):
    def __init__(self, anchor_parquet: Path, positive_parquet: Path, aspect: str):
        self.aspect = aspect
        stems = ASPECT_STEM_MAP[aspect]

        anchors = pd.read_parquet(anchor_parquet)
        positives = pd.read_parquet(positive_parquet)

        anchors = anchors[anchors["stem"].isin(stems)]
        positives = positives[positives["stem"].isin(stems)]

        # average across the aspect's stems per track, so each track_id -> 1 row
        self.anchor_by_track = self._pool_by_track(anchors)
        self.positive_by_track = self._pool_by_track(positives)

        self.track_ids = sorted(set(self.anchor_by_track) & set(self.positive_by_track))
        if not self.track_ids:
            raise ValueError(
                f"No overlapping track_ids between anchor/positive parquet for aspect={aspect}"
            )

    @staticmethod
    def _pool_by_track(df: pd.DataFrame) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for track_id, group in df.groupby("track_id"):
            vecs = np.stack(group["embedding"].to_numpy())
            out[str(track_id)] = vecs.mean(axis=0)
        return out

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, idx: int):
        track_id = self.track_ids[idx]
        anchor = torch.tensor(self.anchor_by_track[track_id], dtype=torch.float32)
        positive = torch.tensor(self.positive_by_track[track_id], dtype=torch.float32)
        return anchor, positive
