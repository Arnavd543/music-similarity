"""Lightweight projection heads on top of frozen MERT features.

One 2-layer MLP (~1M params) per aspect: rhythm, melody, timbre, vocal.
Input is a pooled MERT embedding (768-d) of the relevant stem(s) per
config.ASPECT_STEM_MAP; when an aspect uses multiple stems (melody =
bass+other, timbre = all four), their pooled embeddings are averaged
before the head, so the head input dim is always MODEL.mert_embed_dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.config import ASPECT_STEM_MAP, ASPECTS, MODEL


class AspectHead(nn.Module):
    def __init__(
        self,
        in_dim: int = MODEL.mert_embed_dim,
        hidden_dim: int = MODEL.head_hidden_dim,
        out_dim: int = MODEL.head_output_dim,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)  # unit-norm so cosine sim = dot product


class MultiAspectModel(nn.Module):
    """Bundles one AspectHead per rhythm/melody/timbre/vocal aspect.
    (`lyric` is handled separately by pipeline/lyrics.py's sentence-transformer,
    not trained here -- it's a fixed off-the-shelf text embedding.)"""

    def __init__(self):
        super().__init__()
        trainable_aspects = [a for a in ASPECTS if a != "lyric"]
        self.heads = nn.ModuleDict({aspect: AspectHead() for aspect in trainable_aspects})

    def forward(self, stem_embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """stem_embeddings: {stem_name: (B, 768)} pooled MERT features.
        Returns {aspect: (B, out_dim)} projected, normalized vectors."""
        out = {}
        for aspect, head in self.heads.items():
            stems = ASPECT_STEM_MAP[aspect]
            available = [stem_embeddings[s] for s in stems if s in stem_embeddings]
            if not available:
                continue
            pooled = torch.stack(available, dim=0).mean(dim=0)
            out[aspect] = head(pooled)
        return out

    def project_all(self, stem_embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.forward(stem_embeddings)
