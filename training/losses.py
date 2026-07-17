"""Contrastive objectives for aspect-head training.

InfoNCE is the default (simple, well understood); Circle loss is offered
as the ablation alternative referenced in the plan ("Circle vs. InfoNCE
saturation behavior") since Circle loss avoids InfoNCE's tendency to
saturate once easy negatives dominate a batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """anchor, positive: (B, D), L2-normalized. In-batch negatives (all other
    rows). Standard symmetric InfoNCE."""
    logits = anchor @ positive.T / temperature  # (B, B)
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2


def circle_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    margin: float = 0.25,
    gamma: float = 64.0,
) -> torch.Tensor:
    """Circle loss (Sun et al. 2020) with in-batch negatives, as an
    alternative to InfoNCE that re-weights each similarity pair by how far
    it is from optimal, avoiding gradient saturation on easy negatives."""
    sim = anchor @ positive.T  # (B, B), cosine sims since inputs are normalized
    B = anchor.shape[0]
    pos_mask = torch.eye(B, device=anchor.device, dtype=torch.bool)
    neg_mask = ~pos_mask

    op = 1 + margin
    on = -margin
    delta_p = 1 - margin
    delta_n = margin

    alpha_p = torch.clamp(op - sim, min=0.0)
    alpha_n = torch.clamp(sim - on, min=0.0)

    logit_p = -gamma * alpha_p * (sim - delta_p)
    logit_n = gamma * alpha_n * (sim - delta_n)

    logit_p = logit_p.masked_fill(neg_mask, float("-inf"))
    logit_n = logit_n.masked_fill(pos_mask, float("-inf"))

    loss = F.softplus(
        torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)
    ).mean()
    return loss
