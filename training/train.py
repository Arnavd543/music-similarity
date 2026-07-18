"""Train one AspectHead per aspect via contrastive learning over frozen
MERT features. Heads are tiny (~1M params each) so this is hours on one
cluster GPU.

Usage:
    python -m training.train --aspect rhythm \
        --anchor data/embeddings/mert_anchor.parquet \
        --positive data/embeddings/mert_positive.parquet \
        --loss info_nce
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pipeline.config import PATHS, TRAIN
from training.aspect_heads import AspectHead
from training.dataset import AspectPairDataset
from training.losses import circle_loss, info_nce

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LOSS_FNS = {"info_nce": info_nce, "circle": circle_loss}


def train_one_aspect(
    aspect: str,
    anchor_path: Path,
    positive_path: Path,
    loss_name: str = "info_nce",
    epochs: int = TRAIN.epochs,
    batch_size: int = TRAIN.batch_size,
    lr: float = TRAIN.lr,
    device: str | None = None,
    checkpoint_dir: Path = PATHS.checkpoints_dir,
) -> AspectHead:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(TRAIN.seed)

    dataset = AspectPairDataset(anchor_path, positive_path, aspect)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    log.info("Aspect=%s: %d training pairs, %d batches/epoch", aspect, len(dataset), len(loader))

    head = AspectHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    loss_fn = LOSS_FNS[loss_name]

    for epoch in range(epochs):
        head.train()
        epoch_loss = 0.0
        for anchor, positive in loader:
            anchor, positive = anchor.to(device), positive.to(device)
            z_anchor = head(anchor)
            z_positive = head(positive)

            if loss_name == "info_nce":
                loss = loss_fn(z_anchor, z_positive, temperature=TRAIN.temperature)
            else:
                loss = loss_fn(z_anchor, z_positive)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / max(1, len(loader))
        log.info("[%s] epoch %d/%d loss=%.4f", aspect, epoch + 1, epochs, avg)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{aspect}_head.pt"
    torch.save(head.state_dict(), ckpt_path)
    log.info("Saved %s", ckpt_path)
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aspect", required=True, choices=["rhythm", "melody", "timbre", "vocal"])
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--loss", default="info_nce", choices=list(LOSS_FNS))
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    parser.add_argument("--lr", type=float, default=TRAIN.lr)
    args = parser.parse_args()

    train_one_aspect(
        aspect=args.aspect,
        anchor_path=args.anchor,
        positive_path=args.positive,
        loss_name=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
