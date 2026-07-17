import torch

from training.losses import circle_loss, info_nce


def _normalized(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


def test_info_nce_lower_for_aligned_pairs():
    torch.manual_seed(0)
    anchor = _normalized(torch.randn(16, 32))
    aligned_positive = anchor.clone()  # perfect positives
    random_positive = _normalized(torch.randn(16, 32))  # unrelated

    aligned_loss = info_nce(anchor, aligned_positive)
    random_loss = info_nce(anchor, random_positive)

    assert aligned_loss.item() < random_loss.item()


def test_info_nce_is_finite_and_scalar():
    torch.manual_seed(1)
    anchor = _normalized(torch.randn(8, 16))
    positive = _normalized(torch.randn(8, 16))
    loss = info_nce(anchor, positive)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_circle_loss_lower_for_aligned_pairs():
    torch.manual_seed(2)
    anchor = _normalized(torch.randn(16, 32))
    aligned_positive = anchor.clone()
    random_positive = _normalized(torch.randn(16, 32))

    aligned_loss = circle_loss(anchor, aligned_positive)
    random_loss = circle_loss(anchor, random_positive)

    assert aligned_loss.item() < random_loss.item()
