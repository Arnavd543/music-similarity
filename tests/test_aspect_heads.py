import torch

from pipeline.config import MODEL
from training.aspect_heads import AspectHead, MultiAspectModel


def test_aspect_head_output_is_unit_norm():
    head = AspectHead()
    x = torch.randn(4, MODEL.mert_embed_dim)
    z = head(x)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_multi_aspect_model_projects_available_stems():
    model = MultiAspectModel()
    stem_embeddings = {
        "drums": torch.randn(2, MODEL.mert_embed_dim),
        "bass": torch.randn(2, MODEL.mert_embed_dim),
        "other": torch.randn(2, MODEL.mert_embed_dim),
        "vocals": torch.randn(2, MODEL.mert_embed_dim),
    }
    out = model.project_all(stem_embeddings)
    assert set(out.keys()) == {"rhythm", "melody", "timbre", "vocal"}
    for vec in out.values():
        assert vec.shape == (2, MODEL.head_output_dim)


def test_multi_aspect_model_handles_missing_stems():
    model = MultiAspectModel()
    stem_embeddings = {"drums": torch.randn(1, MODEL.mert_embed_dim)}
    out = model.project_all(stem_embeddings)
    assert "rhythm" in out
    assert "melody" not in out  # needs bass/other, neither present
