"""Modal serverless GPU worker for the upload path.

Runs the full single-track ingest (Demucs -> MERT -> aspect heads -> Qdrant
Cloud upsert) on a T4, billed per second (~half a cent per upload).

One-time setup (on your laptop):
    pip install modal
    modal token new
    modal secret create qdrant-cloud QDRANT_URL=https://<cluster>.cloud.qdrant.io QDRANT_API_KEY=<key>
    # upload trained head checkpoints to a persisted volume:
    modal volume create music-similarity-checkpoints
    modal volume put music-similarity-checkpoints <local>/rhythm_head.pt /rhythm_head.pt
    ... (repeat for melody/timbre/vocal)
Deploy:
    modal deploy deploy/modal_upload.py
Test:
    modal run deploy/modal_upload.py --audio-path some_local.mp3

api/main.py dispatches here when USE_MODAL_UPLOAD=1 is set on the API host
(requires `pip install modal` + MODAL_TOKEN_ID/MODAL_TOKEN_SECRET there).
"""

from __future__ import annotations

import modal

app = modal.App("music-similarity-upload")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "torch", "torchaudio", "torchcodec", "demucs>=4.0.1", "transformers>=4.40",
        "qdrant-client>=1.9", "numpy", "pandas",
    )
    # the pipeline + training modules ride along as local code
    .add_local_python_source("pipeline", "training")
)

checkpoints = modal.Volume.from_name("music-similarity-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    secrets=[modal.Secret.from_name("qdrant-cloud")],
    volumes={"/checkpoints": checkpoints},
)
def ingest_upload(audio_bytes: bytes, track_id: str) -> dict:
    """Returns {track_id, status, aspects_indexed}."""
    import tempfile
    from pathlib import Path

    import torch
    from demucs.pretrained import get_model

    from pipeline.colab_pipeline import process_one_track
    from pipeline.mert_embed import MertEmbedder
    from pipeline.qdrant_index import ensure_collection, get_client, upsert_tracks
    from training.aspect_heads import MultiAspectModel

    scratch = Path(tempfile.mkdtemp())
    audio_path = scratch / f"{track_id}.mp3"
    audio_path.write_bytes(audio_bytes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    demucs_model = get_model("htdemucs")
    demucs_model.to(device)
    demucs_model.eval()
    embedder = MertEmbedder(device=device)

    row = process_one_track(audio_path, demucs_model, embedder, scratch, device)

    model = MultiAspectModel()
    for aspect, head in model.heads.items():
        ckpt = Path("/checkpoints") / f"{aspect}_head.pt"
        if ckpt.exists():
            head.load_state_dict(torch.load(ckpt, map_location="cpu"))
            head.eval()

    stem_embeddings = {
        stem: torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        for stem, vec in row["embeddings"].items()
    }
    with torch.no_grad():
        projected = model.project_all(stem_embeddings)

    import pandas as pd

    out = {"track_id": track_id}
    for aspect, vec in projected.items():
        out[aspect] = vec.squeeze(0).numpy().tolist()

    client = get_client()  # reads QDRANT_URL/API_KEY from the modal secret
    ensure_collection(client)
    upsert_tracks(client, pd.DataFrame([out]))

    return {"track_id": track_id, "status": "done", "aspects_indexed": sorted(projected)}


@app.local_entrypoint()
def main(audio_path: str):
    """Local test: modal run deploy/modal_upload.py --audio-path song.mp3"""
    from pathlib import Path

    result = ingest_upload.remote(Path(audio_path).read_bytes(), Path(audio_path).stem)
    print(result)
