"""FastAPI backend.

Endpoints:
  POST /search           weighted multi-aspect similarity search over an indexed seed
  POST /upload            async job: run Demucs+MERT+heads on an uploaded file, then index it
  GET  /upload/{job_id}   poll job status
  POST /feedback          record a pairwise ("which is more similar") judgment
  GET  /users/{id}/weights  personalized weights fit from that user's feedback so far
  GET  /metrics            p50/p95 latency + cache hit rate (ops polish per plan Phase 3)

The upload path enqueues to a background worker rather than blocking the
request; in production this hands off to Modal/Replicate serverless GPU
(~$0.001-0.01/query per the budget plan) -- here it's a local asyncio
background task so the API is runnable without cloud credentials.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile

from api.schemas import (
    PairwiseFeedback,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SliderWeights,
    UploadJobStatus,
)
from api.search import search_similar
from pipeline.config import PATHS
from training.personalization import PairwiseJudgment, fit_user_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Controllable Multi-Axis Music Search", version="0.1.0")

# --- in-memory state (swap for Redis/Postgres before real deployment) ---
_upload_jobs: dict[str, UploadJobStatus] = {}
_user_feedback: dict[str, list[PairwiseJudgment]] = {}
_latencies_ms: deque[float] = deque(maxlen=2000)
_embedding_cache: dict[str, dict] = {}  # seed_track_id -> vectors, avoids re-hitting Qdrant


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    weights = req.weights.as_dict()
    if req.user_id and req.user_id in _user_feedback:
        weights = fit_user_weights(_user_feedback[req.user_id])
        log.info("Using personalized weights for user %s: %s", req.user_id, weights)

    start = time.perf_counter()
    try:
        results, latency_ms = search_similar(req.seed_track_id, weights, top_k=req.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    total_ms = (time.perf_counter() - start) * 1000
    _latencies_ms.append(total_ms)

    return SearchResponse(
        seed_track_id=req.seed_track_id,
        weights_used=weights,
        results=[SearchResultItem(**r) for r in results],
        latency_ms=round(total_ms, 2),
    )


def _process_upload(job_id: str, audio_path: str) -> None:
    """Runs the full ingest pipeline (Demucs -> MERT -> aspect heads -> Qdrant
    upsert) on one uploaded track. Heavy imports are deferred into this
    function so `uvicorn api.main:app` starts fast even without GPU deps
    installed for a pure-search deployment."""
    from pipeline.demucs_stems import separate_track
    from pipeline.mert_embed import MertEmbedder
    from pipeline.qdrant_index import get_client, upsert_tracks
    from training.aspect_heads import MultiAspectModel

    try:
        _upload_jobs[job_id].status = "processing"
        track_id = job_id
        stems_out = PATHS.stems_dir / track_id
        stems_out.mkdir(parents=True, exist_ok=True)

        from demucs.pretrained import get_model

        demucs_model = get_model("htdemucs")
        separate_track(demucs_model, Path(audio_path), PATHS.stems_dir, device="cpu")

        embedder = MertEmbedder()
        model = MultiAspectModel()
        # load trained head checkpoints if present
        for aspect, head in model.heads.items():
            ckpt = PATHS.checkpoints_dir / f"{aspect}_head.pt"
            if ckpt.exists():
                import torch

                head.load_state_dict(torch.load(ckpt, map_location="cpu"))

        import torch
        import torchaudio

        stem_embeddings = {}
        for stem_path in stems_out.glob("*.wav"):
            wav, sr = torchaudio.load(str(stem_path))
            stem_embeddings[stem_path.stem] = torch.tensor(embedder.embed(wav, sr)).unsqueeze(0)

        aspect_vectors = model.project_all(stem_embeddings)

        import pandas as pd

        row = {"track_id": track_id}
        for aspect, vec in aspect_vectors.items():
            row[aspect] = vec.squeeze(0).detach().numpy().tolist()
        df = pd.DataFrame([row])

        client = get_client()
        upsert_tracks(client, df)

        _upload_jobs[job_id].status = "done"
        _upload_jobs[job_id].track_id = track_id
    except Exception as exc:  # noqa: BLE001
        log.exception("Upload job %s failed", job_id)
        _upload_jobs[job_id].status = "failed"
        _upload_jobs[job_id].error = str(exc)


@app.post("/upload", response_model=UploadJobStatus)
async def upload(file: UploadFile, background_tasks: BackgroundTasks) -> UploadJobStatus:
    job_id = str(uuid.uuid4())
    suffix = Path(file.filename or "upload.mp3").suffix or ".mp3"
    dest = PATHS.audio_dir / f"upload_{job_id}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    contents = await file.read()
    dest.write_bytes(contents)

    status = UploadJobStatus(job_id=job_id, status="queued")
    _upload_jobs[job_id] = status
    background_tasks.add_task(_process_upload, job_id, str(dest))
    return status


@app.get("/upload/{job_id}", response_model=UploadJobStatus)
def upload_status(job_id: str) -> UploadJobStatus:
    status = _upload_jobs.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return status


@app.post("/feedback")
def feedback(fb: PairwiseFeedback) -> dict:
    """Compute per-aspect cosine sims server-side from the stored vectors and
    append a full PairwiseJudgment. (An earlier version only setdefault'd the
    list and never appended -- personalization silently received zero data.)"""
    import numpy as np

    from api.search import fetch_seed_vectors
    from pipeline.qdrant_index import get_client

    client = get_client()
    try:
        seed_v = fetch_seed_vectors(client, fb.seed_track_id)
        a_v = fetch_seed_vectors(client, fb.candidate_a)
        b_v = fetch_seed_vectors(client, fb.candidate_b)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def per_aspect_sims(cand: dict) -> dict[str, float]:
        out = {}
        for aspect, sv in seed_v.items():
            cv = cand.get(aspect)
            if cv is None:
                continue
            s, c = np.asarray(sv), np.asarray(cv)
            denom = (np.linalg.norm(s) * np.linalg.norm(c)) + 1e-8
            out[aspect] = float(s @ c / denom)
        return out

    judgment = PairwiseJudgment(
        seed_sims_a=per_aspect_sims(a_v),
        seed_sims_b=per_aspect_sims(b_v),
        chose_a=fb.chose_a,
    )
    _user_feedback.setdefault(fb.user_id, []).append(judgment)
    n = len(_user_feedback[fb.user_id])
    log.info("Recorded feedback for user=%s seed=%s (%d total)", fb.user_id, fb.seed_track_id, n)
    return {"status": "recorded", "n_judgments": n}


@app.get("/users/{user_id}/weights", response_model=SliderWeights)
def user_weights(user_id: str) -> SliderWeights:
    judgments = _user_feedback.get(user_id, [])
    weights = fit_user_weights(judgments)
    return SliderWeights(**weights)


@app.get("/metrics")
def metrics() -> dict:
    if not _latencies_ms:
        return {"p50_ms": None, "p95_ms": None, "n_requests": 0}
    sorted_lat = sorted(_latencies_ms)
    n = len(sorted_lat)
    p50 = sorted_lat[int(n * 0.50)]
    p95 = sorted_lat[min(n - 1, int(n * 0.95))]
    return {"p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "n_requests": n}
