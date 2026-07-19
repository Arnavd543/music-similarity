"""FastAPI backend.

Endpoints:
  POST /search           weighted multi-aspect similarity search over an indexed seed
  POST /upload            async job: run Demucs+MERT+heads on an uploaded file, then index it
  GET  /upload/{job_id}   poll job status
  POST /feedback          record a pairwise ("which is more similar") judgment
  GET  /users/{id}/weights  personalized weights fit from that user's feedback so far
  GET  /metrics            p50/p95 latency + request count

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
    TrackMatch,
    UploadJobStatus,
)
from api.search import search_similar
from pipeline.config import PATHS
from training.personalization import PairwiseJudgment, fit_user_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Controllable Multi-Axis Music Search", version="0.1.0")

# --- in-memory state (job statuses and metrics are fine to lose on restart;
# feedback judgments are durable in Qdrant via api/feedback_store.py) ---
_upload_jobs: dict[str, UploadJobStatus] = {}
_latencies_ms: deque[float] = deque(maxlen=2000)
_embedding_cache: dict[str, dict] = {}  # seed_track_id -> vectors, avoids re-hitting Qdrant


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


_name_index: list[dict] | None = None


def _load_name_index() -> list[dict]:
    """All (track_id, title, artist) from the collection's payloads, cached
    per process. ~5k entries, so in-memory substring search is plenty."""
    global _name_index
    if _name_index is None:
        from pipeline.config import QDRANT
        from pipeline.qdrant_index import get_client

        client = get_client()
        entries, offset = [], None
        while True:
            points, offset = client.scroll(
                collection_name=QDRANT.collection, limit=1000,
                offset=offset, with_payload=True, with_vectors=False,
            )
            for p in points:
                if p.payload.get("title"):
                    entries.append({
                        "track_id": p.payload["track_id"],
                        "title": p.payload["title"],
                        "artist": p.payload.get("artist", ""),
                    })
            if offset is None:
                break
        _name_index = entries
        log.info("Name index loaded: %d named tracks", len(entries))
    return _name_index


@app.get("/tracks", response_model=list[TrackMatch])
def find_tracks(q: str, limit: int = 20) -> list[TrackMatch]:
    """Substring search over 'artist title'. Requires payloads to carry
    names -- run `python -m pipeline.track_names` once to stamp them."""
    needle = q.strip().lower()
    if len(needle) < 2:
        return []
    matches = [
        e for e in _load_name_index()
        if needle in f"{e['artist']} {e['title']}".lower()
    ]
    return [TrackMatch(**m) for m in matches[:limit]]


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    weights = req.weights.as_dict()
    if req.user_id:
        from api.feedback_store import load_user_judgments
        from pipeline.qdrant_index import get_client

        judgments = load_user_judgments(get_client(), req.user_id)
        if judgments:
            weights = fit_user_weights(judgments)
            log.info("Personalized weights for user %s (%d judgments): %s",
                     req.user_id, len(judgments), weights)

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


_modal_calls: dict[str, object] = {}


def _modal_ingest_fn():
    import modal

    lookup = getattr(modal.Function, "from_name", None) or modal.Function.lookup
    return lookup("music-similarity-upload", "ingest_upload")


@app.post("/upload", response_model=UploadJobStatus)
async def upload(file: UploadFile, background_tasks: BackgroundTasks) -> UploadJobStatus:
    import os

    job_id = str(uuid.uuid4())
    contents = await file.read()
    status = UploadJobStatus(job_id=job_id, status="queued")
    _upload_jobs[job_id] = status

    if os.getenv("USE_MODAL_UPLOAD"):
        # serverless GPU path (deploy/modal_upload.py); ~30s of T4 per track
        try:
            call = _modal_ingest_fn().spawn(contents, job_id)
            _modal_calls[job_id] = call
            status.status = "processing"
        except Exception as exc:  # noqa: BLE001
            log.exception("Modal dispatch failed")
            status.status = "failed"
            status.error = f"modal dispatch: {exc}"
        return status

    # local CPU fallback: works without cloud credentials, but slow (~minutes)
    suffix = Path(file.filename or "upload.mp3").suffix or ".mp3"
    dest = PATHS.audio_dir / f"upload_{job_id}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(contents)
    background_tasks.add_task(_process_upload, job_id, str(dest))
    return status


@app.get("/upload/{job_id}", response_model=UploadJobStatus)
def upload_status(job_id: str) -> UploadJobStatus:
    status = _upload_jobs.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    call = _modal_calls.get(job_id)
    if call is not None and status.status == "processing":
        try:
            result = call.get(timeout=0)  # raises TimeoutError while running
            status.status = result.get("status", "done")
            status.track_id = result.get("track_id", job_id)
            _modal_calls.pop(job_id, None)
        except TimeoutError:
            pass
        except Exception as exc:  # noqa: BLE001
            status.status = "failed"
            status.error = str(exc)
            _modal_calls.pop(job_id, None)
    return status


@app.post("/feedback")
def feedback(fb: PairwiseFeedback) -> dict:
    """Compute per-aspect cosine sims server-side from the stored vectors
    and append a full PairwiseJudgment for this user."""
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
    from api.feedback_store import load_user_judgments, save_judgment

    save_judgment(client, fb.user_id, fb.seed_track_id, judgment)
    n = len(load_user_judgments(client, fb.user_id))
    log.info("Recorded feedback for user=%s seed=%s (%d total)", fb.user_id, fb.seed_track_id, n)
    return {"status": "recorded", "n_judgments": n}


@app.get("/users/{user_id}/weights", response_model=SliderWeights)
def user_weights(user_id: str) -> SliderWeights:
    from api.feedback_store import load_user_judgments
    from pipeline.qdrant_index import get_client

    judgments = load_user_judgments(get_client(), user_id)
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
