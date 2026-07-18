# Deployment guide — from Colab index to public demo

Order: Qdrant Cloud first (data), then the API, then the frontend, then the
upload path. Each step is independently testable.

## 1. Qdrant Cloud (~30 min)

1. cloud.qdrant.io → create free 1GB cluster → copy the cluster URL and API key.
2. Migrate the index by re-running the upsert against the cloud instead of the
   local embedded DB — same code, different env vars. In Colab (or locally with
   the parquets downloaded from Drive):
   ```python
   import os
   os.environ.pop("QDRANT_LOCAL_PATH", None)
   os.environ["QDRANT_URL"] = "https://<cluster>.cloud.qdrant.io"
   os.environ["QDRANT_API_KEY"] = "<key>"
   # then re-run: ensure_collection + upsert_tracks(client, aspect_df)
   ```
   Note: restart the Python session first so `pipeline.config.QDRANT` re-reads
   the env vars.
3. Sanity check: `client.count("tracks")` should report ~4,870.

## 2. API (FastAPI) — Render / Railway / Fly.io free tier (~1 hr)

The pure-search deployment needs **no GPU and no torch** — `api/main.py`
defers all heavy imports into the upload worker path.

1. Dockerfile (put in repo root):
   ```dockerfile
   FROM python:3.12-slim
   WORKDIR /app
   COPY pipeline/ pipeline/  api/ api/  training/ training/
   RUN pip install --no-cache-dir fastapi "uvicorn[standard]" qdrant-client numpy pandas scikit-learn pydantic
   ENV QDRANT_URL="" QDRANT_API_KEY=""
   CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
   (training/ is needed only for `personalization.py` + `aspect_heads` import
   chain; torch isn't imported unless /upload is hit.)
2. Set `QDRANT_URL` / `QDRANT_API_KEY` as secrets on the host.
3. Smoke test: `curl https://<api>/health`, then a `POST /search` with a known
   seed_track_id. Check `GET /metrics` after a few queries.

## 3. Frontend (Streamlit Community Cloud, free) (~30 min)

1. Push the repo to GitHub (share.streamlit.io deploys straight from a repo).
2. New app → `web/app.py` → set env var `MS_API_BASE=https://<your-api-host>`.
3. Audio players stream CC-licensed Jamendo audio via the `path` payload —
   no audio hosting on your side, and it's legal because the catalog is CC.

## 4. Upload path (Modal, free $30/mo credits) (~half a day)

The in-process CPU upload worker in `api/main.py` works but is slow
(~minutes/track on CPU). For the real thing, move `_process_upload`'s body
into a Modal function:

- `modal token new` locally; create `deploy/modal_upload.py` with a
  `@app.function(gpu="T4", image=<image with torch+demucs+transformers>)`
  that takes audio bytes, runs Demucs → MERT → heads (checkpoints baked into
  the image or pulled from a volume), and upserts to Qdrant Cloud.
- `api/main.py`'s `/upload` then calls `modal.Function.lookup(...).spawn(...)`
  instead of `background_tasks.add_task`, and `/upload/{job_id}` polls the
  Modal call.
- Cost: ~30s of T4 per upload ≈ half a cent; per-second billing, no idle cost.

## 5. Ops polish checklist (the resume bullets)

- [ ] `GET /metrics` shows p50/p95 — screenshot for the README
- [ ] Embedding cache hit-rate metric
- [ ] Dockerfile + GitHub Actions CI (pytest + pyflakes on push)
- [ ] Index versioning: name collections `tracks_v1`, `tracks_v2`, ... and
      switch via env var — lets you A/B old vs. new heads without downtime
- [ ] Uptime check (UptimeRobot free) against /health

## Secrets hygiene

Env vars only (`QDRANT_URL`, `QDRANT_API_KEY`, `MS_API_BASE`). Nothing in
git, nothing in notebook cells — Colab Secrets for the notebook side.
