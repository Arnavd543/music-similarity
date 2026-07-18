# Controllable Multi-Axis Music Search

Cross-modal, controllable music similarity search: stem-separated audio
(Demucs) → frozen MERT embeddings → contrastively-trained aspect
projection heads (rhythm / melody / timbre / vocal) + a lyric-semantic
axis → weighted multi-vector ANN search (Qdrant) with per-user
personalization. Shares its pipeline with a benchmark/dataset paper
(factor-controlled triplets from real multitrack stems, human-annotated).

See `music-similarity-mle-plan.md` in this repo's parent folder for the
full 16-week phase plan, budget, and risk analysis this scaffold implements.

## Results (v1 — July 2026 Colab run)

**Pipeline scale:** 4,870 MTG-Jamendo tracks (30s excerpts) through Demucs
htdemucs + frozen MERT-v1-95M at 4.5s/track on a T4; 1,469 pair-tracks with
cross-section positives at 7.45s/track; four aspect heads trained
contrastively (InfoNCE, in-batch negatives). End-to-end weighted 5-axis
query latency: **28ms** against embedded Qdrant.

**Tag-proxy retrieval (precision@10, shared-tag relevance, 300 queries;
chance floor 0.186):**

| variant | all | genre | instrument | mood |
|---|---|---|---|---|
| raw_drums | 0.341 | 0.266 | 0.509 | 0.286 |
| raw_bass | 0.309 | 0.250 | 0.465 | 0.260 |
| raw_other | 0.351 | 0.309 | 0.518 | 0.312 |
| raw_vocals | 0.288 | 0.239 | 0.502 | 0.281 |
| raw_pooled | 0.375 | 0.326 | 0.526 | 0.317 |
| head_rhythm | 0.324 | 0.281 | 0.504 | 0.292 |
| head_melody | 0.349 | 0.275 | 0.510 | 0.292 |
| head_timbre | 0.368 | 0.309 | 0.504 | 0.321 |
| head_vocal | **0.329** | 0.250 | 0.479 | **0.338** |

Notes: tag matching is a *global*-similarity proxy, so `raw_pooled` leading
overall is expected — heads trade a little global-tag precision for axis
control. `head_vocal` beats `raw_vocals` outright and posts the best mood
score in the table. `p@10_instr` is ~0.5 for everything (instrument tags
are too common to discriminate).

**Disentanglement (inter-aspect similarity correlation over 1,000 random
pairs — the collapse check; >0.9 would mean the aspects merged):**

| | rhythm | melody | timbre | vocal |
|---|---|---|---|---|
| rhythm | 1.00 | 0.13 | 0.31 | 0.08 |
| melody | 0.13 | 1.00 | 0.66 | 0.18 |
| timbre | 0.31 | 0.66 | 1.00 | 0.32 |
| vocal | 0.08 | 0.18 | 0.32 | 1.00 |

The aspects are well separated; melody–timbre (0.66) is the closest pair,
consistent with timbre pooling all stems including the melody carriers —
reducing it (e.g. excluding pitched stems from the timbre pool) is an open
ablation.

## Repo layout

```
pipeline/     ingestion: download, Demucs stems, MERT embeddings, lyrics, Qdrant indexing
training/     aspect heads, contrastive losses, triplet engine, personalization
api/          FastAPI backend (search, upload, feedback, metrics)
web/          Streamlit frontend (sliders + audio players)
eval/         triplet accuracy, precision@k, annotator agreement
tests/        unit tests for logic that doesn't require GPU/model downloads
notebooks/    music_similarity_colab.ipynb -- run the pipeline on Google Colab
```

## Running on Google Colab

If you don't have cluster access, `notebooks/music_similarity_colab.ipynb`
runs the same pipeline on a Colab GPU runtime. It exists because Colab's
constraints are different enough from the Cornell-cluster assumptions
elsewhere in this repo that they change the pipeline's shape, not just where
it runs:

- **Ephemeral, small disk.** GPU runtimes get ~64GB of local disk, wiped on
  every runtime recycle. Storing raw stems for thousands of tracks would blow
  past that. `pipeline/colab_pipeline.py` streams instead: separate → embed →
  delete, per track, immediately -- only the small embedding vectors persist.
- **Nothing survives a disconnect except Google Drive.** `setup_colab()`
  mounts Drive and repoints `pipeline.config.PATHS`/`QDRANT` at a folder
  under it, so embeddings, checkpoints, and the local Qdrant index all live
  there. The streaming ingest also appends to its manifest after every
  single track (not batched at the end), so a mid-run disconnect loses at
  most one track's work, and re-running the same cell resumes automatically.
- **Metered, GPU-tiered compute.** `estimate_hours()` / `estimate_feasible_tracks()`
  in `colab_pipeline.py` convert a compute-unit budget into a rough subset
  size (rates as of mid-2026: ~1.76 CU/hr on T4, ~15 CU/hr on A100, L4 in
  between -- these shift over time, so treat the estimate as a starting
  point and check actual per-track timing on your first small batch).
- **No persistent server.** Qdrant runs in embedded local mode
  (`QdrantClient(path=...)`, no server process) instead of the
  Cloud/self-hosted setup `pipeline/qdrant_index.py` uses for the deployed
  app. Same collection schema, same query code in `api/search.py`.

Install `requirements-colab.txt` inside the notebook (it deliberately skips
torch/torchaudio -- Colab already ships a CUDA-matched build, and reinstalling
risks breaking it). Everything else (the `pipeline/`, `training/`, `eval/`,
`api/` modules) is imported unmodified from the same codebase used for the
cluster run; `colab_pipeline.py` is additive, not a fork.

For a 100-compute-unit budget on a T4, a 5,000-8,000 track ingest subset
leaves comfortable headroom for Phase 2 head training and retries. The
notebook's budget-estimator cell prints a recommendation for whatever GPU you
actually get.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

GPU-heavy deps (torch, demucs, transformers, faster-whisper) are declared
in `pyproject.toml` but are large downloads — install them once you're on
a machine with the bandwidth/GPU to use them (laptop for smoke-testing on
a handful of tracks, Cornell cluster for the full run).

## Quickstart (local / cluster)

Small-scale validation on ~100 tracks:

```bash
python -m pipeline.download_jamendo --n-tracks 100
python -m pipeline.demucs_stems --audio-dir data/audio --out data/stems
python -m pipeline.mert_embed --stems-dir data/stems
```

Whole-mix FAISS baseline (no stems, no heads):

```bash
python -m pipeline.baseline_faiss build --audio-dir data/audio
python -m pipeline.baseline_faiss query --track-id <id> --k 10
```

## Full pipeline (once data + compute are available)

```bash
# 1. ingest
python -m pipeline.download_jamendo --n-tracks 25000
python -m pipeline.demucs_stems --audio-dir data/audio --out data/stems --num-shards 20 --shard-index $SLURM_ARRAY_TASK_ID
python -m pipeline.mert_embed --stems-dir data/stems
python -m pipeline.lyrics

# 2. train aspect heads (needs anchor/positive embedding parquets from an
#    offline augmentation pass using training/augmentations.py)
python -m training.train --aspect rhythm --anchor data/embeddings/anchor.parquet --positive data/embeddings/positive.parquet

# 3. eval
python -m eval.triplet_accuracy --triplets data/triplets/pool.jsonl --embeddings data/embeddings/aspect_vectors.parquet
python -m eval.precision_at_k --metadata data/metadata/jamendo.tsv --embeddings data/embeddings/aspect_vectors.parquet

# 4. index + serve
python -m pipeline.qdrant_index
uvicorn api.main:app --reload
streamlit run web/app.py
```

## Paper track

```bash
python -m training.triplet_engine --corpus-dir data/moisesdb --n-per-factor 2000
python -m eval.annotation_agreement --ratings data/annotations/ratings.csv
```

## Tests

```bash
pytest
```

Tests cover pure-logic modules (losses, personalization fitting, triplet
engine, precision@k math, agreement stats) that don't require downloading
model weights or audio.
