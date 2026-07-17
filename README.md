# Controllable Multi-Axis Music Search

Cross-modal, controllable music similarity search: stem-separated audio
(Demucs) → frozen MERT embeddings → contrastively-trained aspect
projection heads (rhythm / melody / timbre / vocal) + a lyric-semantic
axis → weighted multi-vector ANN search (Qdrant) with per-user
personalization. Shares its pipeline with a benchmark/dataset paper
(factor-controlled triplets from real multitrack stems, human-annotated).

See `music-similarity-mle-plan.md` in this repo's parent folder for the
full 16-week phase plan, budget, and risk analysis this scaffold implements.

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

## Week-1 checklist mapping

1. **Data**: `python -m pipeline.download_jamendo --n-tracks 5000` (MTG-Jamendo
   subset, tag-stratified). Music4All access and MoisesDB/MUSDB18-HQ downloads
   are external steps (email/manual download) — not scriptable here.
2. **Repo scaffold**: this repo.
3. **Validate pipeline shape/timing** on ~100 tracks:
   ```bash
   python -m pipeline.demucs_stems --audio-dir data/audio --out data/stems
   python -m pipeline.mert_embed --stems-dir data/stems
   ```
4. **Baseline walking skeleton** (whole-mix embedding + FAISS, no stems/heads):
   ```bash
   python -m pipeline.baseline_faiss build --audio-dir data/audio
   python -m pipeline.baseline_faiss query --track-id <id> --k 10
   ```
5. Read MERIT (arXiv 2605.27346) end-to-end — see plan §1 for the delta
   this project takes (stems at inference, lyrics axis, real-data invariance
   triplets, deployment, personalization).

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
