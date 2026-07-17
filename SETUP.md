# Setup checklist — from "code in Drive" to training, app, and paper data

Ordered by when you'll hit the wall if you skip it. Do Part A today; B before Phase 3; C in parallel (it has external wait times).

---

## Part A — Before any Colab training

### A1. Accounts & keys (training needs almost none)
| Service | Needed? | What to do |
|---|---|---|
| Google Colab | **Yes** | Buy compute units: colab.research.google.com → Pay As You Go (100 CU ≈ $10) or Colab Pro. Free tier works for smoke tests only. |
| Hugging Face | No key required | MERT, GTE, Whisper models are all public. Optional: create an account + `huggingface-cli login` to avoid rate limits on repeated model downloads. |
| LRCLib (lyrics) | **No key** | Free, unauthenticated API. Nothing to set up. |
| MTG-Jamendo | **No key** | Public CDN download. Nothing to sign up for. |

No secrets go in the notebook. The only "key-like" thing in the whole training path is nothing — by design.

### A2. Drive layout
Your Drive should look like:
```
MyDrive/
  music-similarity/          <- this repo (already uploaded)
    pipeline/  training/  eval/  api/  web/  notebooks/  tests/
    colab_run/               <- created automatically by the notebook; all
                                embeddings/checkpoints/manifests land here
```
The notebook's `PROJECT_DIR` (cell 3) must match this path exactly — `MyDrive`, not `My Drive`.

Check Drive quota first (drive.google.com/settings): the 5k-track ingest downloads ~5GB of mp3s. Free tier is 15GB total; clear space or reduce `N_TRACKS`.

### A3. Also put the code on GitHub (10 min, do it now)
Drive is storage, not version control — one bad notebook cell can overwrite a file with no undo. Create a private GitHub repo and push the code. Then in Colab you can `!git clone` instead of importing from Drive (faster imports too; Drive-mounted Python imports are slow). Keep Drive for *data* (embeddings, checkpoints), GitHub for *code*. This also becomes the public repo for the paper later.

### A4. Runtime + smoke test (before spending any real CU)
1. Runtime → Change runtime type → **T4 GPU**.
2. Run notebook cells 1–5 (install, mount, paths). First run of `pip install -r requirements-colab.txt` takes ~5 min.
3. **Verify one Jamendo URL** — run the Phase-0 cell with `n=5` tracks first. The downloader now fails fast if the first 10 downloads all fail; if it does, open one URL from the error in a browser and fix `RAW_30S_BASE`/PATH handling before bulk-downloading.
4. **3-track pair-ingest smoke test** before `N_PAIR_TRACKS=1500`:
   ```python
   run_streaming_pair_ingest(pair_audio_paths[:3], anchor_manifest, positive_manifest)
   ```
   Then assert: both manifests have 3 lines, and for a drums row, anchor != positive embedding:
   ```python
   import json
   a = json.loads(open(anchor_manifest).readline()); p = json.loads(open(positive_manifest).readline())
   assert a["embeddings"]["drums"] != p["embeddings"]["drums"], "rhythm positive is identical to anchor"
   ```
5. Time those 3 tracks; multiply out to check your CU budget (cell 5's estimator) before committing to the full N.

### A5. Order of the big runs
Phase 0 baseline (200 tracks, ~30 min) → Phase 1 ingest (5k tracks, ~8–12 T4-hours, resumable so it's fine to split across sessions) → Phase 2 pair ingest (1.5k tracks) → head training (minutes–hours) → eval cells. Never start a run you can't leave: everything checkpoints to Drive, so a disconnect only costs the current track.

---

## Part B — Before the app (Phase 3)

| Service | What to set up |
|---|---|
| **Qdrant Cloud** | Create account at cloud.qdrant.io → free 1GB cluster → copy cluster URL + API key. Set `QDRANT_URL` and `QDRANT_API_KEY` env vars where the API runs. (Colab uses embedded local mode — no account needed until you deploy.) |
| **Modal** | Sign up at modal.com (free $30/mo credits) → `pip install modal` + `modal token new` on your laptop. Used for the upload-a-song GPU inference path. |
| **ngrok** (optional) | Only if you want to preview the FastAPI/Streamlit app from inside Colab: free account at ngrok.com → copy authtoken → `ngrok config add-authtoken ...`. Skip if you demo locally. |
| **GitHub Actions / Docker Hub** (optional) | For CI + deploy polish later; nothing needed yet. |

Keep all keys in env vars / Colab Secrets (key icon in left sidebar), never in notebook cells or committed files.

---

## Part C — Paper data (start now; these have wait times)

1. **Email Music4All** (contact4music4all@gmail.com) from your Cornell address — academic requests are usually granted but can take days/weeks. Enrichment only; nothing blocks on it.
2. **MoisesDB** — request via the Moises/music.ai research page (form + license agreement). Primary corpus for the triplet engine.
3. **MUSDB18-HQ** — Zenodo (zenodo.org, search "MUSDB18-HQ"): free account + access request form, granted quickly. ~30GB, download to Drive or laptop.
4. **Slakh2100** — free direct download (slakh.com), no request needed. ~100GB full, but the "baby Slakh" subset is fine to start.
5. **Cornell IRB** — email the IRB office (irb.cornell.edu) describing the annotation study (anonymous music-similarity ratings, no sensitive data) and ask whether it qualifies as exempt. Do this now: it's the only external dependency on the paper's critical path. If you can get a faculty advisor (music + CS/IS), ask them first — IRB usually needs a faculty PI.
6. **Prolific** (only if friends aren't enough annotators) — account + ~$100 budget, needed months from now.

Multitrack corpora don't fit Colab's ephemeral disk comfortably — for the paper track, plan to run the triplet engine on the Cornell cluster or your laptop, not Colab.

---

## TL;DR sequence
Today: buy CU → push code to GitHub → check Drive quota → send Music4All + IRB + MoisesDB/MUSDB requests.
First Colab session: cells 1–5 → 5-track download check → 3-track pair smoke test → then let Phase 1 run.
Before Phase 3: Qdrant Cloud + Modal accounts.
