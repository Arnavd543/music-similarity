"""Colab-specific adaptations of the ingest pipeline.

Colab's constraints are different from the Cornell cluster the rest of
pipeline/ was written for, and they change the design, not just the
runner:

  - **Ephemeral, small local disk.** GPU runtimes currently get ~64GB of
    local disk (down from the 350GB Colab used to offer), and it's wiped
    every time the runtime recycles. Storing 25-30k tracks' worth of raw
    stems (4 uncompressed WAVs/track) would blow past that by an order of
    magnitude. So this module never keeps more than one track's stems on
    disk at a time: download -> separate -> embed -> delete, immediately,
    per track. Only the small embedding vectors persist.
  - **No durable local disk across sessions.** Everything that must
    survive a disconnect (embeddings, checkpoints, manifests) has to live
    on a mounted Google Drive, not in /content.
  - **Session/idle limits.** Free tier disconnects after ~90 min idle and
    has a hard runtime cap; Pro/Pro+ extend this but nothing is unlimited.
    The loop below appends to a JSONL manifest after every track (not
    batched at the end) so a disconnect mid-run loses at most one track's
    work, and re-running the same cell skips everything already done.
  - **Metered compute.** Every GPU-hour costs compute units, so this
    module also exposes a budget estimator (see `estimate_hours`) so you
    can pick a subset size that fits what you actually bought.

Usage (inside a Colab cell, after mounting Drive -- see
notebooks/music_similarity_colab.ipynb):

    from pipeline.colab_pipeline import run_streaming_ingest
    run_streaming_ingest(
        audio_paths=my_track_list,
        manifest_path=Path("/content/drive/MyDrive/music-similarity/embeddings/manifest.jsonl"),
        demucs_model_name="htdemucs",
    )
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from pipeline.config import MODEL, STEM_NAMES

# torch (and everything downstream of it: demucs, transformers) is imported
# lazily inside the functions that actually need a GPU/model, not at module
# level. That keeps the pure-logic pieces (budget estimators, manifest
# read/write/resume, parquet flattening) importable and unit-testable on a
# machine that doesn't have torch installed at all -- useful for CI and for
# editing this file outside Colab.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Measured compute-unit burn rates (Colab Pay-As-You-Go / Pro, per Google's
# published rates as of mid-2026): T4 ~1.76 CU/hr, A100 ~15 CU/hr. L4 sits
# between the two -- treat this as a rough planning number, not a guarantee,
# since Colab's rates and GPU availability both shift over time.
CU_PER_HOUR = {"T4": 1.76, "L4": 6.0, "A100": 15.0}


def detect_gpu() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    for key in CU_PER_HOUR:
        if key in name:
            return key
    return name  # unrecognized GPU name; caller should treat CU estimate as unavailable


def estimate_hours(compute_units: float, gpu: str) -> float | None:
    """How many GPU-hours `compute_units` buys on the given GPU."""
    rate = CU_PER_HOUR.get(gpu)
    if rate is None:
        return None
    return compute_units / rate


def estimate_feasible_tracks(
    compute_units: float, gpu: str, seconds_per_track: float = 6.0, safety_margin: float = 0.7
) -> int | None:
    """Rough sizing helper: given a compute-unit budget and a GPU, how many
    tracks can you realistically push through Demucs+MERT?
    `seconds_per_track` defaults to a conservative estimate for a 30s clip
    (htdemucs separation + MERT embedding of 4 stems) on a T4-class GPU;
    scale it down for A100. `safety_margin` reserves headroom for retries,
    experimentation, and Phase 2 training runs on the same budget."""
    hours = estimate_hours(compute_units, gpu)
    if hours is None:
        return None
    usable_seconds = hours * 3600 * safety_margin
    return int(usable_seconds / seconds_per_track)


def mount_drive(mount_point: str = "/content/drive") -> Path:
    """Mounts Google Drive if running in Colab; no-ops (and just returns the
    path) if the mount already exists, so re-running a cell is safe."""
    try:
        from google.colab import drive  # type: ignore

        drive.mount(mount_point)
    except ImportError:
        log.warning("Not running in Colab (or google.colab unavailable) -- skipping drive.mount")
    return Path(mount_point)


def setup_colab(
    drive_root: str = "/content/drive/MyDrive/music-similarity",
    mount_point: str = "/content/drive",
) -> Path:
    """One-call setup for the top of a Colab notebook: mounts Drive and
    repoints PATHS/QDRANT at a persistent folder under it. Returns the
    resolved drive_root so callers can use it directly."""
    from pipeline.config import configure_for_colab

    mount_drive(mount_point)
    root = Path(drive_root)
    root.mkdir(parents=True, exist_ok=True)
    configure_for_colab(root)
    log.info("Colab data root: %s", root)
    log.info(gpu_report())
    return root


def _load_completed_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    done = set()
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row["track_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _append_row(manifest_path: Path, row: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _stem_is_silent(wav, rms_threshold: float = 1e-3) -> bool:
    """Jamendo has many instrumentals; Demucs still emits a (near-silent)
    vocals stem for them. Embedding silence produces garbage vectors that
    pollute the vocal axis at search time -- skip those stems instead
    (MultiAspectModel already handles missing stems)."""
    import torch

    return float(torch.sqrt((wav.float() ** 2).mean())) < rms_threshold


def process_one_track(
    audio_path: Path,
    demucs_model,
    embedder,
    scratch_dir: Path,
    device: str,
    keep_lyrics_vocal_stem: bool = False,
) -> dict:
    """Separate, embed, and immediately clean up. Returns one manifest row:
    {track_id, embeddings: {stem: [floats]}, vocal_stem_path (optional, only
    kept transiently if a caller wants to run Whisper on it before cleanup)}."""
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.audio import save_audio

    track_id = audio_path.stem
    track_scratch = scratch_dir / track_id
    track_scratch.mkdir(parents=True, exist_ok=True)

    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    if sr != demucs_model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, demucs_model.samplerate)
    wav = wav.to(device)

    with torch.no_grad():
        sources = apply_model(demucs_model, wav[None], device=device, progress=False)[0]

    embeddings = {}
    vocal_stem_path = None
    for name, source in zip(demucs_model.sources, sources, strict=True):
        if _stem_is_silent(source):
            continue
        stem_wav_path = track_scratch / f"{name}.wav"
        save_audio(source.cpu(), str(stem_wav_path), demucs_model.samplerate)

        stem_tensor, stem_sr = torchaudio.load(str(stem_wav_path))
        embeddings[name] = embedder.embed(stem_tensor, stem_sr).tolist()

        if name == "vocals" and keep_lyrics_vocal_stem:
            vocal_stem_path = stem_wav_path  # caller cleans this up after Whisper
        else:
            stem_wav_path.unlink(missing_ok=True)

    if not keep_lyrics_vocal_stem:
        shutil.rmtree(track_scratch, ignore_errors=True)

    row = {"track_id": track_id, "embeddings": embeddings}
    if vocal_stem_path:
        row["_vocal_stem_scratch_dir"] = str(track_scratch)  # caller must clean up
    return row


def run_streaming_ingest(
    audio_paths: list[Path],
    manifest_path: Path,
    demucs_model_name: str = MODEL.demucs_model,
    mert_model_name: str = MODEL.mert_model,
    scratch_dir: Path | None = None,
    device: str | None = None,
) -> None:
    """Resumable, disk-bounded ingest loop: for each track not already in
    `manifest_path`, separate + embed + delete stems, then append one row.
    Safe to stop and re-run this exact call after a Colab disconnect."""
    import torch
    from demucs.pretrained import get_model

    from pipeline.mert_embed import MertEmbedder

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scratch_dir = scratch_dir or Path(tempfile.mkdtemp(prefix="ms_scratch_"))

    log.info("Loading Demucs (%s) and MERT (%s) on %s", demucs_model_name, mert_model_name, device)
    demucs_model = get_model(demucs_model_name)
    demucs_model.to(device)
    demucs_model.eval()
    embedder = MertEmbedder(mert_model_name, device=device)

    already_done = _load_completed_ids(manifest_path)
    log.info("%d/%d tracks already in manifest, skipping those", len(already_done), len(audio_paths))

    from tqdm import tqdm

    remaining = [p for p in audio_paths if p.stem not in already_done]
    for audio_path in tqdm(remaining, desc="ingest", unit="track"):
        try:
            row = process_one_track(audio_path, demucs_model, embedder, scratch_dir, device)
            _append_row(manifest_path, row)
        except Exception:
            log.exception("Failed on %s -- skipping, will retry on next run", audio_path)
            continue

    log.info("Done. Manifest: %s", manifest_path)


def _load_for_demucs(audio_path: Path, demucs_model, device: str):
    import torchaudio

    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    if sr != demucs_model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, demucs_model.samplerate)
    return wav.to(device)


def _separate_to_stems(audio_path: Path, demucs_model, device: str) -> dict:
    import torch
    from demucs.apply import apply_model

    wav = _load_for_demucs(audio_path, demucs_model, device)
    with torch.no_grad():
        sources = apply_model(demucs_model, wav[None], device=device, progress=False)[0]
    return {name: src.cpu() for name, src in zip(demucs_model.sources, sources, strict=True)}


def process_one_track_with_positive(
    audio_path: Path,
    demucs_model,
    embedder,
    scratch_dir: Path,
    device: str,
    second_segment_path: Path | None = None,
) -> tuple[dict, dict]:
    """Same as process_one_track, but also runs the Phase-2 invariance
    augmentations (training/augmentations.py) on the freshly separated
    stems -- before deleting them -- and embeds the augmented version too.
    This is what makes streaming-and-deleting compatible with contrastive
    head training: anchor and positive embeddings for a track are produced
    in the same pass, so we never need the raw stems again afterward.

    Returns (anchor_row, positive_row), each shaped like
    {"track_id": ..., "embeddings": {stem: [floats]}}.
    """
    from training.augmentations import make_melody_positive, make_rhythm_positive, make_vocal_positive  # noqa: F401

    # Track id: '123__a' (two-segment download) normalizes to '123'.
    track_id = audio_path.stem.removesuffix("__a")
    track_scratch = scratch_dir / track_id
    track_scratch.mkdir(parents=True, exist_ok=True)

    model_sr = demucs_model.samplerate
    stems = _separate_to_stems(audio_path, demucs_model, device)

    # Cross-section mode: when a disjoint second segment (__b) of the same
    # song exists, its stems are *genuinely different sections* -> much
    # stronger positives for drums/vocals/timbre than sub-windows of one 30s
    # clip (attacks the aspect-collapse risk directly). Melody keeps the
    # tempo-warp positive from segment A: a different section has a
    # *different* melody, so cross-section would be the wrong invariance.
    stems_b: dict = {}
    if second_segment_path is not None and second_segment_path.exists():
        stems_b = _separate_to_stems(second_segment_path, demucs_model, device)

    anchor_embeddings, positive_embeddings = {}, {}
    for name in STEM_NAMES:
        if name not in stems:
            continue
        raw = stems[name]
        if _stem_is_silent(raw):
            continue
        anchor_embeddings[name] = embedder.embed(raw, model_sr).tolist()

        cross_section = name in ("drums", "vocals") and name in stems_b and not _stem_is_silent(stems_b[name])
        if cross_section:
            augmented = stems_b[name]
        elif name == "drums":
            # fallback: different sub-window of the same drum stem
            augmented = make_rhythm_positive(raw, model_sr)
        elif name in ("bass", "other"):
            augmented = make_melody_positive(raw, model_sr)
        elif name == "vocals":
            _seg_a, augmented = make_vocal_positive(raw, model_sr)
        else:
            augmented = raw
        positive_embeddings[name] = embedder.embed(augmented, model_sr).tolist()

    shutil.rmtree(track_scratch, ignore_errors=True)
    for wav_path in scratch_dir.glob(f"{track_id}*"):
        wav_path.unlink(missing_ok=True)

    anchor_row = {"track_id": track_id, "embeddings": anchor_embeddings}
    positive_row = {"track_id": track_id, "embeddings": positive_embeddings}
    return anchor_row, positive_row


def run_streaming_pair_ingest(
    audio_paths: list[Path],
    anchor_manifest_path: Path,
    positive_manifest_path: Path,
    demucs_model_name: str = MODEL.demucs_model,
    mert_model_name: str = MODEL.mert_model,
    scratch_dir: Path | None = None,
    device: str | None = None,
) -> None:
    """Phase-2 variant of run_streaming_ingest: produces both the anchor
    and augmented-positive embeddings needed by training/dataset.py's
    AspectPairDataset, in one disk-bounded pass per track."""
    import torch
    from demucs.pretrained import get_model

    from pipeline.mert_embed import MertEmbedder

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scratch_dir = scratch_dir or Path(tempfile.mkdtemp(prefix="ms_pair_scratch_"))

    demucs_model = get_model(demucs_model_name)
    demucs_model.to(device)
    demucs_model.eval()
    embedder = MertEmbedder(mert_model_name, device=device)

    from tqdm import tqdm

    done = _load_completed_ids(anchor_manifest_path) & _load_completed_ids(positive_manifest_path)
    remaining = [p for p in audio_paths if p.stem.removesuffix("__a") not in done]
    log.info("%d/%d tracks remaining for pair ingest", len(remaining), len(audio_paths))

    for i, audio_path in enumerate(tqdm(remaining, desc="pair ingest", unit="track")):
        # two-segment downloads: '123__a.mp3' with sibling '123__b.mp3'
        sibling = audio_path.with_name(audio_path.name.replace("__a", "__b")) if "__a" in audio_path.name else None
        try:
            anchor_row, positive_row = process_one_track_with_positive(
                audio_path, demucs_model, embedder, scratch_dir, device,
                second_segment_path=sibling,
            )
            _append_row(anchor_manifest_path, anchor_row)
            _append_row(positive_manifest_path, positive_row)
        except Exception:
            log.exception("Failed pair-ingest on %s -- will retry next run", audio_path)
            continue
        if (i + 1) % 25 == 0:
            log.info("Pair-ingested %d/%d", i + 1, len(remaining))


def run_streaming_lyrics(
    audio_paths: list[Path],
    lyrics_manifest_path: Path,
    demucs_model_name: str = MODEL.demucs_model,
    whisper_model_name: str = MODEL.whisper_model,
    device: str | None = None,
) -> None:
    """Streaming lyrics pass: separate the vocal stem, skip instrumentals
    (RMS gate), transcribe the rest with faster-whisper. Same resumable
    JSONL-manifest pattern as run_streaming_ingest.

    Jamendo metadata has no artist/title text, so LRCLib lookup isn't
    possible here -- Whisper on the isolated vocal stem is the primary
    source.

    Appends rows: {track_id, lyrics_text (or null), language, source}.
    """
    import numpy as np
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from faster_whisper import WhisperModel
    from tqdm import tqdm

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    demucs_model = get_model(demucs_model_name)
    demucs_model.to(device)
    demucs_model.eval()
    try:
        whisper = WhisperModel(whisper_model_name, device="cuda" if device == "cuda" else "cpu",
                               compute_type="float16" if device == "cuda" else "auto")
    except Exception:
        log.warning("Could not load whisper '%s' -- falling back to 'small'", whisper_model_name)
        whisper = WhisperModel("small", device="cuda" if device == "cuda" else "cpu")

    done = _load_completed_ids(lyrics_manifest_path)
    remaining = [p for p in audio_paths if p.stem not in done]
    log.info("%d/%d tracks remaining for lyrics", len(remaining), len(audio_paths))

    for audio_path in tqdm(remaining, desc="lyrics", unit="track"):
        try:
            track_id = audio_path.stem
            stems = _separate_to_stems(audio_path, demucs_model, device)
            vocals = stems.get("vocals")
            if vocals is None or _stem_is_silent(vocals, rms_threshold=5e-3):
                _append_row(lyrics_manifest_path, {
                    "track_id": track_id, "lyrics_text": None,
                    "language": None, "source": "instrumental",
                })
                continue
            # faster-whisper wants 16kHz mono float32
            mono = vocals.mean(dim=0)
            audio_np = torchaudio.functional.resample(
                mono, demucs_model.samplerate, 16000
            ).numpy().astype(np.float32)
            segments, info = whisper.transcribe(audio_np, vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
            _append_row(lyrics_manifest_path, {
                "track_id": track_id,
                "lyrics_text": text or None,
                "language": getattr(info, "language", None),
                "source": "whisper" if text else "no_speech_detected",
            })
        except Exception:
            log.exception("Lyrics failed on %s -- will retry next run", audio_path)
            continue


def lyrics_manifest_to_vectors(lyrics_manifest_path: Path, out_parquet: Path) -> Path:
    """Embed transcribed lyrics with the sentence-transformer and write
    {track_id, lyric_embedding} parquet (tracks without lyrics are omitted;
    the search fusion already treats a missing lyric vector as no-op)."""
    import pandas as pd

    from pipeline.lyrics import embed_lyrics

    rows = []
    with lyrics_manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("lyrics_text"):
                rows.append({"track_id": r["track_id"], "lyrics_text": r["lyrics_text"]})

    df = pd.DataFrame(rows).drop_duplicates(subset="track_id", keep="last")
    log.info("Embedding lyrics for %d tracks with text", len(df))
    if len(df):
        embeddings = embed_lyrics(df["lyrics_text"].tolist())
        df["lyric"] = [e.tolist() for e in embeddings]
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["lyrics_text"]).to_parquet(out_parquet)
    return out_parquet


def manifest_to_parquet(manifest_path: Path, out_parquet: Path) -> Path:
    """Flattens the streaming {track_id, embeddings: {stem: [...]}} JSONL
    into the long-format (track_id, stem, embedding) parquet that
    pipeline/mert_embed.py's batch output uses, so both code paths feed
    the same downstream training/eval code."""
    import pandas as pd

    rows = []
    skipped = 0
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Tolerate corrupt lines (e.g. a partial write from a crashed/
            # over-quota session), same as _load_completed_ids: any track
            # with a corrupt row was re-ingested on resume, so a clean
            # duplicate of it exists later in the file.
            try:
                record = json.loads(line)
                stems = record["embeddings"].items()
            except (json.JSONDecodeError, KeyError, AttributeError):
                skipped += 1
                continue
            for stem, emb in stems:
                rows.append({"track_id": record["track_id"], "stem": stem, "embedding": emb})
    if skipped:
        log.warning("Skipped %d corrupt manifest line(s) in %s", skipped, manifest_path)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # A crash between the anchor/positive appends in pair ingest can leave a
    # duplicate row for a track after resume; keep the most recent.
    df = df.drop_duplicates(subset=["track_id", "stem"], keep="last")
    df.to_parquet(out_parquet)
    log.info("Wrote %d rows (%d tracks) to %s", len(df), df["track_id"].nunique(), out_parquet)
    return out_parquet


def check_disk_headroom(min_free_gb: float = 5.0, path: str = "/content") -> None:
    """Cheap guardrail to call periodically in a long Colab loop -- raises
    instead of silently filling the ephemeral disk and crashing the runtime."""
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f}GB free on {path} (min {min_free_gb}GB) -- "
            "stop and clear scratch space before continuing."
        )


def gpu_report() -> str:
    gpu = detect_gpu()
    if gpu is None:
        return "No GPU detected -- Runtime > Change runtime type > GPU (T4 is fine to start)."
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True, timeout=10)
        detail = smi.stdout.strip()
    except Exception:
        detail = gpu
    return f"GPU: {detail}"
