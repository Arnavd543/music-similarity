"""Batched, checkpointed Demucs stem separation.

Designed to run as a Slurm array job on the Cornell cluster over thousands
of tracks: each invocation processes a shard of the input list, skips
tracks whose stems already exist (resumability after preemption), and
writes a manifest row per completed track so a job array can be re-run
idempotently.

Local usage (tiny sample, CPU or single GPU):
    python -m pipeline.demucs_stems --audio-dir data/audio --out data/stems

Cluster usage (array job, shard 3 of 20):
    python -m pipeline.demucs_stems --audio-dir data/audio --out data/stems \
        --shard-index 3 --num-shards 20
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch

from pipeline.config import MODEL, PATHS, STEM_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def list_shard(audio_dir: Path, shard_index: int, num_shards: int) -> list[Path]:
    files = sorted(p for p in audio_dir.glob("*") if p.suffix.lower() in (".mp3", ".wav", ".flac"))
    return [f for i, f in enumerate(files) if i % num_shards == shard_index]


def already_done(track_id: str, out_dir: Path) -> bool:
    track_out = out_dir / track_id
    return all((track_out / f"{stem}.wav").exists() for stem in STEM_NAMES)


def separate_track(model, audio_path: Path, out_dir: Path, device: str) -> None:
    """Run Demucs on a single track and write 4 stem WAVs."""
    import torchaudio
    from demucs.apply import apply_model
    from demucs.audio import save_audio

    track_id = audio_path.stem
    track_out = out_dir / track_id
    track_out.mkdir(parents=True, exist_ok=True)

    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # demucs expects stereo
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)

    wav = wav.to(device)
    with torch.no_grad():
        sources = apply_model(model, wav[None], device=device, progress=False)[0]

    for name, source in zip(model.sources, sources, strict=True):
        save_audio(source.cpu(), str(track_out / f"{name}.wav"), model.samplerate)

    log.info("Separated %s -> %s", track_id, track_out)


def append_manifest(manifest_path: Path, track_id: str, status: str) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not manifest_path.exists()
    with manifest_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["track_id", "status"])
        writer.writerow([track_id, status])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=PATHS.audio_dir)
    parser.add_argument("--out", type=Path, default=PATHS.stems_dir)
    parser.add_argument("--model", default=MODEL.demucs_model)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from demucs.pretrained import get_model

    log.info("Loading Demucs model %s on %s", args.model, args.device)
    model = get_model(args.model)
    model.to(args.device)
    model.eval()

    shard_files = list_shard(args.audio_dir, args.shard_index, args.num_shards)
    log.info(
        "Shard %d/%d: %d tracks to process", args.shard_index, args.num_shards, len(shard_files)
    )

    manifest_path = args.out / f"_manifest_shard{args.shard_index}.csv"
    for audio_path in shard_files:
        track_id = audio_path.stem
        if already_done(track_id, args.out):
            log.info("Skipping %s (already separated)", track_id)
            continue
        try:
            separate_track(model, audio_path, args.out, args.device)
            append_manifest(manifest_path, track_id, "ok")
        except Exception:
            log.exception("Failed on %s", track_id)
            append_manifest(manifest_path, track_id, "failed")


if __name__ == "__main__":
    main()
