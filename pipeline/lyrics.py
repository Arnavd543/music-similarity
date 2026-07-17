"""Lyrics pipeline: LRCLib lookup first, Whisper transcription of the vocal
stem as fallback (Jamendo's lyric-DB coverage is thin, so Whisper is the
primary source in practice), then embed with a sentence-transformer.

Output parquet: track_id | lyrics_text | source (lrclib|whisper) | embedding
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import requests

from pipeline.config import MODEL, PATHS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"


def lrclib_lookup(artist: str, title: str) -> str | None:
    """Free, unauthenticated API. Returns plain lyrics text or None on miss."""
    try:
        resp = requests.get(
            LRCLIB_SEARCH_URL, params={"artist_name": artist, "track_name": title}, timeout=10
        )
        resp.raise_for_status()
        results = resp.json()
    except requests.RequestException:
        log.warning("LRCLib request failed for %s - %s", artist, title)
        return None

    for r in results:
        text = r.get("plainLyrics")
        if text:
            return text
    return None


def whisper_transcribe(vocal_stem_path: Path, model_size: str = MODEL.whisper_model) -> str | None:
    """Fallback: transcribe the isolated vocal stem with faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.error("faster-whisper not installed; run `uv pip install faster-whisper`")
        return None

    model = WhisperModel(model_size, compute_type="auto")
    segments, _info = model.transcribe(str(vocal_stem_path), vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip() or None


def get_lyrics_for_track(
    track_id: str, artist: str, title: str, vocal_stem_path: Path | None
) -> tuple[str | None, str]:
    text = lrclib_lookup(artist, title)
    if text:
        return text, "lrclib"
    if vocal_stem_path and vocal_stem_path.exists():
        text = whisper_transcribe(vocal_stem_path)
        if text:
            return text, "whisper"
    return None, "none"


def embed_lyrics(texts: list[str], model_name: str = MODEL.sentence_model):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path, default=PATHS.metadata_dir / "jamendo.tsv",
        help="TSV with columns including TRACK_ID, ARTIST_NAME, TITLE (as in MTG-Jamendo)",
    )
    parser.add_argument("--stems-dir", type=Path, default=PATHS.stems_dir)
    parser.add_argument("--out", type=Path, default=PATHS.lyrics_dir / "lyrics.parquet")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata, sep="\t")
    rows = []
    for _, row in meta.iterrows():
        track_id = str(row["TRACK_ID"]).replace("track_", "")
        artist = row.get("ARTIST_NAME", row.get("ARTIST", ""))
        title = row.get("TITLE", row.get("TRACK_NAME", ""))
        vocal_path = args.stems_dir / track_id / "vocals.wav"

        text, source = get_lyrics_for_track(track_id, artist, title, vocal_path)
        rows.append({"track_id": track_id, "lyrics_text": text, "source": source})

    df = pd.DataFrame(rows)
    has_text = df["lyrics_text"].notna()
    log.info("Lyrics found for %d / %d tracks", has_text.sum(), len(df))

    if has_text.any():
        embeddings = embed_lyrics(df.loc[has_text, "lyrics_text"].tolist())
        df.loc[has_text, "embedding"] = list(embeddings.tolist())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
