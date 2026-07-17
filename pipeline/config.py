"""Central configuration for the music-similarity pipeline.

All paths/hosts are overridable via environment variables so the same code
runs unchanged on a laptop (tiny local sample), the Cornell cluster (batch
jobs over the full MTG-Jamendo subset), and Modal/Replicate (single-track
upload inference).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ASPECTS = ["rhythm", "melody", "timbre", "vocal", "lyric"]

# Which stems feed which aspect head, per the architecture doc:
#   rhythm (drums) · melody/harmony (other+bass) · timbre (all-stem) · vocal (vocals)
ASPECT_STEM_MAP: dict[str, list[str]] = {
    "rhythm": ["drums"],
    "melody": ["bass", "other"],
    "timbre": ["drums", "bass", "vocals", "other"],
    "vocal": ["vocals"],
}

STEM_NAMES = ["drums", "bass", "vocals", "other"]


@dataclass
class Paths:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("MS_DATA_DIR", REPO_ROOT / "data")))

    @property
    def audio_dir(self) -> Path:
        # Audio is bulky (~1.2MB/track) but re-downloadable, so it can live on
        # ephemeral local disk (e.g. /content on Colab) instead of Drive --
        # set MS_AUDIO_DIR to override independently of data_dir.
        override = os.getenv("MS_AUDIO_DIR")
        return Path(override) if override else self.data_dir / "audio"

    @property
    def stems_dir(self) -> Path:
        return self.data_dir / "stems"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def lyrics_dir(self) -> Path:
        return self.data_dir / "lyrics"

    @property
    def checkpoints_dir(self) -> Path:
        return self.data_dir / "checkpoints"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    def ensure_all(self) -> None:
        for p in [
            self.audio_dir,
            self.stems_dir,
            self.embeddings_dir,
            self.lyrics_dir,
            self.checkpoints_dir,
            self.metadata_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    demucs_model: str = os.getenv("MS_DEMUCS_MODEL", "htdemucs")
    mert_model: str = os.getenv("MS_MERT_MODEL", "m-a-p/MERT-v1-95M")
    mert_embed_dim: int = 768
    # NB: "sentence-transformers/gte-base" is not a real HF repo; the GTE
    # models live under thenlper/ (768-dim output).
    sentence_model: str = os.getenv("MS_SENTENCE_MODEL", "thenlper/gte-base")
    lyric_embed_dim: int = 768  # dim of sentence_model's output, != head_output_dim
    whisper_model: str = os.getenv("MS_WHISPER_MODEL", "large-v3-turbo")
    head_hidden_dim: int = 256
    head_output_dim: int = 128
    sample_rate: int = 24000  # MERT's native input rate
    segment_seconds: int = 30


@dataclass
class QdrantConfig:
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    url: str | None = os.getenv("QDRANT_URL")  # set for Qdrant Cloud
    api_key: str | None = os.getenv("QDRANT_API_KEY")
    collection: str = os.getenv("QDRANT_COLLECTION", "tracks")
    # Local (embedded, no-server) mode: set to a directory path -- typically
    # somewhere under a mounted Google Drive so it survives Colab runtime
    # recycling -- and QdrantClient(path=...) is used instead of a host/port
    # or cloud connection. Takes priority over `url` and `host`/`port` when set.
    local_path: str | None = os.getenv("QDRANT_LOCAL_PATH")


@dataclass
class TrainConfig:
    batch_size: int = int(os.getenv("MS_BATCH_SIZE", "64"))
    lr: float = float(os.getenv("MS_LR", "3e-4"))
    epochs: int = int(os.getenv("MS_EPOCHS", "20"))
    temperature: float = float(os.getenv("MS_TEMPERATURE", "0.07"))
    seed: int = 42


PATHS = Paths()
MODEL = ModelConfig()
QDRANT = QdrantConfig()
TRAIN = TrainConfig()


def configure_for_colab(drive_root: Path) -> None:
    """Repoints PATHS and QDRANT at a directory under mounted Google Drive
    (e.g. Path("/content/drive/MyDrive/music-similarity")) so embeddings,
    checkpoints, and the local Qdrant index survive a runtime recycle.
    Call this once, right after drive.mount(), before running any pipeline
    step -- it mutates the module-level PATHS/QDRANT singletons in place so
    every other module's `from pipeline.config import PATHS` sees the change."""
    PATHS.data_dir = drive_root / "data"
    PATHS.ensure_all()
    QDRANT.local_path = str(drive_root / "qdrant_local")
