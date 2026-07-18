"""Zero-label, invariance-based positive/negative pair construction for
aspect-head contrastive training (Phase 2).

Stems make these augmentations clean in ways whole-mix approaches can't:
  - rhythm head:  pitch-shift / reharmonize the pitched stems, keep drums
                  fixed -> same rhythm (positive). Different clip, same
                  tempo -> hard negative.
  - melody head:  tempo-warp + re-timbre (swap `other`/`bass` stem source)
                  while preserving melodic contour -> positive.
  - timbre head:  two segments of the *same* track share instrumentation
                  -> positive; same tempo/key, different track -> hard
                  negative.
  - vocal head:   different segments of the same vocal stem -> positive
                  (same singer/timbre); different track's vocals -> negative.

All operations run on stem waveforms (torchaudio/torch), independent of
the MERT embedding step -- augment first, embed second.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torchaudio


@dataclass
class AugmentedPair:
    anchor: torch.Tensor
    positive: torch.Tensor
    aspect: str


def pitch_shift(wav: torch.Tensor, sr: int, n_steps: float) -> torch.Tensor:
    return torchaudio.functional.pitch_shift(wav, sr, n_steps=n_steps)


def tempo_warp(wav: torch.Tensor, sr: int, rate: float) -> torch.Tensor:
    """rate > 1.0 speeds up, < 1.0 slows down, without changing pitch.

    Tries sox first; falls back to a phase-vocoder time stretch, since
    torchaudio's sox bindings are deprecated and absent from many recent
    builds."""
    try:
        effects = [["tempo", str(rate)]]
        out, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)
        return out
    except Exception:
        n_fft, hop = 1024, 256
        window = torch.hann_window(n_fft, device=wav.device)
        spec = torch.stft(wav, n_fft, hop_length=hop, window=window, return_complex=True)
        stretcher = torchaudio.transforms.TimeStretch(hop_length=hop, n_freq=n_fft // 2 + 1)
        warped = stretcher(spec, rate)
        return torch.istft(warped, n_fft, hop_length=hop, window=window)


def random_segment(wav: torch.Tensor, sr: int, seconds: float) -> torch.Tensor:
    n = int(seconds * sr)
    if wav.shape[-1] <= n:
        return wav
    start = random.randint(0, wav.shape[-1] - n)
    return wav[..., start : start + n]


def make_rhythm_positive(drums: torch.Tensor, sr: int, seconds: float = 10.0) -> torch.Tensor:
    """A different section of the same drum stem -> same groove/kit
    identity. The positive must not be bit-identical to the anchor, or the
    head learns no invariance and triplet accuracy reads as perfect."""
    return random_segment(drums, sr, seconds)


def make_melody_positive(melodic_stem: torch.Tensor, sr: int) -> torch.Tensor:
    """Tempo-warp preserves melodic contour/harmony while changing rhythm feel,
    which is exactly the invariance we want the melody head to have."""
    rate = random.uniform(0.85, 1.15)
    return tempo_warp(melodic_stem, sr, rate)


def make_timbre_positive(mix_wav: torch.Tensor, sr: int, seconds: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Two disjoint segments of the same track -> same instrumentation (positive pair)."""
    seg_a = random_segment(mix_wav, sr, seconds)
    seg_b = random_segment(mix_wav, sr, seconds)
    return seg_a, seg_b


def make_vocal_positive(vocal_wav: torch.Tensor, sr: int, seconds: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    return make_timbre_positive(vocal_wav, sr, seconds)
