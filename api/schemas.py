"""Pydantic request/response models for the search + upload API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from pipeline.config import ASPECTS


class SliderWeights(BaseModel):
    rhythm: float = Field(0.2, ge=0.0, le=1.0)
    melody: float = Field(0.2, ge=0.0, le=1.0)
    timbre: float = Field(0.2, ge=0.0, le=1.0)
    vocal: float = Field(0.2, ge=0.0, le=1.0)
    lyric: float = Field(0.2, ge=0.0, le=1.0)

    def as_dict(self) -> dict[str, float]:
        d = self.model_dump()
        total = sum(d.values()) or 1.0
        return {k: v / total for k, v in d.items()}  # normalize to sum 1


class SearchRequest(BaseModel):
    seed_track_id: str
    weights: SliderWeights = SliderWeights()
    top_k: int = Field(20, ge=1, le=100)
    user_id: str | None = None  # if set, personalized weights override `weights`


class SearchResultItem(BaseModel):
    track_id: str
    score: float
    per_aspect_score: dict[str, float]
    path: str | None = None  # Jamendo CDN path for a playable audio URL


class SearchResponse(BaseModel):
    seed_track_id: str
    weights_used: dict[str, float]
    results: list[SearchResultItem]
    latency_ms: float


class UploadJobStatus(BaseModel):
    job_id: str
    status: str  # queued | processing | done | failed
    track_id: str | None = None
    error: str | None = None


class PairwiseFeedback(BaseModel):
    user_id: str
    seed_track_id: str
    candidate_a: str
    candidate_b: str
    chose_a: bool


assert set(SliderWeights.model_fields) == set(ASPECTS), "SliderWeights must mirror config.ASPECTS"
