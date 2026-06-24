"""Detection result data structures for the EchoGuard pipeline.

Defines the DetectionResult dataclass that is the canonical output of
every analysis run, carrying scores, flags, timing, and human-readable labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectionResult:
    """Unified output of an EchoGuard deepfake detection run.

    Combines video and audio analysis into a single structured result
    with an explainable set of flags and a confidence-calibrated verdict.

    Attributes:
        video_score: Deepfake probability from video analysis (0.0–1.0).
            None if no video was analyzed.
        audio_score: Voice clone probability from audio analysis (0.0–1.0).
            None if no audio was analyzed.
        combined_score: Weighted fusion (60% video + 40% audio) in [0.0, 1.0].
        flags: Specific anomaly flags raised (e.g. 'lip_sync_mismatch').
        is_deepfake: True when combined_score exceeds the detection threshold.
        processing_time_ms: Wall-clock analysis time in milliseconds.
        frame_count: Number of video frames analyzed (0 for audio-only).
        confidence_label: Human-readable confidence tier — 'Low', 'Medium', 'High'.
        threshold: The detection threshold used for this result.
    """

    video_score: Optional[float]
    audio_score: Optional[float]
    combined_score: float
    flags: list[str] = field(default_factory=list)
    is_deepfake: bool = False
    processing_time_ms: float = 0.0
    frame_count: int = 0
    confidence_label: str = "Low"
    threshold: float = 0.65

    def __post_init__(self) -> None:
        """Derive is_deepfake and confidence_label from the combined score."""
        self.is_deepfake = self.combined_score >= self.threshold
        self.confidence_label = _score_to_confidence(self.combined_score)

    @property
    def verdict(self) -> str:
        """Return a short human-readable verdict string."""
        if self.is_deepfake:
            return "LIKELY DEEPFAKE"
        return "LIKELY REAL"

    @property
    def verdict_emoji(self) -> str:
        """Return a warning or check emoji appropriate to the verdict."""
        return "⚠" if self.is_deepfake else "✓"

    def to_dict(self) -> dict:
        """Serialize the result to a plain dictionary for JSON export."""
        return {
            "video_score": self.video_score,
            "audio_score": self.audio_score,
            "combined_score": round(self.combined_score, 4),
            "is_deepfake": self.is_deepfake,
            "verdict": self.verdict,
            "confidence": self.confidence_label,
            "flags": self.flags,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "frame_count": self.frame_count,
            "threshold": self.threshold,
        }


def _score_to_confidence(score: float) -> str:
    """Map a combined score to a human-readable confidence tier.

    The confidence reflects how far the score is from the decision boundary (0.65),
    not how certain we are that the content is deepfake. Both 0.05 and 0.95 are
    high-confidence; 0.60 and 0.70 are low-confidence (near the boundary).
    """
    distance_from_boundary = abs(score - 0.65)
    if distance_from_boundary > 0.25:
        return "High"
    elif distance_from_boundary > 0.10:
        return "Medium"
    return "Low"


def combine_scores(
    video_score: Optional[float],
    audio_score: Optional[float],
    video_weight: float = 0.6,
    audio_weight: float = 0.4,
) -> float:
    """Compute a weighted fusion of video and audio anomaly scores.

    If only one modality is available, its score is used directly.
    If neither is available, returns 0.0.

    Args:
        video_score: Video deepfake score in [0.0, 1.0] or None.
        audio_score: Audio clone score in [0.0, 1.0] or None.
        video_weight: Weight for video score (default 0.6).
        audio_weight: Weight for audio score (default 0.4).

    Returns:
        Combined score in [0.0, 1.0].
    """
    if video_score is not None and audio_score is not None:
        return video_weight * video_score + audio_weight * audio_score
    elif video_score is not None:
        return video_score
    elif audio_score is not None:
        return audio_score
    return 0.0
