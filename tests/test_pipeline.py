"""Tests for src/pipeline/result.py and src/pipeline/detector.py.

Verifies DetectionResult construction, score combination logic,
confidence labeling, and the full detector pipeline on synthetic inputs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.pipeline.result import DetectionResult, combine_scores, _score_to_confidence


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _write_black_video(path: str, n_frames: int = 10) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25.0, (224, 224))
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(black)
    writer.release()


def _write_sine_audio(path: str, duration: float = 2.0, sr: int = 22050) -> None:
    """Write a sine wave as a WAV file using scipy."""
    try:
        from scipy.io import wavfile  # noqa: PLC0415

        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        wavfile.write(path, sr, y)
    except ImportError:
        # Fallback: write raw PCM data as a trivial WAV
        _write_minimal_wav(path, duration, sr)


def _write_minimal_wav(path: str, duration: float, sr: int) -> None:
    """Write a minimal valid WAV file without scipy."""
    import struct  # noqa: PLC0415

    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    data = samples.tobytes()

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


# ──────────────────────────────────────────────
# DetectionResult tests
# ──────────────────────────────────────────────

class TestDetectionResult:
    def test_construction_sets_is_deepfake_true(self) -> None:
        """combined_score above threshold should set is_deepfake=True."""
        result = DetectionResult(
            video_score=0.8,
            audio_score=0.7,
            combined_score=0.76,
            threshold=0.65,
        )
        assert result.is_deepfake is True

    def test_construction_sets_is_deepfake_false(self) -> None:
        """combined_score below threshold should set is_deepfake=False."""
        result = DetectionResult(
            video_score=0.3,
            audio_score=0.2,
            combined_score=0.26,
            threshold=0.65,
        )
        assert result.is_deepfake is False

    def test_verdict_deepfake(self) -> None:
        result = DetectionResult(video_score=0.9, audio_score=None, combined_score=0.9)
        assert result.verdict == "LIKELY DEEPFAKE"
        assert result.verdict_emoji == "⚠"

    def test_verdict_real(self) -> None:
        result = DetectionResult(video_score=0.1, audio_score=None, combined_score=0.1)
        assert result.verdict == "LIKELY REAL"
        assert result.verdict_emoji == "✓"

    def test_confidence_high_for_extreme_score(self) -> None:
        result = DetectionResult(video_score=0.95, audio_score=None, combined_score=0.95)
        assert result.confidence_label == "High"

    def test_confidence_low_near_boundary(self) -> None:
        result = DetectionResult(video_score=0.65, audio_score=None, combined_score=0.65)
        assert result.confidence_label == "Low"

    def test_to_dict_contains_required_keys(self) -> None:
        result = DetectionResult(video_score=0.7, audio_score=0.6, combined_score=0.66)
        d = result.to_dict()
        for key in ("video_score", "audio_score", "combined_score", "is_deepfake",
                    "verdict", "confidence", "flags", "processing_time_ms"):
            assert key in d

    def test_flags_default_empty(self) -> None:
        result = DetectionResult(video_score=0.5, audio_score=None, combined_score=0.5)
        assert result.flags == []


# ──────────────────────────────────────────────
# combine_scores tests
# ──────────────────────────────────────────────

class TestCombineScores:
    def test_both_modalities(self) -> None:
        score = combine_scores(0.8, 0.4, video_weight=0.6, audio_weight=0.4)
        assert abs(score - (0.6 * 0.8 + 0.4 * 0.4)) < 1e-6

    def test_video_only(self) -> None:
        score = combine_scores(0.7, None)
        assert score == 0.7

    def test_audio_only(self) -> None:
        score = combine_scores(None, 0.3)
        assert score == 0.3

    def test_neither_returns_zero(self) -> None:
        score = combine_scores(None, None)
        assert score == 0.0

    def test_score_in_range(self) -> None:
        """Fusion of valid scores should stay in [0, 1]."""
        for v, a in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.999, 0.001)]:
            assert 0.0 <= combine_scores(v, a) <= 1.0


# ──────────────────────────────────────────────
# _score_to_confidence tests
# ──────────────────────────────────────────────

class TestScoreToConfidence:
    def test_high_confidence_low_score(self) -> None:
        assert _score_to_confidence(0.05) == "High"

    def test_high_confidence_high_score(self) -> None:
        assert _score_to_confidence(0.95) == "High"

    def test_low_confidence_boundary(self) -> None:
        assert _score_to_confidence(0.65) == "Low"

    def test_medium_confidence(self) -> None:
        # 0.50 is 0.15 from boundary — medium
        assert _score_to_confidence(0.50) == "Medium"


# ──────────────────────────────────────────────
# EchoGuardDetector integration tests
# ──────────────────────────────────────────────

class TestEchoGuardDetector:
    def test_analyze_requires_at_least_one_input(self) -> None:
        from src.pipeline.detector import EchoGuardDetector  # noqa: PLC0415

        detector = EchoGuardDetector()
        with pytest.raises(ValueError, match="[Aa]t least one"):
            detector.analyze()

    def test_analyze_video_only_returns_result(self) -> None:
        """Full pipeline on a black video should return a DetectionResult."""
        from src.pipeline.detector import EchoGuardDetector  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = str(Path(tmpdir) / "black.mp4")
            _write_black_video(video_path)

            detector = EchoGuardDetector(max_frames=5)
            result = detector.analyze(video_path=video_path)

        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.combined_score <= 1.0
        assert isinstance(result.is_deepfake, bool)
        assert isinstance(result.flags, list)
        assert result.processing_time_ms > 0

    def test_analyze_audio_only_returns_result(self) -> None:
        """Full pipeline on a sine wave audio file should return a DetectionResult."""
        from src.pipeline.detector import EchoGuardDetector  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = str(Path(tmpdir) / "sine.wav")
            _write_sine_audio(audio_path)

            detector = EchoGuardDetector()
            result = detector.analyze(audio_path=audio_path)

        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.combined_score <= 1.0
        assert result.video_score is None
        assert result.audio_score is not None

    def test_combine_scores_method(self) -> None:
        from src.pipeline.detector import EchoGuardDetector  # noqa: PLC0415

        detector = EchoGuardDetector(video_weight=0.6, audio_weight=0.4)
        score = detector._combine_scores(0.8, 0.4)
        expected = 0.6 * 0.8 + 0.4 * 0.4
        assert abs(score - expected) < 1e-6

    def test_combine_scores_video_only(self) -> None:
        from src.pipeline.detector import EchoGuardDetector  # noqa: PLC0415

        detector = EchoGuardDetector()
        assert detector._combine_scores(0.7, None) == 0.7

    def test_combine_scores_audio_only(self) -> None:
        from src.pipeline.detector import EchoGuardDetector  # noqa: PLC0415

        detector = EchoGuardDetector()
        assert detector._combine_scores(None, 0.3) == 0.3
