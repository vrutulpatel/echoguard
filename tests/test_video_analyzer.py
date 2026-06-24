"""Tests for src/video/analyzer.py and related video analysis modules.

Uses synthetic data (black video frames) to test without real video files
or pretrained models. All tests must pass with only NumPy/OpenCV installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video.face_detector import FaceDetector, LandmarkData
from src.video.temporal import TemporalAnalyzer
from src.video.analyzer import VideoAnalyzer, _laplacian_variance


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _write_black_video(path: str, n_frames: int = 30, fps: float = 25.0) -> None:
    """Write an all-black video with the given number of frames to disk."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (224, 224))
    assert writer.isOpened(), f"VideoWriter could not open: {path}"
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(black)
    writer.release()


def _black_frames(n: int = 10) -> list[np.ndarray]:
    return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(n)]


def _empty_landmarks(n: int = 10) -> list[LandmarkData]:
    return [LandmarkData(frame_idx=i, landmarks=None) for i in range(n)]


# ──────────────────────────────────────────────
# FaceDetector tests
# ──────────────────────────────────────────────

class TestFaceDetector:
    def test_init_does_not_raise(self) -> None:
        """FaceDetector should initialize without raising even if MediaPipe is absent."""
        detector = FaceDetector()
        assert detector is not None

    def test_process_black_frame_returns_landmark_data(self) -> None:
        """Processing a black frame should return LandmarkData with no face detected."""
        detector = FaceDetector()
        black = np.zeros((224, 224, 3), dtype=np.uint8)
        result = detector.process_frame(black, frame_idx=0)
        assert isinstance(result, LandmarkData)
        assert result.frame_idx == 0

    def test_process_frame_no_face(self) -> None:
        """A black frame has no face; landmarks should be None."""
        detector = FaceDetector()
        result = detector.process_frame(np.zeros((224, 224, 3), dtype=np.uint8), 0)
        assert result.landmarks is None
        assert result.detection_confidence == 0.0

    def test_close_does_not_raise(self) -> None:
        """close() should be safe to call even without a loaded model."""
        detector = FaceDetector()
        detector.close()  # should not raise


# ──────────────────────────────────────────────
# TemporalAnalyzer tests
# ──────────────────────────────────────────────

class TestTemporalAnalyzer:
    def test_analyze_empty_sequence_returns_zero(self) -> None:
        analyzer = TemporalAnalyzer()
        score, flags = analyzer.analyze([], [])
        assert score == 0.0
        assert flags == []

    def test_analyze_single_frame_returns_zero(self) -> None:
        analyzer = TemporalAnalyzer()
        lm = _empty_landmarks(1)
        frames = _black_frames(1)
        score, flags = analyzer.analyze(lm, frames)
        assert score == 0.0

    def test_analyze_returns_score_in_range(self) -> None:
        """Score must be in [0.0, 1.0] for any input."""
        analyzer = TemporalAnalyzer(fps=25.0)
        lm = _empty_landmarks(20)
        frames = _black_frames(20)
        score, flags = analyzer.analyze(lm, frames)
        assert 0.0 <= score <= 1.0
        assert isinstance(flags, list)

    def test_blink_rate_low_blinks_flagged(self) -> None:
        """Sequence with no blinks over 10 seconds should trigger abnormal_blink_rate."""
        analyzer = TemporalAnalyzer(fps=25.0, blink_rate_min=10.0)
        # 250 frames at 25fps = 10 seconds, zero blinks
        lm = [LandmarkData(frame_idx=i, landmarks=None, is_blink=False) for i in range(250)]
        score, flags = analyzer._check_blink_rate(lm)
        assert score > 0.0
        assert "abnormal_blink_rate" in flags

    def test_texture_flicker_black_frames_low_variance(self) -> None:
        """Uniform black frames should produce near-zero flicker variance."""
        analyzer = TemporalAnalyzer(flicker_threshold=0.05)
        frames = _black_frames(10)
        score, flags = analyzer._check_texture_flickering(frames)
        assert score == 0.0
        assert "temporal_flicker" not in flags


# ──────────────────────────────────────────────
# VideoAnalyzer tests
# ──────────────────────────────────────────────

class TestVideoAnalyzer:
    def test_analyze_black_video_returns_score(self) -> None:
        """Analyzing a black video file should return a float score without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = str(Path(tmpdir) / "black.mp4")
            _write_black_video(video_path, n_frames=30)

            analyzer = VideoAnalyzer(frame_sample_rate=5, max_frames=10)
            score, flags, landmarks, frames = analyzer.analyze(video_path)

            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            assert isinstance(flags, list)
            assert isinstance(landmarks, list)

    def test_analyze_nonexistent_file_raises(self) -> None:
        analyzer = VideoAnalyzer()
        with pytest.raises(FileNotFoundError):
            analyzer.analyze("/does/not/exist.mp4")

    def test_laplacian_variance_black_frame_near_zero(self) -> None:
        """A uniform black region has near-zero Laplacian variance."""
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        var = _laplacian_variance(black)
        assert var < 1.0

    def test_laplacian_variance_noisy_frame_higher(self) -> None:
        """A noisy frame should produce higher Laplacian variance than black."""
        noise = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        assert _laplacian_variance(noise) > _laplacian_variance(black)
