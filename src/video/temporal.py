"""Temporal anomaly detection across sequences of video frames.

Analyzes sliding windows of landmark data to detect unnatural motion patterns,
texture flickering, and abnormal blink frequency characteristic of deepfakes.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from src.video.face_detector import LandmarkData

logger = logging.getLogger(__name__)

# Natural blink rate range in blinks per minute
_NATURAL_BLINK_MIN = 10.0
_NATURAL_BLINK_MAX = 30.0


class TemporalAnalyzer:
    """Detects temporal anomalies across a sequence of frames.

    Deepfakes often exhibit subtle but consistent patterns:
    - Unnaturally smooth or jerky head movements
    - Pixel-level flickering in face texture
    - Blink frequency outside the natural human range
    """

    def __init__(
        self,
        fps: float = 25.0,
        flicker_window: int = 5,
        pose_change_threshold: float = 15.0,
        flicker_threshold: float = 0.15,
        blink_rate_min: float = _NATURAL_BLINK_MIN,
        blink_rate_max: float = _NATURAL_BLINK_MAX,
    ) -> None:
        """Initialize the temporal analyzer.

        Args:
            fps: Video frame rate (used to convert blink count to blinks/min).
            flicker_window: Number of consecutive frames used in the flickering check.
            pose_change_threshold: Max degrees of head pose change per frame before flagging.
            flicker_threshold: Normalized pixel variance threshold for flickering detection.
            blink_rate_min: Minimum natural blink rate (blinks/min).
            blink_rate_max: Maximum natural blink rate (blinks/min).
        """
        self.fps = fps
        self.flicker_window = flicker_window
        self.pose_change_threshold = pose_change_threshold
        self.flicker_threshold = flicker_threshold
        self.blink_rate_min = blink_rate_min
        self.blink_rate_max = blink_rate_max

    def analyze(
        self,
        landmark_sequence: List[LandmarkData],
        frame_sequence: List[np.ndarray],
    ) -> tuple[float, list[str]]:
        """Compute an overall temporal anomaly score for a sequence of frames.

        Args:
            landmark_sequence: Per-frame landmark data produced by FaceDetector.
            frame_sequence: Corresponding BGR frame arrays from OpenCV.

        Returns:
            Tuple of (anomaly_score: float 0.0–1.0, flags: list[str]).
        """
        if len(landmark_sequence) < 2:
            return 0.0, []

        flags: list[str] = []
        scores: list[float] = []

        pose_score, pose_flags = self._check_pose_changes(landmark_sequence)
        scores.append(pose_score)
        flags.extend(pose_flags)

        blink_score, blink_flags = self._check_blink_rate(landmark_sequence)
        scores.append(blink_score)
        flags.extend(blink_flags)

        if frame_sequence:
            flicker_score, flicker_flags = self._check_texture_flickering(frame_sequence)
            scores.append(flicker_score)
            flags.extend(flicker_flags)

        combined = float(np.mean(scores)) if scores else 0.0
        return min(combined, 1.0), flags

    def _check_pose_changes(
        self, sequence: List[LandmarkData]
    ) -> tuple[float, list[str]]:
        """Flag frames where head pose changes unnaturally between consecutive frames."""
        abrupt_changes = 0
        valid_pairs = 0

        for prev, curr in zip(sequence[:-1], sequence[1:]):
            if prev.landmarks is None or curr.landmarks is None:
                continue
            valid_pairs += 1
            prev_angles = np.array(prev.head_pose_angles)
            curr_angles = np.array(curr.head_pose_angles)
            change = float(np.max(np.abs(curr_angles - prev_angles)))
            if change > self.pose_change_threshold:
                abrupt_changes += 1

        if valid_pairs == 0:
            return 0.0, []

        ratio = abrupt_changes / valid_pairs
        flags = ["unnatural_head_motion"] if ratio > 0.1 else []
        return min(ratio * 3.0, 1.0), flags

    def _check_blink_rate(
        self, sequence: List[LandmarkData]
    ) -> tuple[float, list[str]]:
        """Detect blink frequency outside the natural human range (10–30 blinks/min)."""
        blink_count = sum(1 for ld in sequence if ld.is_blink)
        duration_seconds = len(sequence) / self.fps
        if duration_seconds < 1.0:
            return 0.0, []

        blinks_per_minute = (blink_count / duration_seconds) * 60.0
        logger.debug("Blink rate: %.1f blinks/min", blinks_per_minute)

        if blinks_per_minute < self.blink_rate_min:
            # Too few blinks — common in synthetic faces
            deviation = (self.blink_rate_min - blinks_per_minute) / self.blink_rate_min
            return min(deviation, 1.0), ["abnormal_blink_rate"]
        elif blinks_per_minute > self.blink_rate_max:
            deviation = (blinks_per_minute - self.blink_rate_max) / self.blink_rate_max
            return min(deviation, 1.0), ["abnormal_blink_rate"]

        return 0.0, []

    def _check_texture_flickering(
        self, frames: List[np.ndarray]
    ) -> tuple[float, list[str]]:
        """Detect frame-to-frame texture flickering using pixel variance in a sliding window.

        Deepfake rendering artifacts often manifest as inconsistent pixel values
        that oscillate between frames, producing higher-than-normal variance.
        """
        if len(frames) < self.flicker_window:
            return 0.0, []

        variances: list[float] = []
        for i in range(len(frames) - self.flicker_window + 1):
            window = frames[i : i + self.flicker_window]
            try:
                stack = np.stack([f.astype(np.float32) for f in window], axis=0)
                # Normalize to [0, 1] before computing variance
                var = float(np.mean(np.var(stack / 255.0, axis=0)))
                variances.append(var)
            except Exception:
                continue

        if not variances:
            return 0.0, []

        mean_variance = float(np.mean(variances))
        logger.debug("Mean texture flicker variance: %.4f", mean_variance)

        if mean_variance > self.flicker_threshold:
            score = min((mean_variance - self.flicker_threshold) / self.flicker_threshold, 1.0)
            return score, ["temporal_flicker"]

        return 0.0, []
