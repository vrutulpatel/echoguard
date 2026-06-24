"""Core video deepfake analysis module.

Loads a video file frame by frame, runs face detection, and computes multiple
anomaly signals: lip-sync inconsistency, lighting artifacts, and edge blurring.
Delegates temporal analysis to the TemporalAnalyzer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.video.face_detector import FaceDetector, LandmarkData
from src.video.temporal import TemporalAnalyzer

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Analyzes a video file for deepfake indicators using computer vision heuristics.

    Combines per-frame artifact detection (blur, lighting) with temporal analysis
    (flickering, blink rate, head motion) to produce an overall anomaly score.
    """

    def __init__(
        self,
        frame_sample_rate: int = 5,
        max_frames: int = 300,
        fps: float = 25.0,
        pose_change_threshold: float = 15.0,
        flicker_threshold: float = 0.15,
    ) -> None:
        """Initialize the video analyzer with configurable sampling parameters.

        Args:
            frame_sample_rate: Analyze every Nth frame to balance speed vs. coverage.
            max_frames: Maximum number of frames to analyze per video.
            fps: Video frame rate for temporal calculations.
            pose_change_threshold: Degrees per frame to flag head motion as abrupt.
            flicker_threshold: Pixel variance threshold for flickering detection.
        """
        self.frame_sample_rate = frame_sample_rate
        self.max_frames = max_frames
        self.face_detector = FaceDetector()
        self.temporal_analyzer = TemporalAnalyzer(
            fps=fps,
            pose_change_threshold=pose_change_threshold,
            flicker_threshold=flicker_threshold,
        )

    def analyze(
        self,
        video_path: str,
        audio_energy: Optional[np.ndarray] = None,
    ) -> tuple[float, list[str], list[LandmarkData], list[np.ndarray]]:
        """Analyze a video file and return an anomaly score with explanatory flags.

        Args:
            video_path: Path to the video file.
            audio_energy: Optional per-frame audio energy envelope for lip-sync check.

        Returns:
            Tuple of (score, flags, landmark_sequence, frame_sequence).
            score is in [0.0, 1.0] where 1.0 = highly likely deepfake.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        try:
            return self._process_video(cap, audio_energy)
        finally:
            cap.release()
            self.face_detector.close()

    def _process_video(
        self,
        cap: cv2.VideoCapture,
        audio_energy: Optional[np.ndarray],
    ) -> tuple[float, list[str], list[LandmarkData], list[np.ndarray]]:
        """Internal frame-by-frame processing loop."""
        landmark_sequence: list[LandmarkData] = []
        frame_sequence: list[np.ndarray] = []
        per_frame_scores: list[float] = []
        flags: set[str] = set()

        frame_idx = 0
        analyzed_count = 0

        while analyzed_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_sample_rate != 0:
                frame_idx += 1
                continue

            landmark_data = self.face_detector.process_frame(frame, frame_idx)
            landmark_sequence.append(landmark_data)
            frame_sequence.append(frame)

            frame_score, frame_flags = self._score_frame(frame, landmark_data)
            per_frame_scores.append(frame_score)
            flags.update(frame_flags)

            analyzed_count += 1
            frame_idx += 1

        logger.info("Analyzed %d frames from video.", analyzed_count)

        if analyzed_count == 0:
            return 0.0, [], [], []

        # Lip-sync check using audio energy if available
        if audio_energy is not None and len(audio_energy) > 0:
            lipsync_score, lipsync_flags = self._check_lip_sync(
                landmark_sequence, audio_energy
            )
            per_frame_scores.append(lipsync_score)
            flags.update(lipsync_flags)

        # Temporal analysis
        temporal_score, temporal_flags = self.temporal_analyzer.analyze(
            landmark_sequence, frame_sequence
        )
        per_frame_scores.append(temporal_score)
        flags.update(temporal_flags)

        combined = float(np.mean(per_frame_scores)) if per_frame_scores else 0.0
        return min(combined, 1.0), list(flags), landmark_sequence, frame_sequence

    def _score_frame(
        self, frame: np.ndarray, landmark_data: LandmarkData
    ) -> tuple[float, list[str]]:
        """Compute per-frame deepfake indicators from visual artifacts."""
        scores: list[float] = []
        flags: list[str] = []

        blur_score, blur_flags = self._check_edge_blurring(frame, landmark_data)
        scores.append(blur_score)
        flags.extend(blur_flags)

        lighting_score, lighting_flags = self._check_lighting_consistency(frame, landmark_data)
        scores.append(lighting_score)
        flags.extend(lighting_flags)

        return float(np.mean(scores)) if scores else 0.0, flags

    def _check_edge_blurring(
        self, frame: np.ndarray, landmark_data: LandmarkData
    ) -> tuple[float, list[str]]:
        """Detect unnatural blurring around the face boundary using Laplacian variance.

        Deepfake face-swaps often have a blurry boundary where the synthetic face
        blends with the original background, producing lower Laplacian variance
        in that region compared to the rest of the frame.
        """
        if landmark_data.face_bbox is None:
            return 0.0, []

        x, y, w, h = landmark_data.face_bbox
        if w < 10 or h < 10:
            return 0.0, []

        # Expand the bbox slightly to capture the boundary region
        margin = max(5, int(min(w, h) * 0.1))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        face_region = frame[y : y + h, x : x + w]
        border_region = _extract_border(frame, x1, y1, x2, y2, x, y, w, h)

        if face_region.size == 0 or border_region.size == 0:
            return 0.0, []

        face_sharpness = _laplacian_variance(face_region)
        border_sharpness = _laplacian_variance(border_region)

        # Flag if the border is significantly blurrier than the face interior
        if face_sharpness > 1.0 and border_sharpness < face_sharpness * 0.3:
            return 0.6, ["blurring_artifacts"]

        return 0.0, []

    def _check_lighting_consistency(
        self, frame: np.ndarray, landmark_data: LandmarkData
    ) -> tuple[float, list[str]]:
        """Detect lighting inconsistencies between the face and surrounding scene.

        Deepfakes composited onto real footage often have mismatched lighting;
        gradient histogram comparison catches color/intensity discrepancies.
        """
        if landmark_data.face_bbox is None:
            return 0.0, []

        x, y, w, h = landmark_data.face_bbox
        if w < 10 or h < 10:
            return 0.0, []

        face_region = frame[y : y + h, x : x + w]
        if face_region.size == 0:
            return 0.0, []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        face_gray = gray[y : y + h, x : x + w]
        bg_gray = gray.copy()
        bg_gray[y : y + h, x : x + w] = np.nan  # mask face

        face_mean = float(np.mean(face_gray))
        bg_vals = gray[~np.isnan(bg_gray)]
        if bg_vals.size == 0:
            return 0.0, []

        bg_mean = float(np.mean(bg_vals))

        # Large discrepancy between face brightness and scene brightness
        if abs(face_mean - bg_mean) > 60:
            return 0.4, ["lighting_inconsistency"]

        return 0.0, []

    def _check_lip_sync(
        self,
        landmark_sequence: list[LandmarkData],
        audio_energy: np.ndarray,
    ) -> tuple[float, list[str]]:
        """Measure correlation between lip movement and audio energy.

        Deepfakes often have lip movements that don't match the audio timing.
        We compute the cross-correlation between lip-distance and audio energy
        and flag low correlation as a lip-sync mismatch.
        """
        lip_distances = [ld.lip_distance for ld in landmark_sequence]
        if len(lip_distances) < 4:
            return 0.0, []

        n = min(len(lip_distances), len(audio_energy))
        lip_arr = np.array(lip_distances[:n])
        audio_arr = np.array(audio_energy[:n], dtype=np.float32)

        # Normalize both signals
        lip_arr = (lip_arr - lip_arr.mean()) / (lip_arr.std() + 1e-8)
        audio_arr = (audio_arr - audio_arr.mean()) / (audio_arr.std() + 1e-8)

        correlation = float(np.corrcoef(lip_arr, audio_arr)[0, 1])
        logger.debug("Lip-sync correlation: %.3f", correlation)

        # Low or negative correlation indicates mismatch
        if correlation < 0.3:
            mismatch_score = (0.3 - correlation) / 1.3  # scale to ~[0, 0.23]
            return min(mismatch_score * 3.0, 1.0), ["lip_sync_mismatch"]

        return 0.0, []


def _laplacian_variance(region: np.ndarray) -> float:
    """Compute the variance of the Laplacian as a measure of image sharpness."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _extract_border(
    frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, fx: int, fy: int, fw: int, fh: int
) -> np.ndarray:
    """Extract the border region around a face bounding box (excluding the face interior)."""
    outer = frame[y1:y2, x1:x2].copy()
    # Black out the face interior to isolate just the border pixels
    rel_x = fx - x1
    rel_y = fy - y1
    outer[rel_y : rel_y + fh, rel_x : rel_x + fw] = 0
    # Return only non-zero pixels
    mask = np.any(outer != 0, axis=-1)
    return outer[mask]
