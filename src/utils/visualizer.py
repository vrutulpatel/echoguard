"""Frame annotation and video export utilities.

Provides functions to draw detection results on video frames:
colored bounding boxes, score overlays, flag lists, and an EchoGuard watermark.
Also exports annotated frames as a new video file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.pipeline.result import DetectionResult
from src.video.face_detector import LandmarkData

logger = logging.getLogger(__name__)

# Visual style constants
_COLOR_DEEPFAKE = (0, 0, 220)    # Red in BGR
_COLOR_REAL = (0, 200, 0)        # Green in BGR
_COLOR_WATERMARK = (180, 180, 180)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WATERMARK_TEXT = "EchoGuard"


def annotate_frame(
    frame: np.ndarray,
    result: DetectionResult,
    landmark_data: Optional[LandmarkData] = None,
) -> np.ndarray:
    """Draw detection results overlaid on a single video frame.

    Draws:
    - A colored bounding box around the face (red=deepfake, green=real)
    - Verdict text and combined score in the top-left corner
    - Flags listed below the verdict
    - A small EchoGuard watermark in the bottom-right corner

    Args:
        frame: BGR image array from OpenCV (H, W, 3). Not modified in-place.
        result: DetectionResult from the detection pipeline.
        landmark_data: Optional landmark data for drawing the face bounding box.

    Returns:
        Annotated copy of the frame.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    color = _COLOR_DEEPFAKE if result.is_deepfake else _COLOR_REAL

    # Draw face bounding box if available
    if landmark_data is not None and landmark_data.face_bbox is not None:
        x, y, bw, bh = landmark_data.face_bbox
        thickness = 2
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, thickness)

    # Verdict and score overlay (semi-transparent background)
    _draw_text_with_background(
        annotated,
        text=f"{result.verdict_emoji} {result.verdict}",
        pos=(10, 28),
        color=color,
        font_scale=0.65,
        thickness=2,
    )
    _draw_text_with_background(
        annotated,
        text=f"Score: {result.combined_score:.2f}  Conf: {result.confidence_label}",
        pos=(10, 52),
        color=(220, 220, 220),
        font_scale=0.50,
        thickness=1,
    )

    # Flags
    for i, flag in enumerate(result.flags[:5]):  # cap at 5 to avoid overflow
        _draw_text_with_background(
            annotated,
            text=f"• {flag}",
            pos=(10, 76 + i * 20),
            color=(200, 160, 0),
            font_scale=0.42,
            thickness=1,
        )

    # Watermark in bottom-right
    text_size, _ = cv2.getTextSize(_WATERMARK_TEXT, _FONT, 0.4, 1)
    wm_x = w - text_size[0] - 8
    wm_y = h - 8
    cv2.putText(annotated, _WATERMARK_TEXT, (wm_x, wm_y), _FONT, 0.4, _COLOR_WATERMARK, 1)

    return annotated


def export_annotated_video(
    video_path: str,
    output_path: str,
    results_per_frame: list[tuple[DetectionResult, Optional[LandmarkData]]],
    fps: float = 25.0,
) -> None:
    """Write an annotated video with per-frame detection overlays.

    Args:
        video_path: Path to the original (unannotated) video file.
        output_path: Where to write the annotated video (e.g. output.mp4).
        results_per_frame: List of (DetectionResult, LandmarkData) pairs,
            one per analyzed frame. Frames without analysis receive no overlay.
        fps: Output video frame rate.

    Raises:
        FileNotFoundError: If the input video does not exist.
        RuntimeError: If the video writer cannot be opened.
    """
    src_path = Path(video_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, orig_fps, (orig_w, orig_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer for: {output_path}")

    result_idx = 0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if result_idx < len(results_per_frame):
                res, lm = results_per_frame[result_idx]
                frame = annotate_frame(frame, res, lm)
                result_idx += 1

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    logger.info(
        "Annotated video written to %s (%d/%d frames processed).",
        output_path, frame_idx, total_frames,
    )


def _draw_text_with_background(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Draw text on a frame with a semi-transparent dark background for readability."""
    text_size, baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    tw, th = text_size
    x, y = pos
    padding = 3

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y), _FONT, font_scale, color, thickness, cv2.LINE_AA)
