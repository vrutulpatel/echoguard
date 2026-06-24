"""Face and landmark detection using MediaPipe FaceMesh.

Supports both the legacy MediaPipe API (< 0.10.14, mp.solutions.face_mesh)
and the new Tasks API (>= 0.10.14, mp.tasks.vision.FaceLandmarker). Falls
back to empty data gracefully when neither is available.

The Tasks API requires a face_landmarker.task model file (~3 MB), which is
downloaded automatically to models/ on first use if not present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Landmark indices for left and right eyes (MediaPipe FaceMesh convention)
_LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Landmark indices for lips (outer boundary)
_LIP_UPPER_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
_LIP_LOWER_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# MediaPipe face landmarker model — downloaded automatically on first use
_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_FACE_LANDMARKER_PATH = Path("models/face_landmarker.task")


@dataclass
class LandmarkData:
    """Structured landmark data extracted from a single video frame."""

    frame_idx: int
    landmarks: Optional[np.ndarray]          # shape (468, 3) — x, y, z normalized
    eye_aspect_ratio: float = 0.0            # 0.0 = fully closed, ~0.3 = open
    lip_distance: float = 0.0               # normalized distance between upper/lower lip
    is_blink: bool = False
    head_pose_angles: tuple[float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0)
    )  # pitch, yaw, roll in degrees
    detection_confidence: float = 0.0
    face_bbox: Optional[tuple[int, int, int, int]] = None  # x, y, w, h in pixels


class FaceDetector:
    """Detects faces and extracts 468 MediaPipe FaceMesh landmarks per frame.

    Tries the legacy mp.solutions API first, then the Tasks API (>= 0.10.14),
    and falls back gracefully to empty data if neither is available.
    """

    def __init__(
        self, max_faces: int = 1, min_detection_confidence: float = 0.5
    ) -> None:
        """Initialize FaceMesh, trying both MediaPipe API generations.

        Args:
            max_faces: Maximum number of faces to track simultaneously.
            min_detection_confidence: Minimum confidence threshold for detection.
        """
        self._max_faces = max_faces
        self._min_confidence = min_detection_confidence
        self._face_mesh = None          # legacy API handle
        self._face_landmarker = None    # tasks API handle
        self._api: Optional[str] = None  # 'legacy', 'tasks', or None
        self._initialized = False
        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        """Attempt initialization with legacy API, falling back to Tasks API."""
        try:
            import mediapipe as mp  # noqa: PLC0415

            # Legacy API: available in mediapipe < 0.10.14
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=self._max_faces,
                    refine_landmarks=True,
                    min_detection_confidence=self._min_confidence,
                    min_tracking_confidence=0.5,
                )
                self._api = "legacy"
                self._initialized = True
                logger.debug("MediaPipe FaceMesh (legacy API) initialized.")
                return

            # Tasks API: available in mediapipe >= 0.10.14
            logger.debug("Legacy mp.solutions unavailable — trying Tasks API.")
            self._init_tasks_api()

        except ImportError:
            logger.warning(
                "MediaPipe not installed. Face landmark detection disabled. "
                "Install with: pip install mediapipe>=0.10.0"
            )
        except Exception as exc:
            logger.warning(
                "MediaPipe initialization failed: %s — face detection disabled.", exc
            )

    def _init_tasks_api(self) -> None:
        """Initialize the new MediaPipe Tasks API (mediapipe >= 0.10.14).

        Downloads the face_landmarker.task model (~3 MB) automatically if absent.
        """
        try:
            from mediapipe.tasks import python as mp_python  # noqa: PLC0415
            from mediapipe.tasks.python.vision import (  # noqa: PLC0415
                FaceLandmarker,
                FaceLandmarkerOptions,
            )

            if not _FACE_LANDMARKER_PATH.exists():
                self._download_face_landmarker(_FACE_LANDMARKER_PATH)

            base_opts = mp_python.BaseOptions(
                model_asset_path=str(_FACE_LANDMARKER_PATH)
            )
            opts = FaceLandmarkerOptions(
                base_options=base_opts,
                num_faces=self._max_faces,
                min_face_detection_confidence=self._min_confidence,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._face_landmarker = FaceLandmarker.create_from_options(opts)
            self._api = "tasks"
            self._initialized = True
            logger.debug("MediaPipe FaceLandmarker (Tasks API) initialized.")

        except Exception as exc:
            logger.warning(
                "MediaPipe Tasks API init failed: %s — face detection disabled.", exc
            )

    @staticmethod
    def _download_face_landmarker(model_path: Path) -> None:
        """Download the official MediaPipe face landmarker model file."""
        import urllib.request  # noqa: PLC0415

        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading MediaPipe face landmarker model (~3 MB) to %s …", model_path
        )
        urllib.request.urlretrieve(_FACE_LANDMARKER_URL, str(model_path))
        logger.info("Face landmarker model downloaded.")

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> LandmarkData:
        """Extract landmarks and compute facial metrics from a single BGR frame.

        Args:
            frame: BGR image array from OpenCV (H, W, 3).
            frame_idx: Frame index in the video sequence.

        Returns:
            LandmarkData with landmarks and derived metrics, or empty data if
            no face is detected or MediaPipe is unavailable.
        """
        if not self._initialized:
            return LandmarkData(frame_idx=frame_idx, landmarks=None)

        if self._api == "legacy":
            return self._process_legacy(frame, frame_idx)
        elif self._api == "tasks":
            return self._process_tasks(frame, frame_idx)
        return LandmarkData(frame_idx=frame_idx, landmarks=None)

    def _process_legacy(self, frame: np.ndarray, frame_idx: int) -> LandmarkData:
        """Process frame using the legacy mp.solutions.face_mesh API."""
        import cv2  # noqa: PLC0415

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return LandmarkData(frame_idx=frame_idx, landmarks=None)

        face = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        landmarks_arr = np.array(
            [(lm.x, lm.y, lm.z) for lm in face.landmark], dtype=np.float32
        )
        return self._build_landmark_data(frame_idx, landmarks_arr, w, h)

    def _process_tasks(self, frame: np.ndarray, frame_idx: int) -> LandmarkData:
        """Process frame using the new MediaPipe Tasks API."""
        import cv2  # noqa: PLC0415
        import mediapipe as mp  # noqa: PLC0415

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            return LandmarkData(frame_idx=frame_idx, landmarks=None)

        face = results.face_landmarks[0]
        h, w = frame.shape[:2]
        landmarks_arr = np.array(
            [(lm.x, lm.y, lm.z) for lm in face], dtype=np.float32
        )
        return self._build_landmark_data(frame_idx, landmarks_arr, w, h)

    def _build_landmark_data(
        self, frame_idx: int, landmarks_arr: np.ndarray, img_w: int, img_h: int
    ) -> LandmarkData:
        """Compute derived metrics from a raw landmark array and return LandmarkData."""
        ear = self._eye_aspect_ratio(landmarks_arr)
        lip_dist = self._lip_distance(landmarks_arr)
        is_blink = ear < 0.2
        pose = self._estimate_head_pose(landmarks_arr)
        bbox = self._face_bounding_box(landmarks_arr, img_w, img_h)

        return LandmarkData(
            frame_idx=frame_idx,
            landmarks=landmarks_arr,
            eye_aspect_ratio=float(ear),
            lip_distance=float(lip_dist),
            is_blink=is_blink,
            head_pose_angles=pose,
            detection_confidence=1.0,
            face_bbox=bbox,
        )

    def _eye_aspect_ratio(self, lm: np.ndarray) -> float:
        """Compute mean Eye Aspect Ratio (EAR) across both eyes."""
        def _ear(indices: list[int]) -> float:
            p = lm[indices, :2]
            v1 = np.linalg.norm(p[1] - p[5])
            v2 = np.linalg.norm(p[2] - p[4])
            h = np.linalg.norm(p[0] - p[3])
            return float((v1 + v2) / (2.0 * h)) if h > 1e-6 else 0.0

        return (_ear(_LEFT_EYE_INDICES) + _ear(_RIGHT_EYE_INDICES)) / 2.0

    def _lip_distance(self, lm: np.ndarray) -> float:
        """Compute normalized distance between upper and lower lip centers."""
        upper = lm[_LIP_UPPER_INDICES, :2].mean(axis=0)
        lower = lm[_LIP_LOWER_INDICES, :2].mean(axis=0)
        return float(np.linalg.norm(upper - lower))

    def _estimate_head_pose(self, lm: np.ndarray) -> tuple[float, float, float]:
        """Estimate head pose (pitch, yaw, roll) in degrees from key landmarks."""
        nose = lm[1, :2]
        chin = lm[152, :2]
        left_eye = lm[33, :2]
        right_eye = lm[263, :2]

        eye_vec = right_eye - left_eye
        roll = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))

        eye_mid = (left_eye + right_eye) / 2.0
        eye_span = np.linalg.norm(right_eye - left_eye) + 1e-6
        yaw = float((nose[0] - eye_mid[0]) / eye_span * 90.0)

        face_h = np.linalg.norm(chin - eye_mid) + 1e-6
        pitch = float((nose[1] - eye_mid[1]) / face_h * 90.0)

        return (pitch, yaw, roll)

    def _face_bounding_box(
        self, lm: np.ndarray, img_w: int, img_h: int
    ) -> tuple[int, int, int, int]:
        """Compute tight bounding box around the face in pixel coordinates."""
        x_min = int((lm[:, 0] * img_w).min())
        x_max = int((lm[:, 0] * img_w).max())
        y_min = int((lm[:, 1] * img_h).min())
        y_max = int((lm[:, 1] * img_h).max())
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass
        if self._face_landmarker is not None:
            try:
                self._face_landmarker.close()
            except Exception:
                pass
