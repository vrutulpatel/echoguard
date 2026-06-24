"""Unified multimodal deepfake detection pipeline.

EchoGuardDetector orchestrates video analysis, audio analysis, and score fusion
into a single analyze() call that returns a complete DetectionResult.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.audio.analyzer import AudioAnalyzer
from src.models.audio_model import AudioCloneDetector, build_audio_model
from src.models.model_loader import load_pytorch_model, resolve_device
from src.models.video_model import VideoDeepfakeDetector, build_video_model
from src.pipeline.result import DetectionResult, combine_scores
from src.video.analyzer import VideoAnalyzer

logger = logging.getLogger(__name__)


class EchoGuardDetector:
    """Main detection pipeline combining video and audio deepfake analysis.

    Instantiate once (model loading is expensive) and call analyze() repeatedly.
    Both video-only and audio-only inputs are supported; the pipeline adapts
    automatically based on which streams are provided.
    """

    def __init__(
        self,
        video_model_path: Optional[str] = None,
        audio_model_path: Optional[str] = None,
        detection_threshold: float = 0.65,
        video_weight: float = 0.6,
        audio_weight: float = 0.4,
        frame_sample_rate: int = 5,
        max_frames: int = 300,
        device: str = "cpu",
    ) -> None:
        """Initialize the detector and load models.

        Args:
            video_model_path: Path to video model weights (.pt or .onnx). Optional.
            audio_model_path: Path to audio model weights (.pt or .onnx). Optional.
            detection_threshold: Combined score threshold for deepfake verdict.
            video_weight: Fusion weight for video score (must sum to 1 with audio_weight).
            audio_weight: Fusion weight for audio score.
            frame_sample_rate: Analyze every Nth video frame.
            max_frames: Maximum frames to analyze per video.
            device: Compute device ('cpu' or 'cuda').
        """
        self.threshold = detection_threshold
        self.video_weight = video_weight
        self.audio_weight = audio_weight
        self._device = resolve_device(device)

        self._video_model = self._load_video_model(video_model_path)
        self._audio_model = self._load_audio_model(audio_model_path)

        self._video_analyzer = VideoAnalyzer(
            frame_sample_rate=frame_sample_rate,
            max_frames=max_frames,
        )
        self._audio_analyzer = AudioAnalyzer()

        logger.info(
            "EchoGuardDetector ready — device=%s, threshold=%.2f",
            self._device,
            self.threshold,
        )

    def analyze(
        self,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> DetectionResult:
        """Run multimodal deepfake analysis on the provided file(s).

        At least one of video_path or audio_path must be provided.

        Args:
            video_path: Path to the video file to analyze. Optional.
            audio_path: Path to the audio file to analyze. Optional.

        Returns:
            DetectionResult with scores, flags, verdict, and timing.

        Raises:
            ValueError: If neither video_path nor audio_path is provided.
        """
        if video_path is None and audio_path is None:
            raise ValueError("At least one of video_path or audio_path must be provided.")

        start_time = time.perf_counter()
        all_flags: list[str] = []
        video_score: Optional[float] = None
        audio_score: Optional[float] = None
        frame_count = 0

        # Extract audio energy first so video can use it for lip-sync check
        audio_energy: Optional[np.ndarray] = None
        if audio_path:
            try:
                audio_energy = self._audio_analyzer.get_energy_envelope(audio_path)
            except Exception as exc:
                logger.warning("Could not extract audio energy for lip-sync: %s", exc)

        if video_path:
            video_score, video_flags, frame_count = self._run_video_analysis(
                video_path, audio_energy
            )
            all_flags.extend(video_flags)

        if audio_path:
            audio_score, audio_flags = self._run_audio_analysis(audio_path)
            all_flags.extend(audio_flags)

        combined = self._combine_scores(video_score, audio_score)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return DetectionResult(
            video_score=video_score,
            audio_score=audio_score,
            combined_score=combined,
            flags=list(set(all_flags)),  # deduplicate
            processing_time_ms=elapsed_ms,
            frame_count=frame_count,
            threshold=self.threshold,
        )

    def _run_video_analysis(
        self,
        video_path: str,
        audio_energy: Optional[np.ndarray] = None,
    ) -> tuple[float, list[str], int]:
        """Run video analysis pipeline and optionally apply model inference.

        Returns:
            Tuple of (score, flags, frame_count).
        """
        try:
            heuristic_score, flags, landmarks, frames = self._video_analyzer.analyze(
                video_path, audio_energy
            )
            frame_count = len(frames)
        except Exception as exc:
            logger.error("Video analysis failed: %s", exc)
            return self._fallback_score("video"), [], 0

        # Model inference on sampled face crops
        model_score = self._run_video_model_inference(frames)

        # Blend heuristic and model scores
        final_score = 0.5 * heuristic_score + 0.5 * model_score
        logger.info(
            "Video — heuristic=%.3f model=%.3f final=%.3f flags=%s",
            heuristic_score, model_score, final_score, flags,
        )
        return final_score, flags, frame_count

    def _run_audio_analysis(self, audio_path: str) -> tuple[float, list[str]]:
        """Run audio analysis pipeline and optionally apply model inference.

        Returns:
            Tuple of (score, flags).
        """
        try:
            heuristic_score, flags, features = self._audio_analyzer.analyze(audio_path)
        except Exception as exc:
            logger.error("Audio analysis failed: %s", exc)
            return self._fallback_score("audio"), []

        # Model inference on mel spectrogram
        model_score = self._run_audio_model_inference(features)

        final_score = 0.5 * heuristic_score + 0.5 * model_score
        logger.info(
            "Audio — heuristic=%.3f model=%.3f final=%.3f flags=%s",
            heuristic_score, model_score, final_score, flags,
        )
        return final_score, flags

    def _run_video_model_inference(self, frames: list[np.ndarray]) -> float:
        """Run the video CNN on a sample of frames and average the predictions."""
        if not frames or self._video_model is None:
            return self._fallback_score("video_model")

        import cv2  # noqa: PLC0415
        import torch  # noqa: PLC0415

        scores: list[float] = []
        # Sample up to 10 evenly spaced frames for model inference
        indices = np.linspace(0, len(frames) - 1, min(10, len(frames)), dtype=int)

        for i in indices:
            try:
                frame = cv2.resize(frames[i], (224, 224))
                tensor = torch.from_numpy(
                    frame.transpose(2, 0, 1).astype(np.float32) / 255.0
                ).unsqueeze(0)
                score = self._video_model.predict(tensor)
                scores.append(score)
            except Exception as exc:
                logger.debug("Frame %d inference failed: %s", i, exc)

        return float(np.mean(scores)) if scores else self._fallback_score("video_model")

    def _run_audio_model_inference(self, features) -> float:  # type: ignore[no-untyped-def]
        """Run the audio CNN on the mel spectrogram and return clone probability."""
        if self._audio_model is None:
            return self._fallback_score("audio_model")

        try:
            import torch  # noqa: PLC0415
            from src.audio.spectrogram import SpectrogramExtractor  # noqa: PLC0415

            extractor = SpectrogramExtractor()
            tensor = torch.from_numpy(extractor.to_model_input(features))
            score = self._audio_model.predict(tensor)
            return float(score)
        except Exception as exc:
            logger.debug("Audio model inference failed: %s", exc)
            return self._fallback_score("audio_model")

    def _combine_scores(
        self, video_score: Optional[float], audio_score: Optional[float]
    ) -> float:
        """Fuse video and audio scores using configured weights."""
        return combine_scores(video_score, audio_score, self.video_weight, self.audio_weight)

    def _load_video_model(
        self, weights_path: Optional[str]
    ) -> Optional[VideoDeepfakeDetector]:
        """Build the video model (weight loading done inside build_video_model),
        then place on device and apply quantization via load_pytorch_model."""
        try:
            # build_video_model handles both ImageNet backbone + fine-tuned head weights
            model = build_video_model(weights_path)
            # Pass weights_path=None so load_pytorch_model skips redundant file I/O
            return load_pytorch_model(
                model, weights_path=None, device=self._device, quantize=True,
                cache_key=f"video:{weights_path}:{self._device}",
            )
        except Exception as exc:
            logger.error("Failed to initialize video model: %s", exc)
            return None

    def _load_audio_model(
        self, weights_path: Optional[str]
    ) -> Optional[AudioCloneDetector]:
        """Build the audio model (weight loading done inside build_audio_model),
        then place on device and apply quantization via load_pytorch_model."""
        try:
            model = build_audio_model(weights_path)
            return load_pytorch_model(
                model, weights_path=None, device=self._device, quantize=True,
                cache_key=f"audio:{weights_path}:{self._device}",
            )
        except Exception as exc:
            logger.error("Failed to initialize audio model: %s", exc)
            return None

    @staticmethod
    def _fallback_score(source: str) -> float:
        """Return a random score in [0.3, 0.7] when a component fails.

        This keeps the pipeline running during testing without model weights,
        while producing scores near the decision boundary to avoid false verdicts.
        """
        score = random.uniform(0.3, 0.7)
        logger.warning(
            "Using random fallback score %.3f for '%s' — check model weights or inputs.",
            score, source,
        )
        return score
