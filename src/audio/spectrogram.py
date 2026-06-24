"""Mel spectrogram and MFCC feature extraction for audio deepfake detection.

Uses Librosa to extract perceptually-motivated features from audio signals.
All features are returned as float32 tensors ready for model inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default feature extraction parameters
_SAMPLE_RATE = 22050
_N_FFT = 2048
_HOP_LENGTH = 512
_N_MELS = 128
_N_MFCC = 40
_TARGET_SHAPE = (128, 128)  # (n_mels, time_frames) for model input


@dataclass
class AudioFeatures:
    """Structured audio features extracted from a single audio clip."""

    mel_spectrogram: np.ndarray      # shape (n_mels, time_frames), dB-scaled
    mfcc: np.ndarray                  # shape (n_mfcc, time_frames)
    chroma: np.ndarray                # shape (12, time_frames)
    sample_rate: int
    duration_seconds: float
    rms_energy: np.ndarray           # shape (time_frames,) — per-frame RMS energy
    f0: Optional[np.ndarray] = None  # shape (time_frames,) — fundamental frequency


class SpectrogramExtractor:
    """Extracts mel spectrograms, MFCCs, and chroma features from audio files.

    All output tensors are normalized and resized to fixed shapes for consistent
    model input, regardless of the original audio duration.
    """

    def __init__(
        self,
        sample_rate: int = _SAMPLE_RATE,
        n_fft: int = _N_FFT,
        hop_length: int = _HOP_LENGTH,
        n_mels: int = _N_MELS,
        n_mfcc: int = _N_MFCC,
        target_shape: tuple[int, int] = _TARGET_SHAPE,
    ) -> None:
        """Initialize feature extraction parameters.

        Args:
            sample_rate: Target sample rate; audio is resampled if needed.
            n_fft: FFT window size.
            hop_length: Hop size between STFT frames.
            n_mels: Number of mel filterbank channels.
            n_mfcc: Number of MFCC coefficients to compute.
            target_shape: (height, width) to resize mel spectrogram for model input.
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.target_shape = target_shape

    def extract_from_file(self, audio_path: str) -> AudioFeatures:
        """Load an audio file and extract all features.

        Args:
            audio_path: Path to the audio file (WAV, MP3, FLAC, etc.).

        Returns:
            AudioFeatures dataclass with all extracted features.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            RuntimeError: If Librosa fails to load or process the file.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            import librosa  # noqa: PLC0415

            y, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio file '{audio_path}': {exc}") from exc

        return self.extract_from_array(y, sr)

    def extract_from_array(self, y: np.ndarray, sr: int) -> AudioFeatures:
        """Extract features from a raw audio array.

        Args:
            y: Audio waveform as float32 array in [-1, 1].
            sr: Sample rate of the audio.

        Returns:
            AudioFeatures with mel spectrogram, MFCCs, chroma, and energy.
        """
        try:
            import librosa  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Librosa is required for audio feature extraction. "
                "Install with: pip install librosa>=0.10.0"
            ) from exc

        duration = float(len(y)) / sr

        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
        )

        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

        # Attempt fundamental frequency extraction; skip if pyin is unavailable
        f0 = None
        try:
            f0_raw, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
            f0 = np.where(voiced, f0_raw, np.nan).astype(np.float32)
        except Exception:
            logger.debug("F0 extraction skipped (pyin unavailable or failed).")

        return AudioFeatures(
            mel_spectrogram=mel_db.astype(np.float32),
            mfcc=mfcc.astype(np.float32),
            chroma=chroma.astype(np.float32),
            sample_rate=sr,
            duration_seconds=duration,
            rms_energy=rms.astype(np.float32),
            f0=f0,
        )

    def to_model_input(self, features: AudioFeatures) -> np.ndarray:
        """Resize and normalize the mel spectrogram to a fixed shape for model inference.

        Args:
            features: AudioFeatures object from extract_from_file or extract_from_array.

        Returns:
            Float32 array of shape (1, 1, H, W) suitable for a PyTorch/ONNX model
            expecting batch_size=1, channels=1 (grayscale spectrogram).
        """
        import cv2  # noqa: PLC0415

        mel = features.mel_spectrogram  # (n_mels, time_frames)

        # Resize to target shape
        resized = cv2.resize(mel, (self.target_shape[1], self.target_shape[0]))

        # Normalize to [0, 1]
        mel_min, mel_max = resized.min(), resized.max()
        if mel_max - mel_min > 1e-6:
            resized = (resized - mel_min) / (mel_max - mel_min)
        else:
            resized = np.zeros_like(resized)

        return resized.astype(np.float32)[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
