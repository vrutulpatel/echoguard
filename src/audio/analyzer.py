"""Voice clone and audio deepfake detection module.

Analyzes audio files for synthetic speech artifacts: overly flat pitch,
missing room noise, clipping artifacts, and unnatural silence patterns.
Optionally uses the audio classification model for higher accuracy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.audio.spectrogram import AudioFeatures, SpectrogramExtractor

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Detects voice clone and synthetic speech artifacts in audio signals.

    Heuristic checks are designed to work even without pretrained model weights,
    making the pipeline runnable out-of-the-box on any audio input.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        pitch_flatness_threshold: float = 0.02,
        silence_max_ms: float = 2000.0,
        clipping_threshold: float = 0.98,
    ) -> None:
        """Initialize the audio analyzer.

        Args:
            sample_rate: Target sample rate for audio loading.
            pitch_flatness_threshold: Std dev of pitch in semitones below which
                the voice is flagged as suspiciously monotone.
            silence_max_ms: Contiguous silence longer than this (ms) is flagged.
            clipping_threshold: Absolute sample amplitude above this is clipping.
        """
        self.pitch_flatness_threshold = pitch_flatness_threshold
        self.silence_max_ms = silence_max_ms
        self.clipping_threshold = clipping_threshold
        self.extractor = SpectrogramExtractor(sample_rate=sample_rate)
        self._sample_rate = sample_rate

    def analyze(self, audio_path: str) -> tuple[float, list[str], AudioFeatures]:
        """Analyze an audio file and return a clone suspicion score.

        Args:
            audio_path: Path to the audio file (WAV, MP3, FLAC, etc.).

        Returns:
            Tuple of (score: float 0.0–1.0, flags: list[str], features: AudioFeatures).
            score near 1.0 indicates likely voice clone.

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        features = self.extractor.extract_from_file(str(path))
        return self._analyze_features(features)

    def analyze_array(
        self, y: np.ndarray, sr: int
    ) -> tuple[float, list[str], AudioFeatures]:
        """Analyze a raw audio array (useful for streaming or synthetic test data).

        Args:
            y: Audio waveform as float32 array in [-1, 1].
            sr: Sample rate of the audio.

        Returns:
            Tuple of (score, flags, features).
        """
        features = self.extractor.extract_from_array(y, sr)
        return self._analyze_features(features)

    def _analyze_features(
        self, features: AudioFeatures
    ) -> tuple[float, list[str], AudioFeatures]:
        """Run all heuristic checks on extracted audio features."""
        scores: list[float] = []
        flags: list[str] = []

        pitch_score, pitch_flags = self._check_pitch_flatness(features)
        scores.append(pitch_score)
        flags.extend(pitch_flags)

        silence_score, silence_flags = self._check_silence_patterns(features)
        scores.append(silence_score)
        flags.extend(silence_flags)

        clip_score, clip_flags = self._check_clipping(features)
        scores.append(clip_score)
        flags.extend(clip_flags)

        noise_score, noise_flags = self._check_room_noise(features)
        scores.append(noise_score)
        flags.extend(noise_flags)

        combined = float(np.mean(scores)) if scores else 0.0
        return min(combined, 1.0), flags, features

    def _check_pitch_flatness(
        self, features: AudioFeatures
    ) -> tuple[float, list[str]]:
        """Flag audio where the fundamental frequency is suspiciously monotone.

        AI voice synthesis often produces pitch trajectories that are too smooth
        or stay in a narrow range compared to natural speech.
        """
        if features.f0 is None:
            logger.debug("Pitch data unavailable — skipping flatness check.")
            return 0.0, []

        valid_f0 = features.f0[~np.isnan(features.f0)]
        if len(valid_f0) < 10:
            return 0.0, []

        # Convert Hz to semitones for perceptually meaningful variance
        semitones = 12.0 * np.log2(valid_f0 / 440.0 + 1e-8)
        std_semitones = float(np.std(semitones))
        logger.debug("Pitch std: %.4f semitones", std_semitones)

        if std_semitones < self.pitch_flatness_threshold:
            severity = (self.pitch_flatness_threshold - std_semitones) / self.pitch_flatness_threshold
            return min(severity, 1.0), ["flat_pitch"]

        return 0.0, []

    def _check_silence_patterns(
        self, features: AudioFeatures
    ) -> tuple[float, list[str]]:
        """Detect unnaturally long or abrupt silence segments.

        Natural speech has brief pauses (< 500ms typically); overly long silences
        can indicate spliced or synthetic audio.
        """
        rms = features.rms_energy
        if rms.size == 0:
            return 0.0, []

        silence_mask = rms < 0.01  # threshold below which a frame is considered silent
        hop_ms = (512 / self._sample_rate) * 1000  # ms per RMS frame

        max_run = _max_consecutive_true(silence_mask)
        max_silence_ms = max_run * hop_ms

        logger.debug("Max contiguous silence: %.1f ms", max_silence_ms)

        if max_silence_ms > self.silence_max_ms:
            severity = min((max_silence_ms - self.silence_max_ms) / self.silence_max_ms, 1.0)
            return severity * 0.5, ["unnatural_silence"]  # moderate weight

        return 0.0, []

    def _check_clipping(
        self, features: AudioFeatures
    ) -> tuple[float, list[str]]:
        """Detect amplitude clipping artifacts from over-processed or synthetic audio."""
        mel = features.mel_spectrogram
        if mel.size == 0:
            return 0.0, []

        # High RMS energy combined with very flat mel distribution indicates compression
        rms = features.rms_energy
        if rms.size == 0:
            return 0.0, []

        clipped_ratio = float(np.mean(rms > self.clipping_threshold))
        logger.debug("Clipped frame ratio: %.4f", clipped_ratio)

        if clipped_ratio > 0.05:
            return min(clipped_ratio * 2.0, 1.0), ["clipping_artifacts"]

        return 0.0, []

    def _check_room_noise(self, features: AudioFeatures) -> tuple[float, list[str]]:
        """Detect suspiciously clean audio lacking natural room/background noise.

        Real recordings always contain some ambient noise floor; AI-generated
        speech is often synthesized in a "perfect" anechoic environment.
        Low spectral flatness in the mel spectrogram during silent segments
        indicates an unnaturally clean noise floor.
        """
        mel_db = features.mel_spectrogram  # (n_mels, time_frames)
        rms = features.rms_energy

        if mel_db.size == 0 or rms.size == 0:
            return 0.0, []

        # Find frames with very low energy (nominally silent)
        n_frames = min(mel_db.shape[1], rms.size)
        silent_frames = rms[:n_frames] < 0.005

        if silent_frames.sum() < 5:
            return 0.0, []

        # In natural recordings, silent frames still have visible noise energy
        silent_mel = mel_db[:, :n_frames][:, silent_frames]
        noise_floor = float(np.mean(silent_mel))

        # Very low noise floor (well below -50 dB) suggests synthetic audio
        if noise_floor < -70.0:
            severity = min((-70.0 - noise_floor) / 30.0, 1.0)
            return severity * 0.6, ["missing_room_noise"]

        return 0.0, []

    def get_energy_envelope(self, audio_path: str) -> np.ndarray:
        """Extract per-frame RMS energy envelope (used for lip-sync cross-check).

        Args:
            audio_path: Path to the audio file.

        Returns:
            Float32 array of per-frame RMS energy values.
        """
        features = self.extractor.extract_from_file(audio_path)
        return features.rms_energy


def _max_consecutive_true(mask: np.ndarray) -> int:
    """Return the length of the longest consecutive True run in a boolean array."""
    max_run = 0
    current_run = 0
    for val in mask:
        if val:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run
