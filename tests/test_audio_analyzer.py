"""Tests for src/audio/analyzer.py, spectrogram.py, and voiceprint.py.

Uses synthetic audio data (sine wave, white noise, silence) to test
without real audio files or model weights.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.audio.spectrogram import AudioFeatures, SpectrogramExtractor
from src.audio.voiceprint import VoiceprintMatcher


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

_SR = 22050

def _sine_wave(duration: float = 2.0, freq: float = 440.0, sr: int = _SR) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _white_noise(duration: float = 2.0, sr: int = _SR) -> np.ndarray:
    return np.random.uniform(-0.5, 0.5, int(sr * duration)).astype(np.float32)


def _silence(duration: float = 2.0, sr: int = _SR) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


# ──────────────────────────────────────────────
# SpectrogramExtractor tests
# ──────────────────────────────────────────────

class TestSpectrogramExtractor:
    def test_extract_from_array_returns_features(self) -> None:
        """extract_from_array should return AudioFeatures for any valid signal."""
        extractor = SpectrogramExtractor()
        y = _sine_wave()
        features = extractor.extract_from_array(y, _SR)
        assert isinstance(features, AudioFeatures)

    def test_mel_spectrogram_shape(self) -> None:
        """Mel spectrogram should have n_mels rows."""
        extractor = SpectrogramExtractor(n_mels=128)
        y = _sine_wave(duration=3.0)
        features = extractor.extract_from_array(y, _SR)
        assert features.mel_spectrogram.shape[0] == 128

    def test_mfcc_shape(self) -> None:
        """MFCC should have n_mfcc rows."""
        extractor = SpectrogramExtractor(n_mfcc=40)
        y = _sine_wave()
        features = extractor.extract_from_array(y, _SR)
        assert features.mfcc.shape[0] == 40

    def test_rms_energy_non_negative(self) -> None:
        """RMS energy values should all be non-negative."""
        extractor = SpectrogramExtractor()
        y = _sine_wave()
        features = extractor.extract_from_array(y, _SR)
        assert np.all(features.rms_energy >= 0)

    def test_to_model_input_shape(self) -> None:
        """to_model_input should return (1, 1, 128, 128) tensor."""
        extractor = SpectrogramExtractor(target_shape=(128, 128))
        y = _sine_wave()
        features = extractor.extract_from_array(y, _SR)
        tensor = extractor.to_model_input(features)
        assert tensor.shape == (1, 1, 128, 128)

    def test_to_model_input_normalized(self) -> None:
        """Model input should be in [0, 1]."""
        extractor = SpectrogramExtractor()
        y = _sine_wave()
        features = extractor.extract_from_array(y, _SR)
        tensor = extractor.to_model_input(features)
        assert float(tensor.min()) >= 0.0
        assert float(tensor.max()) <= 1.0

    def test_extract_from_file_missing_raises(self) -> None:
        extractor = SpectrogramExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract_from_file("/no/such/file.wav")


# ──────────────────────────────────────────────
# AudioAnalyzer tests
# ──────────────────────────────────────────────

class TestAudioAnalyzer:
    def test_analyze_array_returns_score(self) -> None:
        """analyze_array should return a float score in [0.0, 1.0]."""
        from src.audio.analyzer import AudioAnalyzer  # noqa: PLC0415

        analyzer = AudioAnalyzer()
        y = _sine_wave()
        score, flags, features = analyzer.analyze_array(y, _SR)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(flags, list)
        assert isinstance(features, AudioFeatures)

    def test_analyze_silence_returns_score(self) -> None:
        """Silence should not raise; score should be in valid range."""
        from src.audio.analyzer import AudioAnalyzer  # noqa: PLC0415

        analyzer = AudioAnalyzer()
        y = _silence()
        score, flags, _ = analyzer.analyze_array(y, _SR)
        assert 0.0 <= score <= 1.0

    def test_analyze_white_noise_returns_score(self) -> None:
        from src.audio.analyzer import AudioAnalyzer  # noqa: PLC0415

        analyzer = AudioAnalyzer()
        y = _white_noise()
        score, flags, _ = analyzer.analyze_array(y, _SR)
        assert 0.0 <= score <= 1.0

    def test_analyze_missing_file_raises(self) -> None:
        from src.audio.analyzer import AudioAnalyzer  # noqa: PLC0415

        analyzer = AudioAnalyzer()
        with pytest.raises(FileNotFoundError):
            analyzer.analyze("/does/not/exist.wav")

    def test_flat_pitch_not_flagged_for_silence(self) -> None:
        """Silence has no voiced frames, so flat_pitch should not be flagged."""
        from src.audio.analyzer import AudioAnalyzer  # noqa: PLC0415

        analyzer = AudioAnalyzer()
        y = _silence()
        features = SpectrogramExtractor().extract_from_array(y, _SR)
        _, flags = analyzer._check_pitch_flatness(features)
        assert "flat_pitch" not in flags


# ──────────────────────────────────────────────
# VoiceprintMatcher tests
# ──────────────────────────────────────────────

class TestVoiceprintMatcher:
    def test_compute_embedding_from_features(self) -> None:
        """Voiceprint embedding should be a non-zero float32 vector."""
        extractor = SpectrogramExtractor()
        matcher = VoiceprintMatcher(extractor)
        y = _sine_wave()
        features = extractor.extract_from_array(y, _SR)
        emb = matcher.compute_embedding(features)
        assert emb.dtype == np.float32
        assert emb.ndim == 1
        assert len(emb) > 0
        assert not np.all(emb == 0)

    def test_embedding_is_unit_normalized(self) -> None:
        """Embedding should be L2-normalized (norm ~1.0)."""
        extractor = SpectrogramExtractor()
        matcher = VoiceprintMatcher(extractor)
        y = _sine_wave()
        features = extractor.extract_from_array(y, _SR)
        emb = matcher.compute_embedding(features)
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-4

    def test_same_signal_high_similarity(self) -> None:
        """The same signal should produce near-perfect similarity with itself."""
        extractor = SpectrogramExtractor()
        matcher = VoiceprintMatcher(extractor)
        y = _sine_wave()
        features = extractor.extract_from_array(y, _SR)
        emb = matcher.compute_embedding(features)
        sim, is_match = matcher.verify(emb, emb)
        assert sim > 0.99

    def test_different_signals_lower_similarity(self) -> None:
        """A sine wave and white noise should have lower similarity than same vs. same."""
        extractor = SpectrogramExtractor()
        matcher = VoiceprintMatcher(extractor)
        f1 = extractor.extract_from_array(_sine_wave(), _SR)
        f2 = extractor.extract_from_array(_white_noise(), _SR)
        e1 = matcher.compute_embedding(f1)
        e2 = matcher.compute_embedding(f2)
        sim_self, _ = matcher.verify(e1, e1)
        sim_diff, _ = matcher.verify(e1, e2)
        assert sim_self > sim_diff

    def test_enroll_and_load(self) -> None:
        """Enrolling a voiceprint and loading it should reproduce the exact embedding."""
        extractor = SpectrogramExtractor()
        matcher = VoiceprintMatcher(extractor)
        y = _sine_wave()
        features = extractor.extract_from_array(y, _SR)
        original_emb = matcher.compute_embedding(features)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "speaker.json")
            matcher.enroll(original_emb, save_path)
            loaded_emb = matcher.load_enrollment(save_path)

        np.testing.assert_allclose(original_emb, loaded_emb, atol=1e-6)

    def test_load_enrollment_missing_raises(self) -> None:
        matcher = VoiceprintMatcher()
        with pytest.raises(FileNotFoundError):
            matcher.load_enrollment("/no/such/file.json")
