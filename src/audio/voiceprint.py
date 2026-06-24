"""Speaker voiceprint computation and verification.

Provides a lightweight speaker embedding approach using MFCC statistics
(mean + std per coefficient) as a voiceprint vector. Supports enrollment
(saving a reference voiceprint) and verification (cosine similarity check).

In production, replace with a dedicated speaker encoder such as resemblyzer
or a fine-tuned d-vector model for higher accuracy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.audio.spectrogram import AudioFeatures, SpectrogramExtractor

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 80  # n_mfcc * 2 (mean + std for each coefficient)


class VoiceprintMatcher:
    """Compute speaker voiceprint embeddings and measure speaker similarity.

    A voiceprint is a fixed-length vector summarizing the long-term spectral
    characteristics of a speaker's voice. Two recordings from the same speaker
    should have high cosine similarity; a voice clone often diverges enough to
    be detected (though accuracy depends heavily on the quality of the clone).
    """

    def __init__(self, extractor: SpectrogramExtractor | None = None) -> None:
        """Initialize with an optional shared SpectrogramExtractor.

        Args:
            extractor: SpectrogramExtractor to reuse. Creates a new one if None.
        """
        self.extractor = extractor or SpectrogramExtractor()

    def compute_embedding(self, features: AudioFeatures) -> np.ndarray:
        """Compute a voiceprint embedding from AudioFeatures.

        The embedding concatenates the per-coefficient mean and standard deviation
        of the MFCC matrix, yielding a 2*n_mfcc-dimensional vector.

        Args:
            features: AudioFeatures with at least the mfcc field populated.

        Returns:
            L2-normalized float32 embedding vector of shape (2*n_mfcc,).
        """
        mfcc = features.mfcc  # (n_mfcc, time_frames)
        mean = mfcc.mean(axis=1)   # (n_mfcc,)
        std = mfcc.std(axis=1)     # (n_mfcc,)
        embedding = np.concatenate([mean, std]).astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        return embedding

    def compute_from_file(self, audio_path: str) -> np.ndarray:
        """Convenience wrapper: load audio and compute voiceprint embedding.

        Args:
            audio_path: Path to the audio file.

        Returns:
            L2-normalized voiceprint embedding.
        """
        features = self.extractor.extract_from_file(audio_path)
        return self.compute_embedding(features)

    def enroll(self, embedding: np.ndarray, save_path: str) -> None:
        """Save a reference voiceprint embedding to disk as JSON.

        Args:
            embedding: Float32 numpy array (the voiceprint).
            save_path: File path where the voiceprint will be saved (e.g. speaker.json).
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"embedding": embedding.tolist(), "dim": int(embedding.shape[0])}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Voiceprint enrolled and saved to %s", save_path)

    def load_enrollment(self, save_path: str) -> np.ndarray:
        """Load a previously enrolled voiceprint from disk.

        Args:
            save_path: Path to the JSON file written by enroll().

        Returns:
            Float32 numpy array of the voiceprint embedding.

        Raises:
            FileNotFoundError: If the enrollment file does not exist.
        """
        path = Path(save_path)
        if not path.exists():
            raise FileNotFoundError(f"Voiceprint enrollment file not found: {save_path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return np.array(data["embedding"], dtype=np.float32)

    def verify(
        self, embedding: np.ndarray, reference: np.ndarray
    ) -> tuple[float, bool]:
        """Compare two voiceprint embeddings using cosine similarity.

        Args:
            embedding: Embedding to verify.
            reference: Reference (enrolled) embedding.

        Returns:
            Tuple of (similarity_score: float in [-1, 1], is_same_speaker: bool).
            Similarity above 0.75 is considered a match (empirical threshold).
        """
        sim = float(np.dot(embedding, reference))  # both are L2-normalized
        is_match = sim > 0.75
        logger.debug("Voiceprint similarity: %.3f — match=%s", sim, is_match)
        return sim, is_match

    def impersonation_score(
        self, embedding: np.ndarray, reference: np.ndarray
    ) -> float:
        """Return a suspicion score for voice impersonation.

        A high similarity between a test recording and a known reference voice
        may indicate that the test audio was cloned from the reference.
        Returns a score in [0.0, 1.0] where 1.0 = highly suspicious.

        Args:
            embedding: Embedding of the audio being tested.
            reference: Embedding of the known real speaker.

        Returns:
            Suspicion score — does NOT mean the audio is deepfake, only that
            the voice closely resembles the reference speaker.
        """
        sim, _ = self.verify(embedding, reference)
        # Remap from [-1, 1] to [0, 1], with high similarity = high suspicion
        return float((sim + 1.0) / 2.0)
