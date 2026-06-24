"""Generate synthetic test fixtures for EchoGuard's test suite.

Creates:
  - tests/fixtures/black_video.mp4    — 5-second all-black 224x224 video
  - tests/fixtures/sine_audio.wav     — 5-second 440Hz sine wave

These files allow all tests to run end-to-end without any real media files
or internet access.

Usage:
    python scripts/generate_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


_FIXTURES_DIR = Path("tests/fixtures")
_VIDEO_PATH = _FIXTURES_DIR / "black_video.mp4"
_AUDIO_PATH = _FIXTURES_DIR / "sine_audio.wav"

# Video parameters
_VIDEO_WIDTH = 224
_VIDEO_HEIGHT = 224
_VIDEO_FPS = 25.0
_VIDEO_DURATION_S = 5.0

# Audio parameters
_SAMPLE_RATE = 22050
_AUDIO_DURATION_S = 5.0
_SINE_FREQ_HZ = 440.0  # Concert A


def generate_black_video(output_path: Path) -> None:
    """Write a 5-second all-black video at 224x224 resolution.

    Args:
        output_path: Path where the MP4 file will be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = int(_VIDEO_FPS * _VIDEO_DURATION_S)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, _VIDEO_FPS, (_VIDEO_WIDTH, _VIDEO_HEIGHT)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    black_frame = np.zeros((_VIDEO_HEIGHT, _VIDEO_WIDTH, 3), dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(black_frame)
    writer.release()

    print(f"  [OK] Black video written: {output_path}  ({n_frames} frames @ {_VIDEO_FPS} fps)")


def generate_sine_audio(output_path: Path) -> None:
    """Write a 5-second 440Hz sine wave as a 16-bit mono WAV file.

    Uses scipy if available, otherwise writes a minimal WAV header manually.

    Args:
        output_path: Path where the WAV file will be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = int(_SAMPLE_RATE * _AUDIO_DURATION_S)
    t = np.linspace(0, _AUDIO_DURATION_S, n_samples, endpoint=False)
    y = (np.sin(2 * np.pi * _SINE_FREQ_HZ * t) * 0.8 * 32767).astype(np.int16)

    try:
        from scipy.io import wavfile  # noqa: PLC0415

        wavfile.write(str(output_path), _SAMPLE_RATE, y)
        print(f"  [OK] Sine audio written: {output_path}  ({_AUDIO_DURATION_S}s @ {_SAMPLE_RATE}Hz via scipy)")
    except ImportError:
        _write_wav_manual(output_path, y, _SAMPLE_RATE)
        print(f"  [OK] Sine audio written: {output_path}  ({_AUDIO_DURATION_S}s @ {_SAMPLE_RATE}Hz via manual WAV)")


def _write_wav_manual(path: Path, samples: np.ndarray, sr: int) -> None:
    """Write 16-bit mono PCM WAV without scipy."""
    import struct  # noqa: PLC0415

    data = samples.tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        # fmt chunk: PCM, 1 channel, sr, byte_rate, block_align, bits
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def main() -> None:
    """Generate all synthetic test fixtures."""
    print()
    print("EchoGuard — Generating Synthetic Test Fixtures")
    print("=" * 50)
    print(f"  Output directory: {_FIXTURES_DIR.resolve()}")
    print()

    try:
        generate_black_video(_VIDEO_PATH)
    except Exception as exc:
        print(f"  [ERROR] Failed to generate video: {exc}")
        sys.exit(1)

    try:
        generate_sine_audio(_AUDIO_PATH)
    except Exception as exc:
        print(f"  [ERROR] Failed to generate audio: {exc}")
        sys.exit(1)

    print()
    print("Done. You can now run the test suite without real media files:")
    print("  pytest tests/")
    print()


if __name__ == "__main__":
    main()
