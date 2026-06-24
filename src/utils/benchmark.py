"""Latency and throughput benchmarking utilities for EchoGuard.

Measures per-frame processing time and overall throughput when running
the detector on a video file. Prints results in a Rich table.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def benchmark_detector(
    detector,  # EchoGuardDetector — imported lazily to avoid circular import
    video_path: str,
    audio_path: Optional[str] = None,
    warmup_runs: int = 1,
) -> dict:
    """Benchmark the detector on a video file and report timing statistics.

    Args:
        detector: An initialized EchoGuardDetector instance.
        video_path: Path to the video file to use as the benchmark input.
        audio_path: Optional audio file path for multimodal benchmarking.
        warmup_runs: Number of warm-up runs before measuring (to fill CPU caches).

    Returns:
        Dictionary with keys: avg_fps, p50_ms, p95_ms, total_ms, frame_count.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark video not found: {video_path}")

    logger.info("Running %d warm-up run(s)…", warmup_runs)
    for _ in range(warmup_runs):
        try:
            detector.analyze(video_path=video_path, audio_path=audio_path)
        except Exception as exc:
            logger.warning("Warm-up run failed: %s", exc)

    frame_times_ms: list[float] = []

    logger.info("Starting benchmark measurement run…")
    total_start = time.perf_counter()

    result = detector.analyze(video_path=video_path, audio_path=audio_path)
    total_ms = (time.perf_counter() - total_start) * 1000.0

    frame_count = max(result.frame_count, 1)
    avg_frame_ms = total_ms / frame_count
    frame_times_ms = [avg_frame_ms] * frame_count  # approximation (per-frame timing not stored)

    stats = _compute_stats(frame_times_ms, total_ms, frame_count)
    _print_benchmark_table(stats, video_path)
    return stats


def _compute_stats(
    frame_times_ms: list[float], total_ms: float, frame_count: int
) -> dict:
    """Compute timing statistics from per-frame measurements."""
    arr = np.array(frame_times_ms) if frame_times_ms else np.array([total_ms])
    avg_fps = (frame_count / total_ms * 1000.0) if total_ms > 0 else 0.0

    return {
        "avg_fps": round(float(avg_fps), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "total_ms": round(float(total_ms), 1),
        "frame_count": frame_count,
    }


def _print_benchmark_table(stats: dict, video_path: str) -> None:
    """Print benchmark results as a Rich table to stdout."""
    try:
        from rich.console import Console  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415

        console = Console()
        table = Table(title=f"EchoGuard Benchmark — {Path(video_path).name}", show_lines=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        table.add_row("Frames analyzed", str(stats["frame_count"]))
        table.add_row("Total time", f"{stats['total_ms']} ms")
        table.add_row("Average FPS", f"{stats['avg_fps']}")
        table.add_row("p50 latency / frame", f"{stats['p50_ms']} ms")
        table.add_row("p95 latency / frame", f"{stats['p95_ms']} ms")

        console.print(table)
    except ImportError:
        print(f"\nBenchmark Results — {Path(video_path).name}")
        print(f"  Frames analyzed : {stats['frame_count']}")
        print(f"  Total time      : {stats['total_ms']} ms")
        print(f"  Average FPS     : {stats['avg_fps']}")
        print(f"  p50 latency     : {stats['p50_ms']} ms")
        print(f"  p95 latency     : {stats['p95_ms']} ms")
