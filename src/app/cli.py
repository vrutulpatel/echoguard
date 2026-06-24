"""Rich CLI application for EchoGuard deepfake detection.

Commands:
  analyze-video <path>  — Analyze a local video file for deepfakes
  analyze-audio <path>  — Analyze a local audio file for voice clones
  live --source N       — Live webcam/microphone detection (stub)
  benchmark <path>      — Run FPS/latency benchmark on a video file
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

from src.utils.logger import setup_logging

console = Console()

_BANNER = """[bold cyan]EchoGuard[/bold cyan] [dim]v0.1.0[/dim]  [dim]|[/dim]  [dim]Privacy-first deepfake detection[/dim]"""

_DEFAULT_CONFIG = Path("configs/default_config.yaml")


def _load_config(config_path: str) -> dict:
    """Load YAML config file, falling back to empty dict on failure."""
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        console.print(f"[yellow]Warning: Could not load config '{config_path}': {exc}[/yellow]")
        return {}


def _build_detector(cfg: dict):
    """Instantiate EchoGuardDetector from config values."""
    from src.pipeline.detector import EchoGuardDetector  # noqa: PLC0415

    return EchoGuardDetector(
        video_model_path=cfg.get("video_model_path"),
        audio_model_path=cfg.get("audio_model_path"),
        detection_threshold=cfg.get("detection_threshold", 0.65),
        video_weight=cfg.get("video_weight", 0.6),
        audio_weight=cfg.get("audio_weight", 0.4),
        frame_sample_rate=cfg.get("frame_sample_rate", 5),
        max_frames=cfg.get("max_frames", 300),
        device=cfg.get("device", "cpu"),
    )


def _print_result_table(result) -> None:
    """Render a DetectionResult as a Rich result table."""
    is_fake = result.is_deepfake
    verdict_color = "red" if is_fake else "green"
    verdict_text = f"[{verdict_color}]{result.verdict_emoji} {result.verdict}[/{verdict_color}]"

    table = Table(
        title="[bold]DETECTION RESULT[/bold]",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
        expand=False,
    )
    table.add_column("Field", style="cyan", no_wrap=True, min_width=22)
    table.add_column("Value", min_width=28)

    if result.video_score is not None:
        table.add_row("Video Score", f"{result.video_score:.3f}")
    if result.audio_score is not None:
        table.add_row("Audio Score", f"{result.audio_score:.3f}")
    table.add_row("Combined Score", f"[bold]{result.combined_score:.3f}[/bold]")
    table.add_row("Verdict", verdict_text)
    table.add_row("Confidence", result.confidence_label)

    flags_text = "\n".join(f"• {f}" for f in result.flags) if result.flags else "[dim]none[/dim]"
    table.add_row("Flags", flags_text)

    table.add_row("Processing Time", f"{result.processing_time_ms:.0f} ms")
    if result.frame_count > 0:
        table.add_row("Frames Analyzed", str(result.frame_count))

    console.print(table)


@click.group()
@click.option(
    "--config",
    default=str(_DEFAULT_CONFIG),
    show_default=True,
    help="Path to YAML configuration file.",
)
@click.pass_context
def cli(ctx: click.Context, config: str) -> None:
    """EchoGuard — real-time, privacy-first deepfake and voice clone detector.

    All processing is local; no data leaves your machine.
    """
    ctx.ensure_object(dict)
    cfg = _load_config(config)
    ctx.obj["config"] = cfg
    setup_logging(level=cfg.get("log_level", "WARNING"), enable_rich=True)


@cli.command("analyze-video")
@click.argument("video_path", type=click.Path(exists=True, readable=True))
@click.option(
    "--audio",
    "audio_path",
    default=None,
    type=click.Path(exists=True, readable=True),
    help="Optional separate audio file for multimodal analysis.",
)
@click.option("--output", "-o", default=None, help="Save annotated video to this path.")
@click.pass_context
def analyze_video(
    ctx: click.Context, video_path: str, audio_path: Optional[str], output: Optional[str]
) -> None:
    """Analyze a video file for deepfake indicators.

    VIDEO_PATH: Path to the video file to analyze (MP4, AVI, MOV, etc.)
    """
    console.rule(_BANNER)
    console.print()

    cfg = ctx.obj["config"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Loading detector…", total=None)
        detector = _build_detector(cfg)
        progress.update(task, description="[cyan]Analyzing video…")

        try:
            result = detector.analyze(video_path=video_path, audio_path=audio_path)
        except Exception as exc:
            console.print(f"[red]Analysis failed: {exc}[/red]")
            sys.exit(1)

        progress.update(task, completed=True, total=1)

    console.print()
    _print_result_table(result)

    if output:
        _save_annotated_video(video_path, output, result)

    sys.exit(0 if not result.is_deepfake else 1)


@cli.command("analyze-audio")
@click.argument("audio_path", type=click.Path(exists=True, readable=True))
@click.pass_context
def analyze_audio(ctx: click.Context, audio_path: str) -> None:
    """Analyze an audio file for voice clone indicators.

    AUDIO_PATH: Path to the audio file to analyze (WAV, MP3, FLAC, etc.)
    """
    console.rule(_BANNER)
    console.print()

    cfg = ctx.obj["config"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Loading detector…", total=None)
        detector = _build_detector(cfg)
        progress.update(task, description="[cyan]Analyzing audio…")

        try:
            result = detector.analyze(audio_path=audio_path)
        except Exception as exc:
            console.print(f"[red]Analysis failed: {exc}[/red]")
            sys.exit(1)

        progress.update(task, completed=True, total=1)

    console.print()
    _print_result_table(result)
    sys.exit(0 if not result.is_deepfake else 1)


@cli.command("live")
@click.option("--source", default=0, show_default=True, help="Camera/microphone device index.")
@click.pass_context
def live(ctx: click.Context, source: int) -> None:
    """Start live webcam and microphone deepfake detection.

    \b
    STATUS: Not yet implemented — coming in a future release.

    TODO: Implement real-time frame capture from OpenCV VideoCapture,
    audio streaming from sounddevice/pyaudio, and display the verdict
    overlay on a live preview window using annotate_frame().
    """
    console.rule(_BANNER)
    console.print()
    console.print(
        Panel(
            "[yellow]Live detection is not yet implemented.[/yellow]\n\n"
            "This feature is planned for the next release. It will support:\n"
            "  • Real-time webcam frame analysis\n"
            "  • Live microphone audio streaming\n"
            "  • On-screen verdict overlay\n\n"
            "Track progress at: [link]https://github.com/vrutulpatel/echoguard/issues[/link]",
            title="[bold yellow]TODO: Live Mode[/bold yellow]",
            border_style="yellow",
        )
    )
    sys.exit(0)


@cli.command("benchmark")
@click.argument("video_path", type=click.Path(exists=True, readable=True))
@click.option("--audio", "audio_path", default=None, type=click.Path(exists=True, readable=True))
@click.option("--warmup", default=1, show_default=True, help="Number of warm-up runs.")
@click.pass_context
def benchmark(
    ctx: click.Context, video_path: str, audio_path: Optional[str], warmup: int
) -> None:
    """Run latency and FPS benchmark on a video file.

    VIDEO_PATH: Path to the video file to use for benchmarking.
    """
    from src.utils.benchmark import benchmark_detector  # noqa: PLC0415

    console.rule(_BANNER)
    console.print()

    cfg = ctx.obj["config"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("[cyan]Initializing detector…", total=None)
        detector = _build_detector(cfg)

    console.print("[cyan]Running benchmark…[/cyan]")
    try:
        benchmark_detector(detector, video_path, audio_path=audio_path, warmup_runs=warmup)
    except Exception as exc:
        console.print(f"[red]Benchmark failed: {exc}[/red]")
        sys.exit(1)


def _save_annotated_video(
    video_path: str, output_path: str, result
) -> None:
    """Attempt to write a single-result annotated video (best-effort)."""
    try:
        from src.utils.visualizer import export_annotated_video  # noqa: PLC0415

        export_annotated_video(
            video_path=video_path,
            output_path=output_path,
            results_per_frame=[(result, None)],
        )
        console.print(f"[green]Annotated video saved to: {output_path}[/green]")
    except Exception as exc:
        console.print(f"[yellow]Warning: Could not save annotated video: {exc}[/yellow]")
