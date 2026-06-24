"""Logging configuration for EchoGuard.

Sets up a Rich-powered console handler for beautiful colored output and a
rotating file handler that writes to logs/echoguard.log. Call setup_logging()
once at application startup.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    log_filename: str = "echoguard.log",
    enable_rich: bool = True,
) -> logging.Logger:
    """Configure application-wide logging with Rich console output and file rotation.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory where the log file will be written.
        log_filename: Name of the rotating log file.
        enable_rich: Use Rich's colored console handler if True; plain otherwise.

    Returns:
        Root logger configured with the specified handlers.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicate output on repeated calls
    root_logger.handlers.clear()

    # Console handler
    if enable_rich:
        try:
            from rich.logging import RichHandler  # noqa: PLC0415

            console_handler = RichHandler(
                level=numeric_level,
                rich_tracebacks=True,
                markup=True,
                show_path=False,
            )
            console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        except ImportError:
            console_handler = _plain_console_handler(numeric_level)
    else:
        console_handler = _plain_console_handler(numeric_level)

    root_logger.addHandler(console_handler)

    # File handler — rotating to prevent unbounded growth
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / log_filename,
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)
    except OSError as exc:
        root_logger.warning("Could not create log file in '%s': %s", log_dir, exc)

    return root_logger


def _plain_console_handler(level: int) -> logging.StreamHandler:
    """Create a simple stream handler with a readable format."""
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    return handler


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger (convenience wrapper around logging.getLogger).

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Named logger that inherits the root configuration.
    """
    return logging.getLogger(name)
