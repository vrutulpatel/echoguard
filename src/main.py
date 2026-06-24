"""Entry point for the EchoGuard CLI application."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'src' is importable
# whether this file is run as 'python src/main.py' or 'python -m src.main'
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.app.cli import cli


def main() -> None:
    """Launch the EchoGuard command-line interface."""
    cli()


if __name__ == "__main__":
    main()
