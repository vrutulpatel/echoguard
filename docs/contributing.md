# Contributing to EchoGuard

Thank you for your interest in contributing! EchoGuard is a solo portfolio project
but welcomes community improvements.

---

## Dev Environment Setup

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/echoguard.git
cd echoguard

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install all dependencies including dev tools
pip install -r requirements.txt
pip install -e .

# 4. (Optional) Generate synthetic test fixtures
python scripts/generate_synthetic.py
```

---

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_pipeline.py -v
```

All tests must pass before submitting a PR. New features require new tests.

---

## Linting and Formatting

```bash
# Check linting (must pass with zero errors)
ruff check src/ tests/ scripts/

# Auto-fix linting issues
ruff check src/ tests/ scripts/ --fix

# Format code (must match black formatting)
black src/ tests/ scripts/ --check

# Auto-format
black src/ tests/ scripts/
```

The CI pipeline enforces `ruff` and `black --check`. PRs that fail these checks will not be merged.

---

## PR Guidelines

1. **One concern per PR** — a bug fix, a new detection heuristic, a new test, or a docs update. Not all four at once.
2. **Tests required** — every new function or class needs at least one pytest test.
3. **No hardcoded paths** — use `pathlib.Path` and config values, never string literals like `/tmp/model.pt`.
4. **Type hints everywhere** — all function signatures must have complete type annotations.
5. **Docstrings on all public symbols** — module, class, and function level.
6. **Keep functions under 40 lines** — split into helpers if needed.
7. **Don't add model weights to the repo** — they belong in `models/` which is `.gitignore`d.

---

## Contributing Detection Heuristics

A new heuristic should:

- Return `(score: float, flags: list[str])` where score is in [0.0, 1.0].
- Include a docstring explaining *what* it detects and *why* that indicates a deepfake.
- Have at least one unit test with a synthetic signal that triggers the flag.
- Be under 40 lines (split into helpers if needed).

Add it to the relevant `_analyze_features()` or `_score_frame()` method in `src/audio/analyzer.py` or `src/video/analyzer.py`.

---

## Contributing Model Weights

If you have fine-tuned model weights on FaceForensics++, ASVspoof, or other relevant datasets:

1. Open an issue describing the dataset, training methodology, and reported metrics.
2. Host the weights on a stable public URL (Hugging Face Hub, Google Drive, etc.).
3. Open a PR updating `scripts/download_models.py` with the real URL and checksum.

We will not accept model weights directly in the repository due to file size constraints.

---

## Code of Conduct

Be respectful. Harassment, discrimination, or bad-faith contributions will result in immediate removal. See our [Ethical Guidelines](ethical_guidelines.md) for scope and intent.
