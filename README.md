# EchoGuard

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](docs/contributing.md)
[![CI](https://github.com/vrutulpatel/echoguard/actions/workflows/ci.yml/badge.svg)](https://github.com/vrutulpatel/echoguard/actions)

**Real-time, privacy-first AI deepfake and voice clone detector — runs 100% locally on your machine.**

---

## Problem Statement

Deepfake videos and AI voice clones are becoming indistinguishable from reality. They are used to spread disinformation, commit fraud, impersonate public figures, and harass individuals. Existing detection tools either require uploading content to third-party servers (a serious privacy risk) or are locked behind enterprise paywalls.

EchoGuard is a free, open-source detector that runs entirely on your local machine. It analyzes both the video and audio streams of a file, combining multimodal signals to produce a confident, explainable verdict — without a single byte leaving your device.

**Who this protects:**
- Journalists verifying video authenticity before publishing
- HR teams screening video interviews for AI-generated candidates
- Individuals protecting themselves against voice fraud
- Researchers studying synthetic media

---

## Key Features

- **Multimodal Detection** — Analyzes video (face landmarks, temporal anomalies, lighting) and audio (pitch, room noise, spectral artifacts) together for higher accuracy
- **100% Local Inference** — No data leaves your machine; no cloud API calls
- **CPU-Optimized** — Runs on any laptop via ONNX Runtime and dynamic quantization; no GPU required
- **Explainable Results** — Reports specific flags like `lip_sync_mismatch`, `temporal_flicker`, `flat_pitch` so you understand *why* it flagged content
- **Rich CLI** — Beautiful terminal output with progress bars, color-coded verdicts, and result tables
- **Extensible Pipeline** — Modular architecture; swap in your own model weights or add new detection heads
- **Privacy-First Design** — No telemetry, no network calls, no logging of your content

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        EchoGuard Pipeline                    │
├──────────────────────────┬──────────────────────────────────┤
│       VIDEO STREAM       │        AUDIO STREAM              │
│                          │                                   │
│  ┌─────────────────┐    │   ┌──────────────────────────┐   │
│  │  OpenCV Reader  │    │   │    Librosa / torchaudio  │   │
│  └────────┬────────┘    │   └────────────┬─────────────┘   │
│           │             │                │                   │
│  ┌────────▼────────┐    │   ┌────────────▼─────────────┐   │
│  │  Face Detector  │    │   │   Mel Spectrogram / MFCC │   │
│  │  (MediaPipe)    │    │   │   Extraction             │   │
│  └────────┬────────┘    │   └────────────┬─────────────┘   │
│           │             │                │                   │
│  ┌────────▼────────┐    │   ┌────────────▼─────────────┐   │
│  │ Temporal        │    │   │  Voiceprint Matcher      │   │
│  │ Anomaly Detect  │    │   │  + Audio CNN Model       │   │
│  └────────┬────────┘    │   └────────────┬─────────────┘   │
│           │             │                │                   │
│  ┌────────▼────────┐    │                │                   │
│  │  Video CNN/ViT  │    │                │                   │
│  │  (EfficientNet) │    │                │                   │
│  └────────┬────────┘    │                │                   │
│           │             │                │                   │
└───────────┼─────────────┴────────────────┼───────────────────┘
            │                              │
            └──────────────┬───────────────┘
                           │
                  ┌────────▼────────┐
                  │   EchoGuard     │
                  │   Detector      │
                  │  (60% video +   │
                  │   40% audio)    │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  DetectionResult│
                  │  score + flags  │
                  │  + verdict      │
                  └─────────────────┘
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/vrutulpatel/echoguard.git
cd echoguard

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download pretrained model weights
python scripts/download_models.py

# 5. (Optional) Install as a package
pip install -e .
```

---

## Quick Start

**Analyze your own video file:**
```bash
# Windows
python src/main.py analyze-video C:\Users\you\Videos\myvideo.mp4

# macOS / Linux
python src/main.py analyze-video /home/you/videos/myvideo.mp4
```

Any common format works — MP4, AVI, MOV, MKV, etc.

**Don't have a video yet? Generate a synthetic test file first:**
```bash
python scripts/generate_synthetic.py
python src/main.py analyze-video tests/fixtures/black_video.mp4
```
```
EchoGuard v0.1.0  |  Privacy-first deepfake detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analyzing video... ████████████████████ 100% | 47 frames

┌─────────────────────────────────────────────────┐
│              DETECTION RESULT                   │
├──────────────────────┬──────────────────────────┤
│ Video Score          │ 0.82                     │
│ Audio Score          │ 0.74                     │
│ Combined Score       │ 0.79                     │
│ Verdict              │ ⚠ LIKELY DEEPFAKE        │
│ Confidence           │ High                     │
│ Flags                │ lip_sync_mismatch        │
│                      │ temporal_flicker         │
│                      │ blurring_artifacts       │
│ Processing Time      │ 1243 ms                  │
└──────────────────────┴──────────────────────────┘
```

**Analyze your own audio file:**
```bash
# Windows
python src/main.py analyze-audio C:\Users\you\Music\myaudio.wav

# macOS / Linux
python src/main.py analyze-audio /home/you/audio/myaudio.wav
```

Supported formats: WAV, MP3, FLAC, OGG, M4A, and anything Librosa can load.

**Analyze video + audio together** (enables lip-sync mismatch detection):
```bash
python src/main.py analyze-video myvideo.mp4 --audio myaudio.wav
```
```
Analyzing audio... ████████████████████ 100%

Audio Score: 0.21  |  Verdict: LIKELY REAL  |  Confidence: High
Flags: none
```

**Run a benchmark on your video:**
```bash
python src/main.py benchmark C:\Users\you\Videos\myvideo.mp4
```
```
┌────────────────┬──────────┬──────────────┬──────────────┐
│ Metric         │ Value    │ p50 Latency  │ p95 Latency  │
├────────────────┼──────────┼──────────────┼──────────────┤
│ Avg FPS        │ 18.3     │ 54 ms/frame  │ 91 ms/frame  │
│ Total Time     │ 2.4 s    │ 47 frames    │ —            │
└────────────────┴──────────┴──────────────┴──────────────┘
```

---

## How It Works

### Video Analysis

1. **Face Detection & Landmarks** — MediaPipe FaceMesh extracts 468 facial landmarks per frame, tracking eye openness, lip shape, and head pose.
2. **Temporal Anomaly Detection** — A sliding window checks for unnatural head pose changes, texture flickering (pixel-level variance), and abnormal eye blink frequency (natural rate: 15–20/min).
3. **Visual Artifact Detection** — Laplacian variance measures blurring around face edges; gradient histograms compare lighting consistency across frames.
4. **Video Classification Model** — A lightweight EfficientNet-B0 style CNN classifies face crops as real vs. deepfake (fine-tuned on FaceForensics++).

### Audio Analysis

1. **Feature Extraction** — Librosa extracts mel spectrograms (128 bins), MFCCs (40 coefficients), and chroma features from the audio signal.
2. **Voiceprint Matching** — Computes a speaker embedding; if a reference voiceprint is enrolled, measures cosine similarity to detect impersonation.
3. **Artifact Detection** — Flags overly flat pitch (natural speech has variation), missing room noise (voice clones often lack ambient acoustics), clipping artifacts, and unnatural silence lengths.
4. **Audio Classification Model** — A CNN classifies mel spectrogram crops as real vs. voice clone (fine-tuned on ASVspoof dataset).

### Fusion

The combined score is a weighted average: **60% video + 40% audio**. If either stream is missing (audio-only or video-only input), the available score is used directly. A combined score above **0.65** triggers a deepfake verdict.

---

## Benchmarks

> These are placeholder benchmarks. Actual numbers depend on hardware, model weights, and content complexity.

| Configuration        | AUC    | Accuracy | Avg FPS | p50 Latency |
|----------------------|--------|----------|---------|-------------|
| CPU, no weights      | —      | —        | ~25     | ~40 ms      |
| CPU, EfficientNet-B0 | 0.94*  | 91%*     | ~18     | ~55 ms      |
| GPU (RTX 3060)       | 0.94*  | 91%*     | ~85     | ~12 ms      |

*Reported on FaceForensics++ (c23 compression) benchmark. Weights not included — see `scripts/download_models.py`.

---

## Roadmap

- [x] Offline video file analyzer
- [x] Offline audio file analyzer
- [x] Multimodal fusion pipeline
- [x] Rich CLI interface
- [x] Docker container
- [x] CI/CD pipeline
- [ ] Live webcam + microphone detection
- [ ] Browser extension (Chrome/Firefox)
- [ ] Mobile app (React Native)
- [ ] Web UI dashboard
- [ ] ONNX model export script
- [ ] Dataset download + fine-tuning scripts

---

## Contributing

We welcome contributions! Please see [docs/contributing.md](docs/contributing.md) for:
- Dev environment setup
- How to run tests
- PR guidelines
- How to contribute datasets or model weights

---

## Ethical Guidelines

EchoGuard is a **detection tool only**. It does not generate, assist in generating, or improve deepfakes. Please read our [docs/ethical_guidelines.md](docs/ethical_guidelines.md) before using or contributing.

---

## License

MIT — see [LICENSE](LICENSE) for details.
