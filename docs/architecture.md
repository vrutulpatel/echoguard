# EchoGuard Architecture

## Overview

EchoGuard is a modular, privacy-first multimodal deepfake detection system. Every component is designed to run entirely on a local machine without network access, ensuring that analyzed content never leaves the user's device.

---

## Module Map

```
src/
├── main.py               Entry point; delegates to CLI
├── app/cli.py            Click + Rich command-line interface
├── video/
│   ├── face_detector.py  MediaPipe FaceMesh landmark extraction
│   ├── analyzer.py       Per-frame heuristic scoring (blur, lighting, lip-sync)
│   └── temporal.py       Sliding-window temporal anomaly detection
├── audio/
│   ├── spectrogram.py    Librosa mel spectrogram / MFCC / chroma extraction
│   ├── voiceprint.py     Speaker embedding enrollment and verification
│   └── analyzer.py       Heuristic audio clone detection
├── models/
│   ├── video_model.py    EfficientNet-B0-style PyTorch CNN for face crops
│   ├── audio_model.py    ResNet-style PyTorch CNN for mel spectrograms
│   └── model_loader.py   Unified .pt / .onnx loader with quantization + caching
├── pipeline/
│   ├── result.py         DetectionResult dataclass and score fusion utilities
│   └── detector.py       EchoGuardDetector — orchestrates all modules
└── utils/
    ├── logger.py          Rich + rotating file logging setup
    ├── visualizer.py      Frame annotation and annotated video export
    └── benchmark.py       FPS / latency benchmarking with Rich table output
```

---

## Data Flow

### Video Analysis Path

```
VideoCapture (OpenCV)
    │
    ▼  every Nth frame (frame_sample_rate)
FaceDetector.process_frame()
    ├── MediaPipe FaceMesh → 468 landmarks (x, y, z)
    ├── Eye Aspect Ratio (EAR) → blink detection
    ├── Lip distance → mouth openness per frame
    └── Head pose estimation (pitch, yaw, roll)
    │
    ▼
VideoAnalyzer._score_frame()
    ├── _check_edge_blurring()
    │       Laplacian variance of face boundary vs. interior
    │       → blurring_artifacts flag
    └── _check_lighting_consistency()
            Face brightness vs. scene brightness delta
            → lighting_inconsistency flag
    │
    ▼
VideoAnalyzer._check_lip_sync()
    Cross-correlation of lip_distance ↔ audio RMS energy
    → lip_sync_mismatch flag
    │
    ▼
TemporalAnalyzer.analyze()
    ├── _check_pose_changes() — abrupt head rotation between frames
    ├── _check_blink_rate()  — blinks/min outside 10–30 range
    └── _check_texture_flickering() — pixel variance in sliding window
    │
    ▼
VideoDeepfakeDetector.predict()  (model inference on face crops)
    EfficientNet-B0-style CNN: (1, 3, 224, 224) → deepfake_prob
    │
    ▼
heuristic_score * 0.5 + model_score * 0.5 → video_score
```

### Audio Analysis Path

```
AudioAnalyzer.analyze()
    │
    ▼
SpectrogramExtractor.extract_from_file()
    ├── librosa.feature.melspectrogram → (128, T) dB-scaled
    ├── librosa.feature.mfcc          → (40, T)
    ├── librosa.feature.chroma_stft   → (12, T)
    ├── librosa.feature.rms           → (T,) energy envelope
    └── librosa.pyin                  → (T,) fundamental frequency (f0)
    │
    ▼
AudioAnalyzer._check_pitch_flatness()
    Std dev of f0 in semitones < threshold → flat_pitch flag

AudioAnalyzer._check_silence_patterns()
    Max contiguous RMS-silent frames > threshold → unnatural_silence flag

AudioAnalyzer._check_clipping()
    Fraction of RMS frames > clipping_threshold → clipping_artifacts flag

AudioAnalyzer._check_room_noise()
    Mean mel dB during silent frames < -70 dB → missing_room_noise flag
    │
    ▼
AudioCloneDetector.predict()  (model inference on mel spectrogram)
    ResNet CNN: (1, 1, 128, 128) → clone_prob
    │
    ▼
heuristic_score * 0.5 + model_score * 0.5 → audio_score
```

### Fusion

```
combine_scores(video_score, audio_score, video_weight=0.6, audio_weight=0.4)
    → combined_score ∈ [0.0, 1.0]

combined_score ≥ detection_threshold (default 0.65)
    → is_deepfake = True
    → verdict = "LIKELY DEEPFAKE"
```

---

## Design Decisions

### Why local-first inference?

Privacy is a core requirement. Deepfake detection use cases often involve sensitive personal media — recordings of individuals, private conversations, or confidential meetings. Uploading content to a third-party API for analysis creates unnecessary risk:

1. **Data exposure**: The server operator can log, store, or analyze uploaded content.
2. **Regulatory compliance**: GDPR, HIPAA, and other regulations restrict sharing of biometric and personal data.
3. **Availability**: Cloud APIs can be rate-limited, offline, or monetized away from free use.

EchoGuard solves this by running all inference locally using PyTorch with CPU fallback and ONNX Runtime for optimized deployment.

### Why CPU-first, not GPU?

The target user (a journalist, HR manager, or individual) is unlikely to have a dedicated GPU. ONNX Runtime and dynamic INT8 quantization (`torch.quantization.quantize_dynamic`) allow acceptable real-time performance (~15–25 FPS on a modern CPU) without requiring CUDA.

### Why multimodal fusion?

Neither video nor audio analysis alone is reliable enough for production use:

- **Video-only** detectors are fooled by audio-swapped content where the face is real but the voice is synthetic.
- **Audio-only** detectors miss face-swap deepfakes.
- **Fused** detection catches both attack surfaces. The 60/40 weighting favors video because face-swap deepfakes are the more common and visually convincing threat.

### Why heuristic + model hybrid scoring?

Heuristic scores (blink rate, lip-sync correlation, pitch flatness) work even without pretrained weights, making the pipeline fully runnable out-of-the-box. When pretrained weights are available, model scores are blended at 50/50 with heuristics. This fallback design allows developers to test and iterate without downloading large model files.

### ONNX Runtime vs. PyTorch for inference

ONNX Runtime's CPU execution provider typically runs faster than native PyTorch CPU inference for fixed-topology models because it applies graph-level optimizations (operator fusion, memory planning) at session load time. For production deployment, export models to ONNX using `scripts/download_models.py` instructions.

---

## Extending EchoGuard

### Adding a new detection head

1. Implement your detector in `src/video/` or `src/audio/` returning `(score: float, flags: list[str])`.
2. Call it from `EchoGuardDetector._run_video_analysis()` or `_run_audio_analysis()`.
3. Add any new flags to the `DetectionResult.flags` list.
4. Add tests in `tests/`.

### Swapping model architectures

Replace the model definition in `src/models/video_model.py` or `src/models/audio_model.py`. As long as your model:
- Accepts the documented input shape
- Returns a sigmoid-gated scalar in [0, 1]
- Can be loaded with `torch.load()`

…it will work with the existing pipeline without any other changes.

### Adding live detection

See the `TODO` stub in `src/app/cli.py → live()`. The key additions needed are:
- OpenCV `VideoCapture(source)` reading in a background thread
- `sounddevice` or `pyaudio` audio buffering in a second thread
- A shared queue feeding frames/audio chunks into the detector
- An OpenCV `imshow` loop calling `annotate_frame()` for real-time overlay
