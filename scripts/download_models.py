"""Download pretrained model weights for EchoGuard.

The EfficientNet-B0 backbone already loads ImageNet pretrained weights
automatically via torchvision — no download needed for that.

This script handles downloading FINE-TUNED HEAD weights from Hugging Face
that were trained specifically on deepfake datasets:

  - Video head:  fine-tuned on FaceForensics++ (FF++) or DFDC
  - Audio head:  fine-tuned on ASVspoof 2019 LA or WaveFake

Usage:
    python scripts/download_models.py                  # interactive prompt
    python scripts/download_models.py --video-only
    python scripts/download_models.py --audio-only
    python scripts/download_models.py --list           # show available models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Registry of known Hugging Face model checkpoints for EchoGuard.
#
# FORMAT:
#   repo_id   — the HuggingFace repo (owner/model-name)
#   filename  — the specific file inside that repo
#   target    — where to save it locally
#   type      — "head_only" saves just the classifier; "full" replaces backbone too
#
# HOW TO FIND MODELS:
#   1. Go to https://huggingface.co/models
#   2. Search: "deepfake detection EfficientNet" (video) or "anti-spoofing" (audio)
#   3. Check the model card for input format compatibility (224x224 RGB / 128x128 mel)
#   4. Add an entry below and submit a PR!
# ──────────────────────────────────────────────────────────────────────────────

_VIDEO_MODELS: list[dict] = [
    {
        "name": "EfficientNet-B0 head — FaceForensics++ (community)",
        "repo_id": "PLACEHOLDER/echoguard-video-head-ff++",   # replace with real repo
        "filename": "video_head.pt",
        "target": "models/video_model.pt",
        "type": "head_only",
        "dataset": "FaceForensics++ (c23, 4 manipulation types)",
        "notes": "Replace PLACEHOLDER with an actual Hugging Face repo once available.",
    },
]

_AUDIO_MODELS: list[dict] = [
    {
        "name": "EfficientNet-B0 head — ASVspoof 2019 LA (community)",
        "repo_id": "PLACEHOLDER/echoguard-audio-head-asvspoof",  # replace with real repo
        "filename": "audio_head.pt",
        "target": "models/audio_model.pt",
        "type": "head_only",
        "dataset": "ASVspoof 2019 Logical Access",
        "notes": "Replace PLACEHOLDER with an actual Hugging Face repo once available.",
    },
]


def _download_from_hf(repo_id: str, filename: str, target: str) -> bool:
    """Download a single file from a Hugging Face repository.

    Args:
        repo_id: HuggingFace repository id (e.g. 'owner/model-name').
        filename: Filename within the repository.
        target: Local path to save the downloaded file.

    Returns:
        True on success, False on failure.
    """
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError:
        print("  [ERROR] huggingface_hub not installed. Run: pip install huggingface_hub>=0.20.0")
        return False

    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {repo_id}/{filename} …")
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target_path.parent),
            local_dir_use_symlinks=False,
        )
        # Rename to expected target path if needed
        downloaded = Path(local_path)
        if downloaded != target_path:
            downloaded.rename(target_path)
        print(f"  [OK] Saved to {target}")
        return True
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return False


def _print_model_info(models: list[dict], label: str) -> None:
    """Print formatted info for a list of model entries."""
    print(f"\n  {label}:")
    for m in models:
        print(f"    ┌─ {m['name']}")
        print(f"    │  Dataset  : {m['dataset']}")
        print(f"    │  HF repo  : {m['repo_id']}")
        print(f"    │  File     : {m['filename']}")
        print(f"    │  Saves to : {m['target']}")
        print(f"    │  Notes    : {m['notes']}")
        print(f"    └─")


def _export_onnx_instructions() -> None:
    """Print ONNX export instructions for production deployment."""
    print("""
  ── ONNX Export (optional, for faster CPU inference) ──────────────────────
  After downloading or training weights, export to ONNX:

    import torch
    from src.models.video_model import build_video_model

    model = build_video_model("models/video_model.pt")
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, "models/video_model.onnx",
        opset_version=13,
        input_names=["face_crop"],
        output_names=["deepfake_prob"],
        dynamic_axes={"face_crop": {0: "batch"}},
    )

  Then set  video_model_path: models/video_model.onnx  in configs/default_config.yaml.
  ──────────────────────────────────────────────────────────────────────────────
""")


def _download_models(models: list[dict]) -> int:
    """Attempt to download all models in the list. Returns count of failures."""
    failures = 0
    for m in models:
        if "PLACEHOLDER" in m["repo_id"]:
            print(f"\n  [SKIP] {m['name']}")
            print(f"         Repo ID is a placeholder — update scripts/download_models.py")
            print(f"         with a real HuggingFace repo before running.")
            failures += 1
            continue
        ok = _download_from_hf(m["repo_id"], m["filename"], m["target"])
        if not ok:
            failures += 1
    return failures


def main() -> None:
    """Entry point — parse flags and download requested model weights."""
    parser = argparse.ArgumentParser(description="Download EchoGuard pretrained model heads.")
    parser.add_argument("--video-only", action="store_true", help="Download only video model head.")
    parser.add_argument("--audio-only", action="store_true", help="Download only audio model head.")
    parser.add_argument("--list", action="store_true", help="List available models without downloading.")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  EchoGuard — Pretrained Model Head Downloader")
    print("=" * 70)
    print()
    print("  NOTE: The EfficientNet-B0 BACKBONE loads ImageNet weights automatically")
    print("  via torchvision on first inference — no download needed here.")
    print()
    print("  This script downloads the CLASSIFICATION HEAD weights fine-tuned on")
    print("  deepfake datasets (FF++, ASVspoof). Without these, scores come from")
    print("  an untrained head on top of ImageNet features — still directional,")
    print("  but not calibrated for deepfake detection accuracy.")
    print()

    if args.list:
        _print_model_info(_VIDEO_MODELS, "Video (deepfake) models")
        _print_model_info(_AUDIO_MODELS, "Audio (voice clone) models")
        _export_onnx_instructions()
        sys.exit(0)

    failures = 0
    if not args.audio_only:
        print("  ── Video model head ──────────────────────────────────────────────────")
        failures += _download_models(_VIDEO_MODELS)

    if not args.video_only:
        print("  ── Audio model head ──────────────────────────────────────────────────")
        failures += _download_models(_AUDIO_MODELS)

    _export_onnx_instructions()

    if failures > 0:
        print(f"  {failures} download(s) failed or skipped (see above).")
        print()
        print("  To contribute pretrained weights:")
        print("  1. Train the classifier head on FF++ or ASVspoof (see docs/contributing.md)")
        print("  2. Upload the .pt file to Hugging Face Hub")
        print("  3. Update the _VIDEO_MODELS / _AUDIO_MODELS registry in this script")
        print("  4. Submit a PR — your weights will help everyone!")
    else:
        print("  All downloads successful!")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
