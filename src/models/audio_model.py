"""Audio voice clone classification using EfficientNet-B0 on mel spectrograms.

Uses torchvision's EfficientNet-B0 backbone with ImageNet pretrained weights.
Single-channel mel spectrograms are expanded to 3 channels before passing to
the backbone (a standard technique for audio classification with image CNNs).

To load ASVspoof fine-tuned head weights on top of ImageNet backbone:
    model = build_audio_model(pretrained_path="models/audio_head.pt")

Input:  (batch, 1, 128, 128) — normalized mel spectrogram, values in [0, 1]
Output: (batch, 1)            — voice clone probability in [0, 1]
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

logger = logging.getLogger(__name__)

# ImageNet normalization applied after channel expansion
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class AudioCloneDetector(nn.Module):
    """EfficientNet-B0 backbone + binary head for voice clone classification.

    Mel spectrograms (single channel) are repeated across 3 channels to match
    the EfficientNet input format. This lets the backbone's pretrained spatial
    feature detectors generalize to frequency-time representations — a technique
    validated in papers like CNN14 and the PANNs audio classification work.

    The binary head is randomly initialized and should be fine-tuned on
    ASVspoof 2019/2021 or WaveFake for production-quality anti-spoofing scores.
    """

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.3) -> None:
        """Initialize EfficientNet-B0 backbone and binary clone detection head.

        Args:
            pretrained: Load ImageNet pretrained weights for the backbone.
                Downloads ~21 MB from PyTorch model zoo on first call.
            dropout_rate: Dropout probability before the classification head.
        """
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Binary classification head (always randomly initialized)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass with channel expansion and ImageNet normalization.

        Args:
            x: Input tensor of shape (batch, 1, 128, 128), values in [0, 1].

        Returns:
            Voice clone probability tensor of shape (batch, 1) in [0, 1].
        """
        # Expand single-channel spectrogram to 3 channels for EfficientNet
        x = x.repeat(1, 3, 1, 1)                                     # (batch, 3, 128, 128)
        x = TF.normalize(x, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)  # ImageNet normalization
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return torch.sigmoid(self.classifier(x))

    def predict(self, x: torch.Tensor) -> float:
        """Run inference and return a scalar voice clone probability.

        Args:
            x: Single spectrogram tensor of shape (1, 1, 128, 128), values in [0, 1].

        Returns:
            Voice clone probability as a Python float in [0.0, 1.0].
        """
        self.eval()
        with torch.no_grad():
            return float(self.forward(x).squeeze().detach())


def build_audio_model(
    pretrained_path: str | None = None,
    imagenet_pretrained: bool = True,
) -> AudioCloneDetector:
    """Build the audio clone detector with optional fine-tuned weights.

    Loading order:
    1. Backbone always loads ImageNet weights (imagenet_pretrained=True by default).
    2. If pretrained_path is given and the file exists, loads those weights on top.
       Supports two checkpoint formats:
         - Full model state dict (all keys) — loaded with strict=False
         - Classifier-head-only state dict (keys starting with "classifier.")

    Args:
        pretrained_path: Path to a .pt fine-tuned checkpoint. Optional.
        imagenet_pretrained: Load ImageNet backbone weights (strongly recommended).

    Returns:
        AudioCloneDetector ready for inference.
    """
    model = AudioCloneDetector(pretrained=imagenet_pretrained)

    if pretrained_path is None:
        if imagenet_pretrained:
            logger.info(
                "Audio model: using ImageNet pretrained EfficientNet-B0 backbone "
                "with randomly initialized head. Fine-tune head on ASVspoof for best results."
            )
        else:
            logger.warning("Audio model: fully random initialization — scores will be meaningless.")
        return model

    try:
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)

        if all(k.startswith("classifier.") for k in state.keys()):
            head_state = {k[len("classifier."):]: v for k, v in state.items()}
            model.classifier.load_state_dict(head_state)
            logger.info("Loaded fine-tuned classifier head from %s", pretrained_path)
        else:
            model.load_state_dict(state, strict=False)
            logger.info("Loaded fine-tuned model weights from %s", pretrained_path)

    except FileNotFoundError:
        logger.warning(
            "Fine-tuned weights not found at '%s'. "
            "Using ImageNet backbone with random head. "
            "Run: python scripts/download_models.py",
            pretrained_path,
        )
    except Exception as exc:
        logger.warning("Could not load weights from '%s': %s — using ImageNet backbone.", pretrained_path, exc)

    return model
