"""Video deepfake classification using EfficientNet-B0.

Uses torchvision's EfficientNet-B0 backbone with ImageNet pretrained weights
(downloaded automatically on first use, ~21 MB) plus a custom binary head for
real vs. deepfake classification on 224x224 face crops.

To load FaceForensics++ fine-tuned head weights on top of ImageNet backbone:
    model = build_video_model(pretrained_path="models/video_head.pt")

Input:  (batch, 3, 224, 224) — BGR face crop normalized to [0, 1]
Output: (batch, 1)            — deepfake probability in [0, 1]
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

logger = logging.getLogger(__name__)

# ImageNet normalization statistics (required by torchvision pretrained models)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class VideoDeepfakeDetector(nn.Module):
    """EfficientNet-B0 backbone + binary head for deepfake classification.

    The backbone is initialized with ImageNet pretrained weights when
    pretrained=True (default). The binary classification head is always
    randomly initialized and should be fine-tuned on FaceForensics++ or DFDC
    for production-quality scores.

    ImageNet pretrained backbone gives meaningful visual features immediately —
    it already detects edges, textures, and face-like structures — making scores
    far more informative than a random-init custom CNN.
    """

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.3) -> None:
        """Initialize EfficientNet-B0 backbone and binary deepfake head.

        Args:
            pretrained: Load ImageNet pretrained weights for the backbone.
                Downloads ~21 MB from PyTorch model zoo on first call.
            dropout_rate: Dropout probability before the classification head.
        """
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # Reuse the EfficientNet-B0 feature extractor unchanged
        self.features = backbone.features    # outputs (batch, 1280, 7, 7) for 224x224 input
        self.avgpool = backbone.avgpool      # → (batch, 1280, 1, 1)

        # Binary classification head (always randomly initialized)
        # Fine-tune this head on FF++/DFDC for real deepfake detection scores
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass with ImageNet normalization applied internally.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224), values in [0, 1].

        Returns:
            Deepfake probability tensor of shape (batch, 1) in [0, 1].
        """
        # Normalize to ImageNet statistics before passing to pretrained backbone
        x = TF.normalize(x, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return torch.sigmoid(self.classifier(x))

    def predict(self, x: torch.Tensor) -> float:
        """Run inference and return a scalar deepfake probability.

        Args:
            x: Single face crop tensor of shape (1, 3, 224, 224), values in [0, 1].

        Returns:
            Deepfake probability as a Python float in [0.0, 1.0].
        """
        self.eval()
        with torch.no_grad():
            return float(self.forward(x).squeeze().detach())


def build_video_model(
    pretrained_path: str | None = None,
    imagenet_pretrained: bool = True,
) -> VideoDeepfakeDetector:
    """Build the video deepfake detector with optional fine-tuned weights.

    Loading order:
    1. Backbone always loads ImageNet weights (pretrained=True by default).
    2. If pretrained_path is given and the file exists, loads those weights on top.
       Supports two checkpoint formats:
         - Full model state dict (all keys) — loaded with strict=False
         - Classifier-head-only state dict (keys starting with "classifier.")

    Args:
        pretrained_path: Path to a .pt fine-tuned checkpoint. Optional.
        imagenet_pretrained: Load ImageNet backbone weights (strongly recommended).

    Returns:
        VideoDeepfakeDetector ready for inference.
    """
    model = VideoDeepfakeDetector(pretrained=imagenet_pretrained)

    if pretrained_path is None:
        if imagenet_pretrained:
            logger.info(
                "Video model: using ImageNet pretrained EfficientNet-B0 backbone "
                "with randomly initialized head. Fine-tune head on FF++ for best results."
            )
        else:
            logger.warning("Video model: fully random initialization — scores will be meaningless.")
        return model

    try:
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)

        # Detect checkpoint format and load accordingly
        if all(k.startswith("classifier.") for k in state.keys()):
            # Head-only checkpoint — load just the binary classification head
            head_state = {k[len("classifier."):]: v for k, v in state.items()}
            model.classifier.load_state_dict(head_state)
            logger.info("Loaded fine-tuned classifier head from %s", pretrained_path)
        else:
            # Full model checkpoint
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
