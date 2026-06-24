"""Model loading utilities supporting PyTorch and ONNX Runtime.

Provides a unified interface for device placement, dynamic quantization,
and in-memory caching of already-constructed models. Weight loading is
handled upstream by build_video_model() and build_audio_model() — this
module only handles post-construction operations.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Module-level model cache keyed by (model_class_name, weights_path, device)
_model_cache: dict[str, object] = {}

# Type alias for supported model backends
OnnxSession = object  # onnxruntime.InferenceSession, imported lazily


def load_pytorch_model(
    model: nn.Module,
    weights_path: Optional[str],
    device: str = "cpu",
    quantize: bool = True,
    cache_key: Optional[str] = None,
) -> nn.Module:
    """Place a model on device, apply optional dynamic quantization, and cache it.

    Weight loading is NOT done here — it is the responsibility of the caller
    (build_video_model / build_audio_model). This function only handles:
      1. Moving the model to the target device
      2. Applying dynamic INT8 quantization on CPU (reduces memory ~2x)
      3. Caching the model in memory for fast repeated access

    Args:
        model: Already-constructed and weight-loaded PyTorch model.
        weights_path: Used only as part of the cache key; no file I/O is done.
        device: Target device string ('cpu' or 'cuda').
        quantize: Apply torch.quantization.quantize_dynamic when device == 'cpu'.
        cache_key: Optional explicit cache key. Defaults to
            '{model_class}:{weights_path}:{device}'.

    Returns:
        Model in eval mode on the target device, quantized if applicable.
    """
    key = cache_key or f"{type(model).__name__}:{weights_path}:{device}"
    if key in _model_cache:
        logger.debug("Returning cached model (key=%s)", key)
        return _model_cache[key]  # type: ignore[return-value]

    actual_device = torch.device(
        device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    model = model.to(actual_device)
    model.eval()

    if quantize and str(actual_device) == "cpu":
        try:
            # Suppress DeprecationWarning from torch.ao.quantization in PyTorch >= 2.6
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
            logger.debug("Applied dynamic INT8 quantization.")
        except Exception as exc:
            logger.warning("Dynamic quantization failed: %s — using FP32.", exc)

    _model_cache[key] = model
    return model


def load_onnx_model(onnx_path: str) -> OnnxSession:
    """Load an ONNX model for inference via ONNX Runtime.

    ONNX Runtime provides CPU-optimized graph execution that is faster than
    PyTorch CPU inference for fixed-topology models. Prefers CPU execution
    provider unless CUDA is explicitly configured via the environment.

    Args:
        onnx_path: Path to the .onnx model file.

    Returns:
        onnxruntime.InferenceSession ready for inference.

    Raises:
        FileNotFoundError: If the ONNX file does not exist.
        ImportError: If onnxruntime is not installed.
    """
    if not Path(onnx_path).exists():
        raise FileNotFoundError(
            f"ONNX model not found at '{onnx_path}'. "
            "Run python scripts/download_models.py to download model weights."
        )

    cache_key = f"onnx:{onnx_path}"
    if cache_key in _model_cache:
        logger.debug("Returning cached ONNX session for %s", onnx_path)
        return _model_cache[cache_key]

    try:
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for ONNX model inference. "
            "Install with: pip install onnxruntime>=1.16.0"
        ) from exc

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    logger.info("Loaded ONNX model from %s (provider: CPU)", onnx_path)

    _model_cache[cache_key] = session
    return session


def run_onnx_inference(session: OnnxSession, input_array: "np.ndarray") -> float:  # noqa: F821
    """Run inference on an ONNX session and return a scalar probability.

    Args:
        session: onnxruntime.InferenceSession loaded via load_onnx_model.
        input_array: Input numpy array with the correct shape for the model.

    Returns:
        Predicted probability as a Python float in [0.0, 1.0].
    """
    import numpy as np  # noqa: PLC0415

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_array.astype(np.float32)})
    prob = float(np.squeeze(output[0]))
    # Clamp to valid probability range in case the model output isn't sigmoid-gated
    return max(0.0, min(1.0, prob))


def clear_model_cache() -> None:
    """Remove all cached models from memory (useful for testing or memory management)."""
    _model_cache.clear()
    logger.debug("Model cache cleared.")


def resolve_device(requested: str = "cpu") -> str:
    """Resolve the target device, falling back to CPU if CUDA is unavailable.

    Args:
        requested: Desired device string ('cpu' or 'cuda').

    Returns:
        Actual device string to use.
    """
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")
        return "cpu"
    return requested
