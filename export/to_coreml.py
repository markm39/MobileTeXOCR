"""
Export model to CoreML format for iOS/macOS deployment.

Features:
- INT8/INT4 quantization for reduced model size
- Neural Engine optimization
- Separate encoder/decoder for efficient inference
"""

from typing import Optional, Tuple, List
from pathlib import Path
import logging
import tempfile

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EncoderForCoreML(nn.Module):
    """Encoder wrapper for CoreML export."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.encoder(image)
        return output.features


class DecoderForCoreML(nn.Module):
    """Decoder wrapper for CoreML export."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        encoder_features: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.decoder(encoder_features, input_ids)
        return logits[:, -1, :]


def export_to_coreml(
    model: nn.Module,
    output_path: str,
    image_size: int = 384,
    quantize: Optional[str] = None,
    compute_units: str = "ALL",
) -> Tuple[str, str]:
    """Export model to CoreML format.

    Args:
        model: HandwrittenLaTeXOCR model
        output_path: Base path for output (without extension)
        image_size: Input image size
        quantize: Quantization mode: None, "int8", "int4", or "float16"
        compute_units: "ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"

    Returns:
        Tuple of (encoder_path, decoder_path)
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError("coremltools not installed. Install with: pip install coremltools")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(output_path).stem

    encoder_path = output_dir / f"{base_name}_encoder.mlpackage"
    _export_encoder_coreml(
        model, str(encoder_path), image_size, quantize, compute_units
    )

    decoder_path = output_dir / f"{base_name}_decoder.mlpackage"
    _export_decoder_coreml(
        model, str(decoder_path), quantize, compute_units
    )

    logger.info(f"Exported CoreML encoder to {encoder_path}")
    logger.info(f"Exported CoreML decoder to {decoder_path}")

    return str(encoder_path), str(decoder_path)


def _export_encoder_coreml(
    model: nn.Module,
    output_path: str,
    image_size: int,
    quantize: Optional[str],
    compute_units: str,
):
    """Export encoder to CoreML."""
    import coremltools as ct

    encoder_wrapper = EncoderForCoreML(model.encoder)
    encoder_wrapper.eval()

    dummy_image = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        traced = torch.jit.trace(encoder_wrapper, dummy_image)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, image_size, image_size),
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name="features")],
        compute_units=_get_compute_units(compute_units),
        minimum_deployment_target=ct.target.iOS16,
    )

    if quantize:
        mlmodel = _quantize_model(mlmodel, quantize)

    mlmodel.author = "Handwritten LaTeX OCR"
    mlmodel.short_description = "Vision encoder for handwritten math recognition"
    mlmodel.version = "1.0"

    mlmodel.save(output_path)


def _export_decoder_coreml(
    model: nn.Module,
    output_path: str,
    quantize: Optional[str],
    compute_units: str,
):
    """Export decoder to CoreML."""
    import coremltools as ct

    decoder_wrapper = DecoderForCoreML(model.decoder)
    decoder_wrapper.eval()

    patch_size = model.encoder.patch_size
    image_size = model.config.image_size
    num_patches = (image_size // patch_size) ** 2
    d_model = model.config.d_model

    dummy_features = torch.randn(1, num_patches, d_model)
    dummy_ids = torch.randint(0, model.tokenizer.vocab_size, (1, 10))

    with torch.no_grad():
        traced = torch.jit.trace(decoder_wrapper, (dummy_features, dummy_ids))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="encoder_features",
                shape=(1, num_patches, d_model),
            ),
            ct.TensorType(
                name="input_ids",
                shape=ct.Shape((1, ct.RangeDim(1, 512))),
                dtype=int,
            ),
        ],
        outputs=[ct.TensorType(name="logits")],
        compute_units=_get_compute_units(compute_units),
        minimum_deployment_target=ct.target.iOS16,
    )

    if quantize:
        mlmodel = _quantize_model(mlmodel, quantize)

    mlmodel.author = "Handwritten LaTeX OCR"
    mlmodel.short_description = "Decoder for handwritten math recognition"
    mlmodel.version = "1.0"

    mlmodel.save(output_path)


def _get_compute_units(compute_units: str):
    """Convert compute units string to CoreML enum."""
    import coremltools as ct

    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }
    return mapping.get(compute_units, ct.ComputeUnit.ALL)


def _quantize_model(mlmodel, quantize: str):
    """Apply quantization to CoreML model."""
    import coremltools as ct
    from coremltools.models.neural_network import quantization_utils

    if quantize == "float16":
        return quantization_utils.quantize_weights(mlmodel, nbits=16)
    elif quantize == "int8":
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
        )
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        return ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config)
    elif quantize == "int4":
        op_config = ct.optimize.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=4,
        )
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        return ct.optimize.coreml.palettize_weights(mlmodel, config=config)
    else:
        return mlmodel


def get_model_size(model_path: str) -> float:
    """Get model size in MB."""
    import os

    if os.path.isdir(model_path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total / (1024 * 1024)
    else:
        return os.path.getsize(model_path) / (1024 * 1024)
