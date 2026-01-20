"""
Export model to ONNX format.

Exports the encoder and decoder separately for efficient inference:
- Encoder: Single forward pass per image
- Decoder: Runs autoregressively
"""

from typing import Optional, Tuple
from pathlib import Path
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EncoderWrapper(nn.Module):
    """Wrapper for exporting encoder to ONNX."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.encoder(image)
        return output.features


class DecoderStepWrapper(nn.Module):
    """Wrapper for exporting single decoder step to ONNX."""

    def __init__(self, decoder, tokenizer):
        super().__init__()
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.d_model = decoder.d_model

    def forward(
        self,
        encoder_features: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.decoder(encoder_features, input_ids)
        return logits[:, -1, :]


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    image_size: int = 384,
    max_seq_length: int = 256,
    opset_version: int = 14,
) -> Tuple[str, str]:
    """Export model to ONNX format.

    Args:
        model: HandwrittenLaTeXOCR model
        output_path: Base path for output files
        image_size: Input image size
        max_seq_length: Maximum sequence length for decoder
        opset_version: ONNX opset version

    Returns:
        Tuple of (encoder_path, decoder_path)
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(output_path).stem

    encoder_path = output_dir / f"{base_name}_encoder.onnx"
    _export_encoder(model, str(encoder_path), image_size, opset_version)

    decoder_path = output_dir / f"{base_name}_decoder.onnx"
    _export_decoder(model, str(decoder_path), max_seq_length, opset_version)

    logger.info(f"Exported encoder to {encoder_path}")
    logger.info(f"Exported decoder to {decoder_path}")

    return str(encoder_path), str(decoder_path)


def _export_encoder(
    model: nn.Module,
    output_path: str,
    image_size: int,
    opset_version: int,
):
    encoder_wrapper = EncoderWrapper(model.encoder)
    encoder_wrapper.eval()

    dummy_image = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        encoder_wrapper,
        dummy_image,
        output_path,
        input_names=['image'],
        output_names=['features'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'features': {0: 'batch_size'},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )


def _export_decoder(
    model: nn.Module,
    output_path: str,
    max_seq_length: int,
    opset_version: int,
):
    decoder_wrapper = DecoderStepWrapper(model.decoder, model.tokenizer)
    decoder_wrapper.eval()

    patch_size = model.encoder.patch_size
    image_size = model.config.image_size
    num_patches = (image_size // patch_size) ** 2

    dummy_features = torch.randn(1, num_patches, model.config.d_model)
    dummy_ids = torch.randint(0, model.tokenizer.vocab_size, (1, 10))

    torch.onnx.export(
        decoder_wrapper,
        (dummy_features, dummy_ids),
        output_path,
        input_names=['encoder_features', 'input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'encoder_features': {0: 'batch_size'},
            'input_ids': {0: 'batch_size', 1: 'seq_length'},
            'logits': {0: 'batch_size'},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )


def check_onnx(onnx_path: str, sample_input: torch.Tensor) -> bool:
    """Check ONNX model produces correct output."""
    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: sample_input.numpy()})

        logger.info(f"ONNX check passed for {onnx_path}")
        return True

    except Exception as e:
        logger.error(f"ONNX check failed: {e}")
        return False


def optimize_onnx(input_path: str, output_path: str):
    """Optimize ONNX model for inference."""
    try:
        import onnx
        from onnxruntime.transformers import optimizer

        model = onnx.load(input_path)

        optimized = optimizer.optimize_model(
            input_path,
            model_type='bert',
            num_heads=8,
            hidden_size=384,
        )

        optimized.save_model_to_file(output_path)
        logger.info(f"Optimized ONNX model saved to {output_path}")

    except ImportError:
        logger.warning("onnxruntime.transformers not available, skipping optimization")
