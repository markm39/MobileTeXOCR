"""
Full End-to-End Handwritten LaTeX OCR Model

Combines vision encoder (FastViTHD or Perception Encoder) with unified decoder
to produce bounding boxes and LaTeX from handwritten math images.
"""

from typing import Optional, List, Tuple, Union, Dict, Any
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder.base import BaseEncoder, EncoderOutput
from models.encoder.fastvithd import FastViTHD, fastvithd_base, fastvithd_small
from models.encoder.perception_encoder import PerceptionEncoder, perception_encoder_tiny
from models.decoder.unified_decoder import UnifiedDecoder, create_decoder
from models.decoder.tokenizer import LaTeXTokenizer, TokenizerConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the full OCR model."""
    # Encoder settings
    encoder_type: str = "fastvithd"  # "fastvithd" or "perception"
    encoder_size: str = "base"  # "small", "base", "large" for fastvithd; "tiny", "small" for perception
    image_size: int = 384
    encoder_pretrained: bool = False

    # Decoder settings
    d_model: int = 384
    num_decoder_layers: int = 6
    num_heads: int = 8
    decoder_dropout: float = 0.1

    # Tokenizer settings
    num_location_bins: int = 1000
    max_seq_length: int = 512

    # Training settings
    freeze_encoder: bool = False
    label_smoothing: float = 0.1


@dataclass
class OCROutput:
    """Output from the OCR model."""
    # Raw outputs
    logits: Optional[torch.Tensor] = None  # [B, T, vocab_size] during training
    loss: Optional[torch.Tensor] = None

    # Decoded outputs (during inference)
    predictions: Optional[List[List[Tuple[Optional[Tuple[float, ...]], str]]]] = None
    token_ids: Optional[List[List[int]]] = None


class HandwrittenLaTeXOCR(nn.Module):
    """End-to-end model for handwritten LaTeX OCR.

    Architecture:
        Image -> Encoder -> Visual Tokens -> Decoder -> BBox + LaTeX

    Supports:
        - FastViTHD encoder (Apple CVPR 2025)
        - Perception Encoder (Meta NeurIPS 2025)
        - Unified text spotting decoder for joint detection + recognition
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config or ModelConfig()

        # Create tokenizer
        self.tokenizer = LaTeXTokenizer(TokenizerConfig(
            num_location_bins=self.config.num_location_bins,
            max_seq_length=self.config.max_seq_length,
        ))

        # Create encoder
        self.encoder = self._create_encoder()

        # Create decoder
        self.decoder = UnifiedDecoder(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.config.d_model,
            num_layers=self.config.num_decoder_layers,
            num_heads=self.config.num_heads,
            dim_feedforward=self.config.d_model * 4,
            dropout=self.config.decoder_dropout,
            max_seq_length=self.config.max_seq_length,
            num_location_bins=self.config.num_location_bins,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=self.config.label_smoothing,
        )

        # Optionally freeze encoder
        if self.config.freeze_encoder:
            self.encoder.freeze()

        logger.info(f"Created model with {self.count_parameters():,} parameters")
        logger.info(f"  Encoder: {self.encoder.count_parameters():,} parameters")
        logger.info(f"  Decoder: {sum(p.numel() for p in self.decoder.parameters()):,} parameters")

    def _create_encoder(self) -> BaseEncoder:
        """Create encoder based on config."""
        if self.config.encoder_type == "fastvithd":
            if self.config.encoder_size == "small":
                return fastvithd_small(
                    image_size=self.config.image_size,
                    output_dim=self.config.d_model,
                )
            elif self.config.encoder_size == "base":
                return fastvithd_base(
                    image_size=self.config.image_size,
                    output_dim=self.config.d_model,
                )
            else:
                raise ValueError(f"Unknown FastViTHD size: {self.config.encoder_size}")

        elif self.config.encoder_type == "perception":
            if self.config.encoder_size == "tiny":
                return perception_encoder_tiny(
                    image_size=self.config.image_size,
                    output_dim=self.config.d_model,
                )
            else:
                raise ValueError(f"Unknown Perception Encoder size: {self.config.encoder_size}")

        else:
            raise ValueError(f"Unknown encoder type: {self.config.encoder_type}")

    def forward(
        self,
        images: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
    ) -> OCROutput:
        """Forward pass.

        Args:
            images: Input images [B, 3, H, W], normalized to [0, 1]
            target_ids: Target token IDs [B, T] for teacher forcing (training)
            target_padding_mask: [B, T] True for padding positions

        Returns:
            OCROutput with logits and loss (training) or predictions (inference)
        """
        # Encode images
        encoder_output = self.encoder(images)
        encoder_features = encoder_output.features  # [B, N, D]

        if target_ids is not None:
            # Training mode: teacher forcing
            # Shift targets for autoregressive training
            input_ids = target_ids[:, :-1]  # Input: all but last
            label_ids = target_ids[:, 1:]   # Labels: all but first

            if target_padding_mask is not None:
                input_padding_mask = target_padding_mask[:, :-1]
            else:
                input_padding_mask = None

            # Forward through decoder
            logits = self.decoder(
                encoder_features,
                input_ids,
                encoder_padding_mask=encoder_output.attention_mask,
                target_padding_mask=input_padding_mask,
            )

            # Compute loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                label_ids.reshape(-1),
            )

            return OCROutput(logits=logits, loss=loss)

        else:
            # Inference mode: autoregressive generation
            token_ids = self.decoder.greedy_decode(
                encoder_features,
                self.tokenizer,
                max_length=self.config.max_seq_length,
                encoder_padding_mask=encoder_output.attention_mask,
            )

            # Decode to (bbox, latex) pairs
            predictions = [
                self.tokenizer.decode_sequence(ids)
                for ids in token_ids
            ]

            return OCROutput(predictions=predictions, token_ids=token_ids)

    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> OCROutput:
        """Generate predictions with sampling options.

        Args:
            images: Input images [B, 3, H, W]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold

        Returns:
            OCROutput with predictions
        """
        encoder_output = self.encoder(images)

        token_ids = self.decoder.generate(
            encoder_output.features,
            self.tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            encoder_padding_mask=encoder_output.attention_mask,
        )

        predictions = [
            self.tokenizer.decode_sequence(ids)
            for ids in token_ids
        ]

        return OCROutput(predictions=predictions, token_ids=token_ids)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoder(self):
        """Freeze encoder weights."""
        self.encoder.freeze()

    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        self.encoder.unfreeze()

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "HandwrittenLaTeXOCR":
        """Create model from config dictionary."""
        config = ModelConfig(**config_dict)
        return cls(config)

    def save_pretrained(self, path: str):
        """Save model weights and config."""
        import json
        from pathlib import Path

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), save_dir / "model.pt")

        # Save config
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save tokenizer vocab
        self.tokenizer.save_vocab(str(save_dir / "vocab.json"))

        logger.info(f"Saved model to {path}")

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "HandwrittenLaTeXOCR":
        """Load model from saved weights."""
        import json
        from pathlib import Path

        load_dir = Path(path)

        # Load config
        with open(load_dir / "config.json", "r") as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)
        config = ModelConfig(**config_dict)

        # Create model
        model = cls(config)

        # Load weights
        state_dict = torch.load(load_dir / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)

        logger.info(f"Loaded model from {path}")
        return model


def create_model(
    encoder_type: str = "fastvithd",
    encoder_size: str = "base",
    image_size: int = 384,
    d_model: int = 384,
    num_decoder_layers: int = 6,
    **kwargs
) -> HandwrittenLaTeXOCR:
    """Convenience function to create a model.

    Args:
        encoder_type: "fastvithd" or "perception"
        encoder_size: Size variant of encoder
        image_size: Input image size
        d_model: Model dimension
        num_decoder_layers: Number of decoder layers
        **kwargs: Additional config options

    Returns:
        Configured model
    """
    config = ModelConfig(
        encoder_type=encoder_type,
        encoder_size=encoder_size,
        image_size=image_size,
        d_model=d_model,
        num_decoder_layers=num_decoder_layers,
        **kwargs
    )
    return HandwrittenLaTeXOCR(config)


# Predefined model configurations
def latex_ocr_small() -> HandwrittenLaTeXOCR:
    """Small model (~100M params) for faster training/inference."""
    return create_model(
        encoder_type="fastvithd",
        encoder_size="small",
        d_model=256,
        num_decoder_layers=4,
    )


def latex_ocr_base() -> HandwrittenLaTeXOCR:
    """Base model (~300M params) for production use."""
    return create_model(
        encoder_type="fastvithd",
        encoder_size="base",
        d_model=384,
        num_decoder_layers=6,
    )


def latex_ocr_perception() -> HandwrittenLaTeXOCR:
    """Model with Perception Encoder (~200M params)."""
    return create_model(
        encoder_type="perception",
        encoder_size="tiny",
        d_model=384,
        num_decoder_layers=6,
    )
