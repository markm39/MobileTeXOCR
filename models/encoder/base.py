"""
Abstract base class for vision encoders.

All encoders must output visual tokens in a consistent format for the decoder.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EncoderOutput:
    """Standard output format for all encoders.

    Attributes:
        features: Visual tokens of shape [B, N, D] where N = H*W / patch_size^2
        spatial_shape: Tuple of (H, W) representing the spatial dimensions before flattening
        attention_mask: Optional mask for padded regions, shape [B, N]
    """
    features: torch.Tensor
    spatial_shape: Tuple[int, int]
    attention_mask: Optional[torch.Tensor] = None


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for vision encoders.

    This defines the interface that both FastViTHD and Perception Encoder must implement.
    The key contract is that forward() returns EncoderOutput with consistent shapes.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        output_dim: int = 384,
        image_size: int = 384,
        patch_size: int = 16,
    ):
        """
        Args:
            embed_dim: Internal embedding dimension of the encoder
            output_dim: Output dimension (projected if different from embed_dim)
            image_size: Expected input image size (square)
            patch_size: Effective patch/downsampling size
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.image_size = image_size
        self.patch_size = patch_size

        # Projection to common output dimension if needed
        if embed_dim != output_dim:
            self.output_proj = nn.Linear(embed_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

    @property
    def num_patches(self) -> int:
        """Number of patches for a square image of size image_size."""
        return (self.image_size // self.patch_size) ** 2

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Input images of shape [B, 3, H, W], normalized to [-1, 1] or [0, 1]

        Returns:
            Features of shape [B, N, embed_dim] where N depends on input size
        """
        pass

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """Forward pass with standardized output format.

        Args:
            x: Input images of shape [B, 3, H, W]

        Returns:
            EncoderOutput with projected features and spatial information
        """
        B, C, H, W = x.shape

        # Get raw features from encoder
        features = self.encode(x)  # [B, N, embed_dim]

        # Project to output dimension
        features = self.output_proj(features)  # [B, N, output_dim]

        # Compute spatial shape (assumes square patches)
        h = H // self.patch_size
        w = W // self.patch_size

        return EncoderOutput(
            features=features,
            spatial_shape=(h, w),
            attention_mask=None,
        )

    @abstractmethod
    def get_intermediate_features(
        self,
        x: torch.Tensor,
        layer_indices: Optional[list] = None
    ) -> list:
        """Extract features from intermediate layers.

        This is particularly important for Perception Encoder, which achieves
        best results from intermediate rather than final layers.

        Args:
            x: Input images of shape [B, 3, H, W]
            layer_indices: Which layers to extract from (encoder-specific)

        Returns:
            List of feature tensors from requested layers
        """
        pass

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self):
        """Freeze all encoder parameters for fine-tuning decoder only."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True
