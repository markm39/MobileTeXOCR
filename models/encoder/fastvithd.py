"""
FastViTHD Encoder - Apple CVPR 2025

A hybrid conv-transformer architecture optimized for on-device VLM inference.
Key features:
- 3 convolutional stages with RepMixer blocks
- 2 transformer stages with MHSA
- 32x downsampling before attention for efficiency
- ~125M parameters

Reference: https://github.com/apple/ml-fastvlm
"""

from typing import Optional, List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder.base import BaseEncoder, EncoderOutput


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class RepMixerBlock(nn.Module):
    """Reparameterizable Mixer Block from FastViT.

    During training: depthwise conv + pointwise conv with skip connection
    During inference: can be fused into a single conv for speed
    """

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.dim = dim

        # Spatial mixing (depthwise conv)
        self.spatial_mix = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        )

        # Channel mixing (pointwise conv)
        self.channel_mix = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial mixing with residual
        x = x + self.spatial_mix(x)
        # Channel mixing with residual
        x = x + self.channel_mix(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard Multi-Head Self-Attention for transformer stages."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with MHSA and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvStage(nn.Module):
    """Convolutional stage with RepMixer blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        downsample: bool = True,
    ):
        super().__init__()

        layers = []

        # Downsampling at start of stage
        if downsample:
            layers.append(ConvBNAct(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        else:
            if in_channels != out_channels:
                layers.append(ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1, padding=0))

        # RepMixer blocks
        for _ in range(num_blocks):
            layers.append(RepMixerBlock(out_channels))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class TransformerStage(nn.Module):
    """Transformer stage operating on flattened feature maps."""

    def __init__(
        self,
        dim: int,
        num_blocks: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, drop)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class FastViTHD(BaseEncoder):
    """FastViTHD: Hybrid Encoder from Apple's FastVLM (CVPR 2025).

    Architecture:
    - Stem: 7x7 conv with stride 2 (2x downsample)
    - Stage 1: RepMixer blocks, stride 2 (4x cumulative)
    - Stage 2: RepMixer blocks, stride 2 (8x cumulative)
    - Stage 3: RepMixer blocks, stride 2 (16x cumulative)
    - Stage 4: Transformer blocks, stride 2 (32x cumulative)
    - Stage 5: Transformer blocks, no downsample (32x cumulative)

    Total downsampling: 32x
    For 384x384 input -> 12x12 = 144 tokens
    """

    def __init__(
        self,
        image_size: int = 384,
        output_dim: int = 384,
        # Stage configurations: (out_channels, num_blocks)
        conv_stages: List[Tuple[int, int]] = None,
        transformer_stages: List[Tuple[int, int]] = None,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ):
        """
        Args:
            image_size: Input image size (square)
            output_dim: Output embedding dimension
            conv_stages: List of (channels, num_blocks) for conv stages
            transformer_stages: List of (channels, num_blocks) for transformer stages
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            drop_rate: Dropout rate
        """
        # Default configuration for ~125M params
        if conv_stages is None:
            conv_stages = [
                (64, 2),    # Stage 1: 64 channels, 2 blocks
                (128, 2),   # Stage 2: 128 channels, 2 blocks
                (256, 6),   # Stage 3: 256 channels, 6 blocks
            ]
        if transformer_stages is None:
            transformer_stages = [
                (512, 6),   # Stage 4: 512 channels, 6 transformer blocks
                (512, 6),   # Stage 5: 512 channels, 6 transformer blocks
            ]

        embed_dim = transformer_stages[-1][0]
        super().__init__(
            embed_dim=embed_dim,
            output_dim=output_dim,
            image_size=image_size,
            patch_size=32,  # Effective 32x downsampling
        )

        self.conv_stages_config = conv_stages
        self.transformer_stages_config = transformer_stages

        # Stem: 7x7 conv with stride 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, conv_stages[0][0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(conv_stages[0][0]),
            nn.GELU(),
        )

        # Build convolutional stages
        self.conv_stages = nn.ModuleList()
        in_ch = conv_stages[0][0]
        for i, (out_ch, num_blocks) in enumerate(conv_stages):
            self.conv_stages.append(
                ConvStage(in_ch, out_ch, num_blocks, downsample=(i > 0))
            )
            in_ch = out_ch

        # Transition from conv to transformer
        # One more downsample to get to 32x total
        self.conv_to_transformer = ConvBNAct(
            conv_stages[-1][0],
            transformer_stages[0][0],
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Build transformer stages
        self.transformer_stages = nn.ModuleList()
        for i, (dim, num_blocks) in enumerate(transformer_stages):
            self.transformer_stages.append(
                TransformerStage(dim, num_blocks, num_heads, mlp_ratio, drop_rate)
            )

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Features [B, N, embed_dim] where N = (H/32) * (W/32)
        """
        # Stem
        x = self.stem(x)  # [B, 64, H/2, W/2]

        # Convolutional stages
        for stage in self.conv_stages:
            x = stage(x)  # Progressive downsampling

        # Transition to transformer (includes final downsample)
        x = self.conv_to_transformer(x)  # [B, 512, H/32, W/32]

        # Reshape for transformer: [B, C, H, W] -> [B, N, C]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Transformer stages
        for stage in self.transformer_stages:
            x = stage(x)

        # Final normalization
        x = self.norm(x)

        return x

    def get_intermediate_features(
        self,
        x: torch.Tensor,
        layer_indices: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """Extract features from intermediate layers.

        Args:
            x: Input images [B, 3, H, W]
            layer_indices: Which stages to extract from (0-4 for 5 stages)
                          If None, returns all stage outputs

        Returns:
            List of feature tensors from requested stages
        """
        if layer_indices is None:
            layer_indices = list(range(len(self.conv_stages) + len(self.transformer_stages)))

        features = []
        stage_idx = 0

        # Stem
        x = self.stem(x)

        # Convolutional stages
        for i, stage in enumerate(self.conv_stages):
            x = stage(x)
            if stage_idx in layer_indices:
                features.append(x.clone())
            stage_idx += 1

        # Transition
        x = self.conv_to_transformer(x)

        # Reshape for transformer
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        # Transformer stages
        for stage in self.transformer_stages:
            x = stage(x)
            if stage_idx in layer_indices:
                features.append(self.norm(x.clone()))
            stage_idx += 1

        return features


def fastvithd_small(image_size: int = 384, output_dim: int = 384) -> FastViTHD:
    """FastViTHD-Small: ~50M parameters."""
    return FastViTHD(
        image_size=image_size,
        output_dim=output_dim,
        conv_stages=[
            (48, 2),
            (96, 2),
            (192, 4),
        ],
        transformer_stages=[
            (384, 4),
            (384, 4),
        ],
        num_heads=8,
    )


def fastvithd_base(image_size: int = 384, output_dim: int = 384) -> FastViTHD:
    """FastViTHD-Base: ~125M parameters (default)."""
    return FastViTHD(
        image_size=image_size,
        output_dim=output_dim,
        conv_stages=[
            (64, 2),
            (128, 2),
            (256, 6),
        ],
        transformer_stages=[
            (512, 6),
            (512, 6),
        ],
        num_heads=12,
    )


def fastvithd_large(image_size: int = 384, output_dim: int = 512) -> FastViTHD:
    """FastViTHD-Large: ~200M parameters."""
    return FastViTHD(
        image_size=image_size,
        output_dim=output_dim,
        conv_stages=[
            (96, 2),
            (192, 3),
            (384, 8),
        ],
        transformer_stages=[
            (768, 8),
            (768, 8),
        ],
        num_heads=16,
    )
