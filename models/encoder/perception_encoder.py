"""
Perception Encoder Wrapper - Meta NeurIPS 2025 Oral

Wraps Meta's Perception Encoder for use in our unified OCR system.
Key insight: Extract features from INTERMEDIATE layers, not final layer.

Reference: https://github.com/facebookresearch/perception_models
Paper: "Perception Encoder" (NeurIPS 2025 Oral)
"""

from typing import Optional, List, Tuple
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder.base import BaseEncoder, EncoderOutput

logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x


class Attention(nn.Module):
    """Multi-head Self-Attention with qk-norm (Perception Encoder style)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # QK normalization (key for training stability in large models)
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 4)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block for Perception Encoder."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PerceptionEncoder(BaseEncoder):
    """Perception Encoder from Meta (NeurIPS 2025).

    Key innovation: Best visual representations come from intermediate layers,
    not the final layer. We provide hooks to extract from any layer.

    Available configurations:
    - PE-Core-T16: ~85M params, patch_size=16 (Tiny)
    - PE-Core-S16: ~300M params, patch_size=16 (Small)
    """

    def __init__(
        self,
        image_size: int = 384,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        output_dim: int = 384,
        extract_layers: Optional[List[int]] = None,
    ):
        """
        Args:
            image_size: Input image size
            patch_size: Patch size for tokenization
            in_chans: Number of input channels
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            qkv_bias: Use bias in qkv projection
            qk_norm: Use QK normalization (important for PE)
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            output_dim: Output dimension
            extract_layers: Which layers to extract features from (default: last 4)
        """
        super().__init__(
            embed_dim=embed_dim,
            output_dim=output_dim,
            image_size=image_size,
            patch_size=patch_size,
        )

        self.depth = depth
        self.num_heads = num_heads
        self.num_patches = (image_size // patch_size) ** 2

        # Default: extract from middle-to-late layers (empirically best)
        if extract_layers is None:
            # For 12-layer model, extract from layers 6, 8, 10, 11 (0-indexed)
            self.extract_layers = [depth - 6, depth - 4, depth - 2, depth - 1]
        else:
            self.extract_layers = extract_layers

        # Patch embedding
        self.patch_embed = PatchEmbed(image_size, patch_size, in_chans, embed_dim)

        # CLS token (not used in OCR, but kept for compatibility with pretrained weights)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)  # +1 for CLS
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias,
                qk_norm, drop_rate, attn_drop_rate
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Feature aggregation from multiple layers
        # Concatenate features from extract_layers and project
        self.feature_dim = embed_dim * len(self.extract_layers)
        self.feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ViT conventions."""
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize other weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Interpolate position encodings for different image sizes."""
        npatch = x.shape[1] - 1  # Exclude CLS token
        N = self.pos_embed.shape[1] - 1

        if npatch == N and h == w:
            return self.pos_embed

        # Interpolate patch positional embeddings
        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]

        dim = x.shape[-1]
        h0 = h // self.patch_size
        w0 = w // self.patch_size

        # Reshape to 2D grid
        sqrt_n = int(math.sqrt(N))
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_n, sqrt_n, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # Interpolate
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(h0, w0),
            mode='bicubic',
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat([class_pos_embed, patch_pos_embed], dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using multi-layer aggregation.

        This is the key innovation of Perception Encoder: we aggregate
        features from multiple intermediate layers rather than just the final.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Aggregated features [B, N, embed_dim]
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # [B, N, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, embed_dim]

        # Add positional encoding
        x = x + self._interpolate_pos_encoding(x, H, W)

        # Collect features from specified layers
        layer_features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.extract_layers:
                # Store features without CLS token
                layer_features.append(x[:, 1:])  # [B, N, embed_dim]

        # Concatenate multi-layer features
        if len(layer_features) > 1:
            combined = torch.cat(layer_features, dim=-1)  # [B, N, embed_dim * num_layers]
            features = self.feature_proj(combined)  # [B, N, embed_dim]
        else:
            features = self.norm(layer_features[0])

        return features

    def get_intermediate_features(
        self,
        x: torch.Tensor,
        layer_indices: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """Extract features from specified intermediate layers.

        Args:
            x: Input images [B, 3, H, W]
            layer_indices: Which layers to extract from (0-indexed)

        Returns:
            List of feature tensors from requested layers
        """
        if layer_indices is None:
            layer_indices = self.extract_layers

        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token and position encoding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self._interpolate_pos_encoding(x, H, W)

        # Collect features
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in layer_indices:
                # Return without CLS token
                features.append(self.norm(x[:, 1:].clone()))

        return features

    @classmethod
    def from_pretrained(cls, model_name: str = "PE-Core-T16") -> "PerceptionEncoder":
        """Load pretrained weights from Meta's model zoo.

        Note: Requires perception_models package installed.
        """
        try:
            from perception_models import create_model
            logger.info(f"Loading pretrained {model_name}...")
            pretrained = create_model(model_name, pretrained=True)

            # Create our wrapper with matching config
            model = cls(
                image_size=pretrained.image_size,
                patch_size=pretrained.patch_size,
                embed_dim=pretrained.embed_dim,
                depth=len(pretrained.blocks),
                num_heads=pretrained.num_heads,
            )

            # Copy weights
            model.load_state_dict(pretrained.state_dict(), strict=False)
            logger.info(f"Loaded pretrained {model_name}")
            return model
        except ImportError:
            logger.warning(
                "perception_models not installed. "
                "Install with: pip install perception-models"
            )
            raise


def perception_encoder_tiny(image_size: int = 384, output_dim: int = 384) -> PerceptionEncoder:
    """PE-Core-T16: ~85M parameters."""
    return PerceptionEncoder(
        image_size=image_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        output_dim=output_dim,
    )


def perception_encoder_small(image_size: int = 384, output_dim: int = 512) -> PerceptionEncoder:
    """PE-Core-S16: ~300M parameters."""
    return PerceptionEncoder(
        image_size=image_size,
        patch_size=16,
        embed_dim=768,
        depth=24,
        num_heads=12,
        output_dim=output_dim,
    )
