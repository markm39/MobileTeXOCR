# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SHViT-style encoder for Handwritten Mathematical Expression (HME) Recognition.

Based on: "SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design" (CVPR 2024)
Adapted for:
- Grayscale input (1 channel for handwriting)
- 2D spatial feature output (for sequence-to-sequence decoding)
- CoreML-safe operations (no einsum, explicit matmul)

Key innovations from SHViT:
1. Single-head self-attention reduces memory bandwidth bottleneck
2. Partial attention: only subset of channels go through attention
3. Conv-BN fusion for inference optimization
4. Early stages use conv only (no attention)
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant

__all__ = ["HME_SHViT", "HME_SHViT_Tiny", "HME_SHViT_Small", "HME_SHViT_Base"]

trunc_normal_ = TruncatedNormal(std=0.02)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


class GroupNorm1(nn.GroupNorm):
    """
    Group Normalization with 1 group (equivalent to Layer Norm for 2D).
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Conv2d_BN(nn.Layer):
    """
    Conv2d + BatchNorm2d with optional fusion for inference.
    CoreML-compatible: uses standard conv and bn operations.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bn_weight_init=1.0,
    ):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias_attr=False,
        )
        self.bn = nn.BatchNorm2D(out_channels)

        # Initialize BN
        ones_(self.bn.weight)
        zeros_(self.bn.bias)
        if bn_weight_init != 1.0:
            self.bn.weight.set_value(
                paddle.full_like(self.bn.weight, bn_weight_init)
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    @paddle.no_grad()
    def fuse(self):
        """Fuse Conv and BN for inference (CoreML optimization)."""
        conv, bn = self.conv, self.bn
        w = bn.weight / paddle.sqrt(bn._variance + bn._epsilon)
        w = conv.weight * w.reshape([-1, 1, 1, 1])
        b = bn.bias - bn._mean * bn.weight / paddle.sqrt(bn._variance + bn._epsilon)

        fused_conv = nn.Conv2D(
            conv._in_channels,
            conv._out_channels,
            conv._kernel_size,
            conv._stride,
            conv._padding,
            conv._dilation,
            conv._groups,
            bias_attr=True,
        )
        fused_conv.weight.set_value(w)
        fused_conv.bias.set_value(b)
        return fused_conv


class BN_Linear(nn.Layer):
    """BatchNorm1D + Linear with optional fusion."""

    def __init__(self, in_features, out_features, bias=True, std=0.02):
        super().__init__()
        self.bn = nn.BatchNorm1D(in_features)
        self.linear = nn.Linear(in_features, out_features, bias_attr=bias)
        trunc_normal_(self.linear.weight)
        if bias:
            zeros_(self.linear.bias)

    def forward(self, x):
        x = self.bn(x)
        x = self.linear(x)
        return x


class SqueezeExcite(nn.Layer):
    """Squeeze-and-Excitation module."""

    def __init__(self, channels, rd_ratio=0.25):
        super().__init__()
        rd_channels = int(channels * rd_ratio)
        self.fc1 = nn.Conv2D(channels, rd_channels, 1, bias_attr=True)
        self.fc2 = nn.Conv2D(rd_channels, channels, 1, bias_attr=True)

    def forward(self, x):
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self.fc1(x_se)
        x_se = F.relu(x_se)
        x_se = self.fc2(x_se)
        x_se = F.sigmoid(x_se)
        return x * x_se


class PatchMerging(nn.Layer):
    """Downsample spatial dimensions by 2x while increasing channels."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        hid_dim = int(in_dim * 4)
        self.conv1 = Conv2d_BN(in_dim, hid_dim, 1, 1, 0)
        self.act = nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, 0.25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.se(x)
        x = self.conv3(x)
        return x


class Residual(nn.Layer):
    """Residual connection with optional drop path."""

    def __init__(self, module, drop=0.0):
        super().__init__()
        self.m = module
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            # Stochastic depth
            keep_prob = 1 - self.drop
            shape = (paddle.shape(x)[0],) + (1,) * (x.ndim - 1)
            random_tensor = paddle.rand(shape, dtype=x.dtype)
            random_tensor = paddle.floor(random_tensor + keep_prob)
            return x + self.m(x) * random_tensor / keep_prob
        else:
            return x + self.m(x)


class FFN(nn.Layer):
    """Feed-Forward Network with expansion ratio 2."""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.pw1 = Conv2d_BN(dim, hidden_dim)
        self.act = nn.ReLU()
        self.pw2 = Conv2d_BN(hidden_dim, dim, bn_weight_init=0.0)

    def forward(self, x):
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x


class SHSA(nn.Layer):
    """
    Single-Head Self-Attention (SHSA) - Key innovation from SHViT.
    
    Memory efficient design:
    - Only `pdim` channels go through attention (partial attention)
    - Single head instead of multi-head
    - Uses explicit matmul instead of einsum for CoreML compatibility
    """

    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = GroupNorm1(pdim)
        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = nn.Sequential(
            nn.ReLU(),
            Conv2d_BN(dim, dim, bn_weight_init=0.0),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Split channels: only pdim goes through attention
        x1 = x[:, :self.pdim, :, :]  # [B, pdim, H, W]
        x2 = x[:, self.pdim:, :, :]  # [B, dim-pdim, H, W]

        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)

        # Split into q, k, v
        q = qkv[:, :self.qk_dim, :, :]  # [B, qk_dim, H, W]
        k = qkv[:, self.qk_dim:self.qk_dim * 2, :, :]  # [B, qk_dim, H, W]
        v = qkv[:, self.qk_dim * 2:, :, :]  # [B, pdim, H, W]

        # Flatten spatial dimensions
        q = q.flatten(2)  # [B, qk_dim, H*W]
        k = k.flatten(2)  # [B, qk_dim, H*W]
        v = v.flatten(2)  # [B, pdim, H*W]

        # CoreML-safe attention computation (no einsum)
        # attn = softmax(q^T @ k * scale)
        attn = paddle.matmul(q.transpose([0, 2, 1]), k) * self.scale  # [B, H*W, H*W]
        attn = F.softmax(attn, axis=-1)

        # out = v @ attn^T
        x1 = paddle.matmul(v, attn.transpose([0, 2, 1]))  # [B, pdim, H*W]
        x1 = x1.reshape([B, self.pdim, H, W])

        # Merge channels back
        x = paddle.concat([x1, x2], axis=1)
        x = self.proj(x)

        return x


class BasicBlock(nn.Layer):
    """
    Basic block for SHViT.
    
    Args:
        dim: Channel dimension
        qk_dim: Query/Key dimension for attention
        pdim: Partial dimension (channels going through attention)
        block_type: "s" for attention stages, "i" for conv-only stages
        drop_path: Drop path rate
    """

    def __init__(self, dim, qk_dim, pdim, block_type="s", drop_path=0.0):
        super().__init__()
        # Depthwise conv
        self.conv = Residual(
            Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0.0),
            drop=drop_path,
        )

        # Attention or identity based on block type
        if block_type == "s":
            # Later stages: use SHSA
            self.mixer = Residual(SHSA(dim, qk_dim, pdim), drop=drop_path)
        else:
            # Early stages: no attention (conv only)
            self.mixer = nn.Identity()

        # FFN
        self.ffn = Residual(FFN(dim, dim * 2), drop=drop_path)

    def forward(self, x):
        x = self.conv(x)
        x = self.mixer(x)
        x = self.ffn(x)
        return x


class HME_SHViT(nn.Layer):
    """
    SHViT-style encoder for Handwritten Mathematical Expression Recognition.
    
    Outputs 2D spatial features suitable for transformer decoder.
    
    Args:
        in_channels: Input channels (1 for grayscale, 3 for RGB)
        embed_dim: List of embedding dimensions for each stage
        partial_dim: List of partial dimensions for SHSA
        qk_dim: List of query/key dimensions for SHSA
        depth: List of block depths for each stage
        block_types: List of block types ("s" for attention, "i" for conv-only)
        out_channels: Output feature dimension (if None, uses embed_dim[-1])
        downsample_stages: Which stages to add downsampling after (default: first 2)
        drop_path_rate: Stochastic depth rate
    """

    def __init__(
        self,
        in_channels=1,
        embed_dim=[128, 256, 384],
        partial_dim=[32, 64, 96],
        qk_dim=[16, 16, 16],
        depth=[1, 2, 3],
        block_types=["i", "s", "s"],
        out_channels=None,
        downsample_stages=[0, 1],
        drop_path_rate=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or embed_dim[-1]
        self.num_stages = len(embed_dim)

        # Patch embedding: 4 conv layers with stride 2 each -> 16x downsample
        self.patch_embed = nn.Sequential(
            Conv2d_BN(in_channels, embed_dim[0] // 8, 3, 2, 1),
            nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1),
            nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1),
            nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1),
        )

        # Calculate drop path rates for each block
        total_depth = sum(depth)
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, total_depth)]
        dpr_idx = 0

        # Build stages
        self.stages = nn.LayerList()
        for i in range(self.num_stages):
            stage_layers = []

            # Blocks for this stage
            for _ in range(depth[i]):
                stage_layers.append(
                    BasicBlock(
                        dim=embed_dim[i],
                        qk_dim=qk_dim[i],
                        pdim=partial_dim[i],
                        block_type=block_types[i],
                        drop_path=dpr[dpr_idx],
                    )
                )
                dpr_idx += 1

            self.stages.append(nn.Sequential(*stage_layers))

            # Add downsampling after specified stages (except the last)
            if i in downsample_stages and i < self.num_stages - 1:
                # Add transition blocks before and after patch merging
                transition = nn.Sequential(
                    Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i])),
                    Residual(FFN(embed_dim[i], embed_dim[i] * 2)),
                    PatchMerging(embed_dim[i], embed_dim[i + 1]),
                    Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1])),
                    Residual(FFN(embed_dim[i + 1], embed_dim[i + 1] * 2)),
                )
                self.stages.append(transition)

        # Output projection if needed
        if self.out_channels != embed_dim[-1]:
            self.output_proj = nn.Conv2D(embed_dim[-1], self.out_channels, 1)
        else:
            self.output_proj = nn.Identity()

        # Final layer norm
        self.norm = nn.LayerNorm(self.out_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.Conv2D):
                fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                fan_out //= m._groups
                m.weight.set_value(
                    paddle.randn(m.weight.shape) * (2.0 / fan_out) ** 0.5
                )
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            features: [B, out_channels, H', W'] spatial features
            or [B, H'*W', out_channels] if flattened for transformer
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Process through all stages
        for stage in self.stages:
            x = stage(x)

        # Output projection
        x = self.output_proj(x)

        # Reshape to [B, H, W, C] for LayerNorm, then back
        B, C, H, W = x.shape
        x = x.transpose([0, 2, 3, 1])  # [B, H, W, C]
        x = self.norm(x)
        x = x.transpose([0, 3, 1, 2])  # [B, C, H, W]

        return x

    def forward_features_2d(self, x):
        """
        Returns 2D features in [B, H, W, C] format for decoder with 2D position encoding.
        """
        x = self.forward(x)
        x = x.transpose([0, 2, 3, 1])  # [B, H, W, C]
        return x


def HME_SHViT_Tiny(in_channels=1, out_channels=256, **kwargs):
    """
    Tiny variant for ultra-light deployment.
    Target: <10MB, <50ms on iPhone 14
    """
    return HME_SHViT(
        in_channels=in_channels,
        embed_dim=[64, 128, 192],
        partial_dim=[16, 32, 48],
        qk_dim=[8, 8, 8],
        depth=[1, 2, 2],
        block_types=["i", "s", "s"],
        out_channels=out_channels,
        downsample_stages=[0, 1],
        drop_path_rate=0.0,
        **kwargs,
    )


def HME_SHViT_Small(in_channels=1, out_channels=256, **kwargs):
    """
    Small variant for balanced accuracy/speed.
    Target: 15-30MB, <80ms on iPhone 14
    """
    return HME_SHViT(
        in_channels=in_channels,
        embed_dim=[128, 256, 384],
        partial_dim=[32, 64, 96],
        qk_dim=[16, 16, 16],
        depth=[1, 3, 4],
        block_types=["i", "s", "s"],
        out_channels=out_channels,
        downsample_stages=[0, 1],
        drop_path_rate=0.1,
        **kwargs,
    )


def HME_SHViT_Base(in_channels=1, out_channels=256, **kwargs):
    """
    Base variant for maximum accuracy.
    Target: 50-80MB, <150ms on iPhone 14
    """
    return HME_SHViT(
        in_channels=in_channels,
        embed_dim=[128, 256, 512],
        partial_dim=[32, 64, 128],
        qk_dim=[16, 16, 16],
        depth=[1, 4, 6],
        block_types=["i", "s", "s"],
        out_channels=out_channels,
        downsample_stages=[0, 1],
        drop_path_rate=0.1,
        **kwargs,
    )

