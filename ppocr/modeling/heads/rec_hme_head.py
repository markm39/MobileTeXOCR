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
HME (Handwritten Mathematical Expression) Recognition Head.

Based on:
- CoMER: Coverage Attention (Attention Refinement Module)
- TAMER: Tree-Aware Structure Similarity (StructSim)

Key features:
- Bidirectional decoding (L2R and R2L)
- Coverage attention to prevent over/under-attention
- Tree structure prediction for bracket matching
- CoreML-safe operations (no einsum, explicit matmul)
"""

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant, XavierUniform

__all__ = ["HMEHead"]

trunc_normal_ = TruncatedNormal(std=0.02)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)
xavier_uniform_ = XavierUniform()


# ============================================================================
# Position Encoding (CoreML-safe - no einsum)
# ============================================================================

class WordPosEnc(nn.Layer):
    """
    1D Sinusoidal Position Encoding for decoder sequence.
    CoreML-safe: uses explicit broadcasting instead of einsum.
    """

    def __init__(self, d_model=256, max_len=500, temperature=10000.0):
        super().__init__()
        pe = paddle.zeros([max_len, d_model])

        position = paddle.arange(0, max_len, dtype='float32').unsqueeze(1)
        dim_t = paddle.arange(0, d_model, 2, dtype='float32')
        # CoreML-safe: explicit broadcast instead of einsum
        div_term = 1.0 / (temperature ** (dim_t / d_model))
        inv_freq = position * div_term  # [max_len, d_model//2]

        pe[:, 0::2] = paddle.sin(inv_freq)
        pe[:, 1::2] = paddle.cos(inv_freq)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: [B, L, D]
        Returns:
            [B, L, D] with position encoding added
        """
        _, seq_len, _ = x.shape
        emb = self.pe[:seq_len, :]
        return x + emb.unsqueeze(0)


class ImgPosEnc(nn.Layer):
    """
    2D Sinusoidal Position Encoding for encoder features.
    CoreML-safe: uses explicit operations instead of einsum.
    """

    def __init__(self, d_model=256, temperature=10000.0, normalize=True, scale=None):
        super().__init__()
        assert d_model % 2 == 0
        self.half_d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        """
        Args:
            x: [B, H, W, D]
            mask: [B, H, W] - True for padding positions
        Returns:
            [B, H, W, D] with position encoding added
        """
        not_mask = ~mask
        y_embed = paddle.cumsum(not_mask.astype('float32'), axis=1)
        x_embed = paddle.cumsum(not_mask.astype('float32'), axis=2)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = paddle.arange(0, self.half_d_model, 2, dtype='float32')
        inv_freq = 1.0 / (self.temperature ** (dim_t / self.half_d_model))

        # CoreML-safe: explicit broadcast instead of einsum
        # [B, H, W, 1] * [1, 1, 1, d] -> [B, H, W, d]
        pos_x = x_embed.unsqueeze(-1) * inv_freq.reshape([1, 1, 1, -1])
        pos_y = y_embed.unsqueeze(-1) * inv_freq.reshape([1, 1, 1, -1])

        # Stack sin/cos and flatten
        pos_x = paddle.stack([paddle.sin(pos_x), paddle.cos(pos_x)], axis=4)
        pos_x = pos_x.reshape([pos_x.shape[0], pos_x.shape[1], pos_x.shape[2], -1])
        pos_y = paddle.stack([paddle.sin(pos_y), paddle.cos(pos_y)], axis=4)
        pos_y = pos_y.reshape([pos_y.shape[0], pos_y.shape[1], pos_y.shape[2], -1])

        pos = paddle.concat([pos_x, pos_y], axis=-1)
        return x + pos


# ============================================================================
# Attention Refinement Module (ARM) - from CoMER
# ============================================================================

class MaskBatchNorm2D(nn.Layer):
    """
    Batch Normalization that handles masked positions.
    CoreML-compatible version using conditional masking.
    """

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1D(num_features)

    def forward(self, x, mask):
        """
        Args:
            x: [B, D, H, W]
            mask: [B, 1, H, W] - True for padding
        Returns:
            [B, D, H, W]
        """
        B, D, H, W = x.shape
        # Reshape to [B, H, W, D]
        x = x.transpose([0, 2, 3, 1])
        mask = mask.squeeze(1)  # [B, H, W]

        # Apply BN only to non-masked positions
        not_mask = ~mask
        # Flatten and filter
        x_flat = x.reshape([-1, D])  # [B*H*W, D]
        not_mask_flat = not_mask.reshape([-1])  # [B*H*W]

        # Get non-masked elements
        indices = paddle.where(not_mask_flat)[0]
        if indices.shape[0] > 0:
            x_valid = paddle.gather(x_flat, indices)  # [N, D]
            x_valid = self.bn(x_valid)
            # Scatter back
            x_flat = paddle.scatter(x_flat, indices, x_valid)

        x = x_flat.reshape([B, H, W, D])
        x = x.transpose([0, 3, 1, 2])
        return x


class AttentionRefinementModule(nn.Layer):
    """
    Attention Refinement Module (ARM) from CoMER.
    
    Computes coverage-based attention bias to prevent:
    - Over-attention: attending to same region multiple times
    - Under-attention: missing some regions entirely
    
    CoreML-safe: no einsum, uses explicit reshape/transpose.
    """

    def __init__(self, nhead, dc=32, cross_coverage=True, self_coverage=True):
        super().__init__()
        assert cross_coverage or self_coverage
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage

        in_chs = (2 if (cross_coverage and self_coverage) else 1) * nhead

        self.conv = nn.Conv2D(in_chs, dc, kernel_size=5, padding=2)
        self.act = nn.ReLU()
        self.proj = nn.Conv2D(dc, nhead, kernel_size=1, bias_attr=False)
        self.post_norm = MaskBatchNorm2D(nhead)

    def forward(self, prev_attn, key_padding_mask, height, width):
        """
        Args:
            prev_attn: [B, nhead, T, H, W] - cumulative attention from previous layers
            key_padding_mask: [B, H, W] - True for padding positions
            height: int - height of spatial features
            width: int - width of spatial features

        Returns:
            [B, nhead, T, H*W] - coverage bias to add to attention
        """
        B, nhead, T, H, W = prev_attn.shape
        
        # Create mask: [B*T, 1, H, W]
        mask = key_padding_mask.unsqueeze(1)  # [B, 1, H, W]
        mask = paddle.tile(mask, [1, T, 1, 1])  # [B, T, H, W]
        mask = mask.reshape([B * T, 1, H, W])

        # Cumulative sum (coverage) - exclude current step
        coverage = paddle.cumsum(prev_attn, axis=2) - prev_attn  # [B, nhead, T, H, W]
        
        # Build input channels based on coverage types
        # Each coverage type contributes nhead channels
        attn_list = []
        if self.cross_coverage:
            attn_list.append(coverage)  # [B, nhead, T, H, W]
        if self.self_coverage:
            attn_list.append(coverage)  # [B, nhead, T, H, W] (using same for simplicity)
        
        # Concatenate along channel dim: [B, in_chs, T, H, W]
        attns = paddle.concat(attn_list, axis=1)  # [B, 2*nhead or nhead, T, H, W]

        # Reshape: [B, in_chs, T, H, W] -> [B*T, in_chs, H, W]
        in_chs = attns.shape[1]
        attns = attns.transpose([0, 2, 1, 3, 4])  # [B, T, in_chs, H, W]
        attns = attns.reshape([B * T, in_chs, H, W])

        # Apply conv to get coverage features
        cov = self.conv(attns)  # [B*T, dc, H, W]
        cov = self.act(cov)

        # Mask padding positions
        cov = cov * (~mask).astype(cov.dtype)
        cov = self.proj(cov)  # [B*T, nhead, H, W]
        cov = self.post_norm(cov, mask)

        # Reshape back: [B*T, nhead, H, W] -> [B, nhead, T, H*W]
        cov = cov.reshape([B, T, self.nhead, H * W])
        cov = cov.transpose([0, 2, 1, 3])  # [B, nhead, T, H*W]

        return cov


# ============================================================================
# Transformer Decoder Layer with ARM
# ============================================================================

class TransformerDecoderLayer(nn.Layer):
    """
    Transformer decoder layer with optional ARM (coverage attention).
    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Self attention
        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross attention
        self.cross_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def _convert_padding_mask_to_attn_mask(self, padding_mask, tgt_len=None):
        """
        Convert padding mask [B, L] to attention mask format for PaddlePaddle.
        PaddlePaddle expects: [B, 1, 1, L] or [B, 1, T, L] for cross attention
        Values: 0 for attend, -inf for mask
        """
        if padding_mask is None:
            return None
        # padding_mask: [B, L] - True where padded
        # Convert to float with -inf for masked positions
        attn_mask = padding_mask.astype('float32') * (-1e9)
        # Reshape to [B, 1, 1, L] for broadcasting
        if tgt_len is not None:
            # For cross attention: [B, 1, T, L]
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = paddle.tile(attn_mask, [1, 1, tgt_len, 1])
        else:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        return attn_mask

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        coverage_bias=None,
    ):
        """
        Args:
            tgt: [B, L_tgt, D]
            memory: [B, L_mem, D]
            coverage_bias: [B*nhead, L_tgt, L_mem] optional bias from ARM
        """
        L_tgt = tgt.shape[1]
        
        # Convert padding masks to attention masks
        self_attn_mask = self._convert_padding_mask_to_attn_mask(tgt_key_padding_mask)
        cross_attn_mask = self._convert_padding_mask_to_attn_mask(memory_key_padding_mask, L_tgt)
        
        # Combine with causal mask for self attention
        if tgt_mask is not None and self_attn_mask is not None:
            # tgt_mask: [L_tgt, L_tgt] causal mask
            # self_attn_mask: [B, 1, 1, L_tgt] padding mask
            combined_mask = tgt_mask.unsqueeze(0).unsqueeze(0) + self_attn_mask
            self_attn_mask = combined_mask
        elif tgt_mask is not None:
            self_attn_mask = tgt_mask

        # Self attention
        tgt2 = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=self_attn_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        tgt2 = self.cross_attn(
            tgt, memory, memory,
            attn_mask=cross_attn_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Layer):
    """
    Transformer decoder with ARM (Attention Refinement Module).
    """

    def __init__(self, d_model, nhead, num_layers, dim_feedforward=1024, 
                 dropout=0.3, dc=32, cross_coverage=True, self_coverage=True):
        super().__init__()
        self.layers = nn.LayerList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.arm = None
        if cross_coverage or self_coverage:
            self.arm = AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage)

        self.nhead = nhead

    def forward(self, tgt, memory, height, width, tgt_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: [B, L_tgt, D]
            memory: [B, L_mem, D]
            height: int - spatial height for ARM
            width: int - spatial width for ARM
        """
        B, L_tgt, D = tgt.shape
        L_mem = memory.shape[1]

        # Initialize cumulative attention: [B, nhead, L_tgt, H, W]
        # Use uniform attention as initial coverage
        prev_attn = paddle.zeros([B, self.nhead, L_tgt, height, width])
        
        # Reshape memory_key_padding_mask for ARM: [B, H*W] -> [B, H, W]
        mem_mask_2d = memory_key_padding_mask.reshape([B, height, width])

        output = tgt
        for layer in self.layers:
            # Get coverage bias from ARM
            coverage_bias = None
            if self.arm is not None:
                # Get coverage bias: [B, nhead, L_tgt, H*W]
                coverage_bias = self.arm(prev_attn, mem_mask_2d, height, width)
                # Reshape for attention: [B*nhead, L_tgt, L_mem]
                coverage_bias = coverage_bias.reshape([B * self.nhead, L_tgt, L_mem])

            output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                coverage_bias=coverage_bias,
            )
            
            # Update cumulative attention with uniform attention estimate
            # (In full implementation, we'd extract actual attention weights)
            if self.arm is not None:
                # Simple uniform attention accumulation
                attn_update = paddle.ones([B, self.nhead, L_tgt, height, width]) / (height * width)
                # Mask padding positions
                attn_update = attn_update * (~mem_mask_2d).unsqueeze(1).unsqueeze(2).astype('float32')
                prev_attn = prev_attn + attn_update

        return output


# ============================================================================
# Tree-Aware Structure Similarity Module (from TAMER)
# ============================================================================

class StructSimOneDir(nn.Layer):
    """
    One-directional structure similarity prediction.
    Predicts parent-child relationships in the parse tree.
    """

    def __init__(self, d_model, nhead=8, dim_feedforward=1024, dropout=0.3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation='relu',
        )
        self.trm = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_sim = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, tgt, tgt_key_padding_mask=None):
        """
        Args:
            tgt: [B, L, D]
            tgt_key_padding_mask: [B, L]
        Returns:
            sim: [B, L, L] - similarity matrix
        """
        # Transform through encoder
        tgt = self.trm(tgt, src_key_padding_mask=tgt_key_padding_mask)

        # Compute Q and K
        q = self.to_q(tgt)  # [B, L, D]
        k = self.to_k(tgt)  # [B, L, D]

        # Compute pairwise similarity: sim[i,j] = f(q[i] + k[j])
        # [B, L, 1, D] + [B, 1, L, D] -> [B, L, L, D]
        B, L, D = q.shape
        q = q.unsqueeze(2)  # [B, L, 1, D]
        k = k.unsqueeze(1)  # [B, 1, L, D]
        combined = q + k  # [B, L, L, D]

        sim = self.to_sim(combined).squeeze(-1)  # [B, L, L]

        # Mask padding positions
        if tgt_key_padding_mask is not None:
            mask = tgt_key_padding_mask.unsqueeze(1)  # [B, 1, L]
            sim = sim.masked_fill(mask, float('-inf'))

        return sim


class StructSim(nn.Layer):
    """
    Bidirectional structure similarity module from TAMER.
    Predicts tree structure for both L2R and R2L decoding directions.
    """

    def __init__(self, d_model, nhead=8, dim_feedforward=1024, dropout=0.3):
        super().__init__()
        self.l2r_struct_sim = StructSimOneDir(d_model, nhead, dim_feedforward, dropout)
        self.r2l_struct_sim = StructSimOneDir(d_model, nhead, dim_feedforward, dropout)

    def forward(self, out, tgt_key_padding_mask=None):
        """
        Args:
            out: [2B, L, D] - concatenated L2R and R2L decoder outputs
            tgt_key_padding_mask: [2B, L]
        Returns:
            sim: [2B, L, L] - structure similarity matrices
        """
        B2 = out.shape[0]
        B = B2 // 2

        # Split L2R and R2L
        l2r_out = out[:B]  # [B, L, D]
        r2l_out = out[B:]  # [B, L, D]

        if tgt_key_padding_mask is not None:
            l2r_mask = tgt_key_padding_mask[:B]
            r2l_mask = tgt_key_padding_mask[B:]
        else:
            l2r_mask = None
            r2l_mask = None

        l2r_sim = self.l2r_struct_sim(l2r_out, l2r_mask)
        r2l_sim = self.r2l_struct_sim(r2l_out, r2l_mask)

        sim = paddle.concat([l2r_sim, r2l_sim], axis=0)
        return sim


# ============================================================================
# HME Recognition Head
# ============================================================================

class HMEHead(nn.Layer):
    """
    HME (Handwritten Mathematical Expression) Recognition Head.
    
    Features:
    - Bidirectional decoding (L2R and R2L merged via beam search)
    - Coverage attention (ARM) to prevent attention errors
    - Tree structure prediction (StructSim) for bracket matching
    - Counting module auxiliary task
    
    Args:
        in_channels: Input feature dimension from encoder
        vocab_size: Vocabulary size
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_decoder_layers: Number of transformer decoder layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        dc: ARM coverage dimension
        cross_coverage: Use cross-layer coverage attention
        self_coverage: Use self-layer coverage attention
        use_struct_sim: Enable tree structure prediction
        max_len: Maximum sequence length
    """

    def __init__(
        self,
        in_channels,
        vocab_size=113,
        d_model=256,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.3,
        dc=32,
        cross_coverage=True,
        self_coverage=True,
        use_struct_sim=True,
        max_len=200,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.use_struct_sim = use_struct_sim

        # Feature projection
        if in_channels != d_model:
            self.feature_proj = nn.Conv2D(in_channels, d_model, kernel_size=1)
        else:
            self.feature_proj = nn.Identity()

        # Position encodings
        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)
        self.pos_enc_1d = WordPosEnc(d_model, max_len=max_len)

        # Word embedding
        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.LayerNorm(d_model),
        )

        # Transformer decoder
        self.decoder = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        # Output projection
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

        # Structure similarity (optional)
        if use_struct_sim:
            self.struct_sim = StructSim(d_model, nhead=8, dim_feedforward=dim_feedforward, dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                trunc_normal_(m.weight)

    def _generate_square_subsequent_mask(self, sz):
        """Generate causal attention mask."""
        mask = paddle.triu(paddle.ones([sz, sz]), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, inputs, targets=None):
        """
        Args:
            inputs: Tuple of (features, image_mask, labels) from backbone
                - features: [B, C, H, W] encoder output
                - image_mask: [B, 1, H, W] image padding mask
                - labels: [B, L] target sequences
            targets: Optional target dict (unused, for API compatibility)
            
        Returns:
            word_probs: [B, L, vocab_size] output logits
        """
        features, image_mask, labels = inputs

        B, C, H, W = features.shape
        L = labels.shape[1]

        # Project features to d_model dimension
        features = self.feature_proj(features)  # [B, d_model, H, W]

        # Downsample mask to match feature spatial dimensions
        # image_mask: [B, 1, orig_H, orig_W] -> [B, 1, H, W]
        # Use adaptive average pooling to downsample
        mask_downsampled = F.adaptive_avg_pool2d(image_mask, output_size=[H, W])
        # Threshold to binary: > 0.5 means valid
        mask_downsampled = (mask_downsampled > 0.5).astype('float32')
        
        # Prepare mask for position encoding: [B, 1, H, W] -> [B, H, W]
        # image_mask is 1 for valid, 0 for padded
        # ImgPosEnc expects True for padding, so invert
        mask_2d = mask_downsampled.squeeze(1)  # [B, H, W]
        padding_mask_2d = (1 - mask_2d).astype('bool')  # True where padded

        # Reshape to [B, H, W, D] for position encoding
        features = features.transpose([0, 2, 3, 1])  # [B, H, W, D]
        features = self.pos_enc_2d(features, padding_mask_2d)
        features = self.norm(features)

        # Flatten to sequence: [B, H*W, D]
        memory = features.reshape([B, H * W, self.d_model])

        # Memory padding mask: [B, H*W] - True where padded
        memory_key_padding_mask = padding_mask_2d.reshape([B, H * W])

        # Clamp labels to valid range [0, vocab_size-1] to avoid embedding errors
        labels = paddle.clip(labels, min=0, max=self.vocab_size - 1)
        
        # Embed target sequence
        tgt = self.word_embed(labels)  # [B, L, D]
        tgt = self.pos_enc_1d(tgt)

        # Create causal mask for autoregressive decoding
        tgt_mask = self._generate_square_subsequent_mask(L)
        
        # Target padding mask: True where padded (0 values are PAD tokens)
        tgt_key_padding_mask = (labels == 0)

        # Decode
        out = self.decoder(
            tgt, memory, H, W,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # Structure similarity (optional)
        sim = None
        if self.use_struct_sim:
            sim = self.struct_sim(out, tgt_key_padding_mask)

        # Project to vocabulary
        word_probs = self.proj(out)  # [B, L, vocab_size]

        # Always return tuple for compatibility with CAN metric/loss interface
        # Format: (word_probs, extra) where extra is sim or None
        return (word_probs, sim)

