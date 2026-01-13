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
HMEHeadV2: Next-generation Handwritten Mathematical Expression Recognition Head.

Features:
1. Proper autoregressive generation (fixed SOS/EOS handling)
2. Differential Attention (ICLR 2025) - noise cancellation
3. Mamba/SSM layers - O(n) complexity alternative to attention
4. Mixture of Experts FFN - sparse computation with capacity

Key fixes from HMEHead:
- Vocabulary: EOS=0, SOS=1 (matches latex_symbol_dict.txt)
- Shifted labels: input[i] -> predict target[i] (proper teacher forcing)
- True autoregressive inference without ground truth labels
"""

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant, XavierUniform

__all__ = ["HMEHeadV2"]

trunc_normal_ = TruncatedNormal(std=0.02)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)
xavier_uniform_ = XavierUniform()


# ============================================================================
# Vocabulary Constants
# ============================================================================
EOS_IDX = 0  # End of Sequence
SOS_IDX = 1  # Start of Sequence


# ============================================================================
# Position Encoding (from HMEHead)
# ============================================================================

class WordPosEnc(nn.Layer):
    """1D Sinusoidal Position Encoding for decoder sequence."""

    def __init__(self, d_model=256, max_len=500, temperature=10000.0):
        super().__init__()
        pe = paddle.zeros([max_len, d_model])
        position = paddle.arange(0, max_len, dtype='float32').unsqueeze(1)
        dim_t = paddle.arange(0, d_model, 2, dtype='float32')
        div_term = 1.0 / (temperature ** (dim_t / d_model))
        inv_freq = position * div_term
        pe[:, 0::2] = paddle.sin(inv_freq)
        pe[:, 1::2] = paddle.cos(inv_freq)
        self.register_buffer("pe", pe)

    def forward(self, x):
        _, seq_len, _ = x.shape
        if seq_len <= self.pe.shape[0]:
            emb = self.pe[:seq_len, :]
        else:
            emb = paddle.concat([
                self.pe,
                self.pe[-1:, :].expand([seq_len - self.pe.shape[0], -1])
            ], axis=0)
        return x + emb.unsqueeze(0)


class ImgPosEnc(nn.Layer):
    """2D Sinusoidal Position Encoding for encoder features."""

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
        not_mask = ~mask
        y_embed = paddle.cumsum(not_mask.astype('float32'), axis=1)
        x_embed = paddle.cumsum(not_mask.astype('float32'), axis=2)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = paddle.arange(0, self.half_d_model, 2, dtype='float32')
        inv_freq = 1.0 / (self.temperature ** (dim_t / self.half_d_model))

        pos_x = x_embed.unsqueeze(-1) * inv_freq.reshape([1, 1, 1, -1])
        pos_y = y_embed.unsqueeze(-1) * inv_freq.reshape([1, 1, 1, -1])

        pos_x = paddle.stack([paddle.sin(pos_x), paddle.cos(pos_x)], axis=4)
        pos_x = pos_x.reshape([pos_x.shape[0], pos_x.shape[1], pos_x.shape[2], -1])
        pos_y = paddle.stack([paddle.sin(pos_y), paddle.cos(pos_y)], axis=4)
        pos_y = pos_y.reshape([pos_y.shape[0], pos_y.shape[1], pos_y.shape[2], -1])

        pos = paddle.concat([pos_x, pos_y], axis=-1)
        return x + pos


# ============================================================================
# PaTH Position Encoding (Householder Transformations)
# ============================================================================

class PaTHPositionEncoding(nn.Layer):
    """
    Position via Accumulated Householder Transformations (PaTH).

    From "PaTH Attention" - uses Householder reflections for data-dependent
    position encoding. Key idea: accumulate orthogonal transformations that
    encode position in a learnable, input-dependent way.

    Householder reflection: H = I - beta * w * w^T
    where w is computed from input and beta controls reflection magnitude.
    """

    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Project input to reflection vector
        self.w_proj = nn.Linear(d_model, d_model)

        # Learnable beta per head (controls reflection strength)
        # Initialize near 2.0 for full Householder reflection
        self.beta = self.create_parameter(
            shape=[d_model],
            default_initializer=Constant(value=2.0)
        )

        # Optional: learnable position-specific bias
        self.pos_bias = self.create_parameter(
            shape=[max_len, d_model],
            default_initializer=TruncatedNormal(std=0.02)
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _householder_transform(self, x, w):
        """
        Apply Householder transformation: H(x) = x - beta * (w^T x) * w

        Args:
            x: [B, L, D] input
            w: [B, L, D] reflection vectors (should be unit normalized)
        Returns:
            [B, L, D] transformed output
        """
        # Normalize w to unit vectors (critical for stable reflections)
        w_norm = F.normalize(w, p=2, axis=-1)

        # Compute dot product: [B, L, 1]
        dot = (x * w_norm).sum(axis=-1, keepdim=True)

        # Apply Householder: x - beta * (w^T x) * w
        # beta is [D], broadcast to [1, 1, D]
        beta = paddle.sigmoid(self.beta) * 2.0  # Keep in [0, 2] range
        transformed = x - beta.unsqueeze(0).unsqueeze(0) * dot * w_norm

        return transformed

    def forward(self, x):
        """
        Apply PaTH position encoding via accumulated Householder transforms.

        Args:
            x: [B, L, D] input embeddings
        Returns:
            [B, L, D] position-encoded embeddings
        """
        B, L, D = x.shape

        # Compute position-dependent reflection vectors
        w = self.w_proj(x)  # [B, L, D]

        # Add position bias for position-specific information
        if L <= self.max_len:
            pos_bias = self.pos_bias[:L, :]
        else:
            # Extend if needed
            pos_bias = paddle.concat([
                self.pos_bias,
                self.pos_bias[-1:, :].expand([L - self.max_len, -1])
            ], axis=0)

        w = w + pos_bias.unsqueeze(0)

        # Apply Householder transformation
        out = self._householder_transform(x, w)

        # Residual connection and normalization
        out = self.norm(x + self.dropout(out - x))

        return out


class PaTHCrossAttention(nn.Layer):
    """
    Cross-attention with PaTH-style position encoding.

    Applies Householder transformations to queries and keys before attention
    to encode relative position information.
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # PaTH: learnable reflection vectors for Q and K
        self.q_path = nn.Linear(d_model, d_model)
        self.k_path = nn.Linear(d_model, d_model)

        # Beta for Householder (per head)
        self.beta_q = self.create_parameter(
            shape=[nhead, 1, 1],
            default_initializer=Constant(value=1.0)
        )
        self.beta_k = self.create_parameter(
            shape=[nhead, 1, 1],
            default_initializer=Constant(value=1.0)
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def _apply_path(self, x, path_proj, beta):
        """Apply PaTH transformation to Q or K."""
        B, nhead, L, D = x.shape

        # Compute reflection vector
        # Reshape for projection: [B, nhead, L, D] -> [B, L, nhead*D]
        x_flat = x.transpose([0, 2, 1, 3]).reshape([B, L, self.d_model])
        w = path_proj(x_flat)
        w = w.reshape([B, L, self.nhead, self.head_dim]).transpose([0, 2, 1, 3])

        # Normalize
        w_norm = F.normalize(w, p=2, axis=-1)

        # Householder: x - beta * (w^T x) * w
        dot = (x * w_norm).sum(axis=-1, keepdim=True)
        beta_val = paddle.sigmoid(beta) * 2.0
        transformed = x - beta_val * dot * w_norm

        return transformed

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: [B, Tq, D]
            key: [B, Tk, D]
            value: [B, Tk, D]
        Returns:
            [B, Tq, D]
        """
        B, Tq, _ = query.shape
        Tk = key.shape[1]

        # Project
        q = self.q_proj(query).reshape([B, Tq, self.nhead, self.head_dim]).transpose([0, 2, 1, 3])
        k = self.k_proj(key).reshape([B, Tk, self.nhead, self.head_dim]).transpose([0, 2, 1, 3])
        v = self.v_proj(value).reshape([B, Tk, self.nhead, self.head_dim]).transpose([0, 2, 1, 3])

        # Apply PaTH transformations
        q = self._apply_path(q, self.q_path, self.beta_q)
        k = self._apply_path(k, self.k_path, self.beta_k)

        # Standard scaled dot-product attention
        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * self.scale

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn = attn + attn_mask

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2).astype('float32') * (-1e9)
            attn = attn + mask

        attn = F.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        out = paddle.matmul(attn, v)
        out = out.transpose([0, 2, 1, 3]).reshape([B, Tq, self.d_model])
        out = self.out_proj(out)

        return out


# ============================================================================
# Standard Multi-Head Attention (Baseline)
# ============================================================================

class StandardAttention(nn.Layer):
    """
    Standard multi-head attention as baseline for comparison.
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: [B, Tq, D]
            key: [B, Tk, D]
            value: [B, Tk, D]
        Returns:
            [B, Tq, D]
        """
        B, Tq, _ = query.shape
        Tk = key.shape[1]

        # Project and reshape
        q = self.q_proj(query).reshape([B, Tq, self.nhead, self.head_dim]).transpose([0, 2, 1, 3])
        k = self.k_proj(key).reshape([B, Tk, self.nhead, self.head_dim]).transpose([0, 2, 1, 3])
        v = self.v_proj(value).reshape([B, Tk, self.nhead, self.head_dim]).transpose([0, 2, 1, 3])

        # Scaled dot-product attention
        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * self.scale

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn = attn + attn_mask

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2).astype('float32') * (-1e9)
            attn = attn + mask

        attn = F.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        out = paddle.matmul(attn, v)
        out = out.transpose([0, 2, 1, 3]).reshape([B, Tq, self.d_model])
        out = self.out_proj(out)

        return out


# ============================================================================
# Differential Attention (ICLR 2025)
# ============================================================================

class DifferentialAttention(nn.Layer):
    """
    Differential Attention from "Differential Transformer" (ICLR 2025).

    Key idea: Split Q/K into two groups and subtract attention maps:
        attn = softmax(Q1 @ K1.T) - lambda * softmax(Q2 @ K2.T)

    This cancels out noise and improves attention focus.
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Split each head into 2 sub-heads for differential
        self.sub_head_dim = self.head_dim // 2

        # Q, K, V projections (2x for differential)
        self.q_proj = nn.Linear(d_model, d_model * 2)
        self.k_proj = nn.Linear(d_model, d_model * 2)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable lambda parameter for differential (scalar, properly bounded)
        # Using sigmoid to keep lambda in stable range [0.1, 0.9]
        self.lambda_param = self.create_parameter(
            shape=[1], default_initializer=Constant(value=0.0)
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = self.sub_head_dim ** -0.5

    def _compute_lambda(self):
        """Compute adaptive lambda, bounded to [0.1, 0.9] for stability."""
        lambda_val = paddle.sigmoid(self.lambda_param) * 0.8 + 0.1
        return lambda_val

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: [B, Tq, D]
            key: [B, Tk, D]
            value: [B, Tk, D]
            attn_mask: [Tq, Tk] or [B, Tq, Tk] causal/attention mask
            key_padding_mask: [B, Tk] True for padding
        Returns:
            [B, Tq, D]
        """
        B, Tq, _ = query.shape
        Tk = key.shape[1]

        # Project and reshape: [B, T, D] -> [B, nhead, T, head_dim*2]
        q = self.q_proj(query).reshape([B, Tq, self.nhead, self.head_dim * 2]).transpose([0, 2, 1, 3])
        k = self.k_proj(key).reshape([B, Tk, self.nhead, self.head_dim * 2]).transpose([0, 2, 1, 3])
        v = self.v_proj(value).reshape([B, Tk, self.nhead, self.head_dim]).transpose([0, 2, 1, 3])

        # Split Q/K into two groups: [B, nhead, T, head_dim]
        q1, q2 = q[:, :, :, :self.head_dim], q[:, :, :, self.head_dim:]
        k1, k2 = k[:, :, :, :self.head_dim], k[:, :, :, self.head_dim:]

        # Compute attention scores: [B, nhead, Tq, Tk]
        attn1 = paddle.matmul(q1, k1.transpose([0, 1, 3, 2])) * self.scale
        attn2 = paddle.matmul(q2, k2.transpose([0, 1, 3, 2])) * self.scale

        # Apply masks
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn1 = attn1 + attn_mask
            attn2 = attn2 + attn_mask

        if key_padding_mask is not None:
            # [B, Tk] -> [B, 1, 1, Tk]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2).astype('float32') * (-1e9)
            attn1 = attn1 + mask
            attn2 = attn2 + mask

        # Differential attention: subtract BEFORE softmax, then normalize
        # This cancels noise while maintaining valid attention distribution
        lambda_val = self._compute_lambda()

        # Compute differential scores before softmax
        attn_scores = attn1 - lambda_val * attn2

        # Apply softmax to get valid probability distribution
        attn = F.softmax(attn_scores, axis=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = paddle.matmul(attn, v)  # [B, nhead, Tq, head_dim]
        out = out.transpose([0, 2, 1, 3]).reshape([B, Tq, self.d_model])
        out = self.out_proj(out)

        return out


# ============================================================================
# Mamba Layer (Selective State Space Model)
# ============================================================================

class MambaLayer(nn.Layer):
    """
    Simplified Mamba layer for sequence modeling.

    Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

    Key features:
    - O(n) complexity vs O(n^2) for attention
    - Input-dependent state space parameters
    - Local context via 1D convolution

    Note: This is a simplified version for PaddlePaddle compatibility.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # 1D conv for local context
        self.conv1d = nn.Conv1D(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner
        )

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)  # B, C, delta

        # Learnable A parameter (diagonal state matrix)
        A = paddle.arange(1, d_state + 1, dtype='float32')
        self.A_log = self.create_parameter(
            shape=[self.d_inner, d_state],
            default_initializer=Constant(value=0.0)
        )
        with paddle.no_grad():
            self.A_log.set_value(paddle.log(A.unsqueeze(0).expand([self.d_inner, -1])))

        # D skip connection
        self.D = self.create_parameter(
            shape=[self.d_inner],
            default_initializer=ones_
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, D]
            mask: [B, L] optional padding mask
        Returns:
            [B, L, D]
        """
        B, L, D = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_in, z = paddle.split(xz, 2, axis=-1)  # [B, L, d_inner] each

        # 1D conv for local context
        x_conv = x_in.transpose([0, 2, 1])  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[:, :, :L]  # Trim to original length
        x_conv = x_conv.transpose([0, 2, 1])  # [B, L, d_inner]
        x_conv = F.silu(x_conv)

        # Compute input-dependent SSM parameters
        x_dbl = self.x_proj(x_conv)  # [B, L, d_state*2 + 1]
        B_param = x_dbl[:, :, :self.d_state]  # [B, L, d_state]
        C_param = x_dbl[:, :, self.d_state:2*self.d_state]  # [B, L, d_state]
        delta = F.softplus(x_dbl[:, :, -1])  # [B, L]

        # State matrix A
        A = -paddle.exp(self.A_log)  # [d_inner, d_state]

        # Discretized SSM: simplified sequential scan
        # For efficiency, we use a parallel-friendly approximation
        y = self._ssm_forward(x_conv, A, B_param, C_param, delta)

        # Skip connection
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # Gate and output
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        return y

    def _ssm_forward(self, x, A, B, C, delta):
        """
        Simplified SSM forward pass using parallel scan approximation.

        For a true Mamba implementation, this would use a selective scan.
        This version uses a causal convolution approximation for export compatibility.
        """
        B_size, L, d_inner = x.shape

        # Simple causal aggregation (approximates SSM behavior)
        # This is export-friendly but less accurate than true selective scan

        # Expand delta for element-wise ops: [B, L] -> [B, L, 1]
        delta = delta.unsqueeze(-1)

        # Discretize: dA = exp(delta * A)
        # For simplicity, use first-order approximation
        dA = paddle.exp(delta * A.unsqueeze(0))  # [B, L, d_state]
        dB = delta * B  # [B, L, d_state]

        # Project x to state space: [B, L, d_inner] @ [d_inner, d_state] -> [B, L, d_state]
        # We use a learned projection implicitly through B
        x_proj = x.unsqueeze(-1) * dB.unsqueeze(2)  # [B, L, d_inner, d_state]
        x_proj = x_proj.sum(axis=-1)  # [B, L, d_inner]

        # Causal cumsum with decay (approximates state evolution)
        # Using exponential moving average as approximation
        alpha = 0.9  # Decay factor
        h = paddle.zeros([B_size, d_inner])
        outputs = []

        for t in range(L):
            h = alpha * h + (1 - alpha) * x_proj[:, t, :]
            outputs.append(h)

        y = paddle.stack(outputs, axis=1)  # [B, L, d_inner]

        # Output projection via C
        y = y * C.mean(axis=-1, keepdim=True)  # Simple approximation

        return y


# ============================================================================
# Mixture of Experts FFN
# ============================================================================

class ExpertFFN(nn.Layer):
    """Single expert feed-forward network."""

    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MoEFFN(nn.Layer):
    """
    Mixture of Experts Feed-Forward Network.

    Uses top-k routing to select which experts process each token.
    Includes load balancing auxiliary loss.
    """

    def __init__(self, d_model, d_ffn, num_experts=2, top_k=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Router: predicts which expert(s) to use
        self.router = nn.Linear(d_model, num_experts)

        # Expert networks
        self.experts = nn.LayerList([
            ExpertFFN(d_model, d_ffn, dropout)
            for _ in range(num_experts)
        ])

        # For auxiliary loss
        self.aux_loss = None

    def forward(self, x):
        """
        Args:
            x: [B, L, D]
        Returns:
            [B, L, D]
        """
        B, L, D = x.shape

        # Compute router logits: [B, L, num_experts]
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, axis=-1)

        # Top-k selection: [B, L, top_k]
        top_k_probs, top_k_indices = paddle.topk(router_probs, self.top_k, axis=-1)

        # Normalize top-k probs
        top_k_probs = top_k_probs / (top_k_probs.sum(axis=-1, keepdim=True) + 1e-9)

        # Compute auxiliary load balancing loss
        # Encourages balanced expert utilization
        expert_counts = paddle.zeros([self.num_experts])
        for i in range(self.num_experts):
            expert_counts[i] = (top_k_indices == i).astype('float32').sum()
        expert_frac = expert_counts / (B * L * self.top_k + 1e-9)
        avg_router_prob = router_probs.mean(axis=[0, 1])
        self.aux_loss = self.num_experts * (expert_frac * avg_router_prob).sum()

        # Route tokens to experts
        # For efficiency with small num_experts, compute all and mask
        outputs = paddle.zeros_like(x)

        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # [B, L]
            gate_score = top_k_probs[:, :, k:k+1]  # [B, L, 1]

            for e in range(self.num_experts):
                mask = (expert_idx == e).unsqueeze(-1).astype('float32')  # [B, L, 1]
                if mask.sum() > 0:
                    expert_out = self.experts[e](x)  # [B, L, D]
                    outputs = outputs + mask * gate_score * expert_out

        return outputs


# ============================================================================
# Decoder Layer with Differential Attention and MoE
# ============================================================================

class DecoderLayerV2(nn.Layer):
    """
    Decoder layer with:
    - Standard attention (baseline) OR Differential attention OR PaTH attention
    - MoE FFN (optional)
    """

    def __init__(self, d_model, nhead, d_ffn=1024, dropout=0.1,
                 use_moe=False, num_experts=2, moe_top_k=1,
                 attention_type='standard'):
        super().__init__()
        self.d_model = d_model
        self.attention_type = attention_type

        # Self attention and cross attention
        if attention_type == 'path':
            self.self_attn = PaTHCrossAttention(d_model, nhead, dropout)
            self.cross_attn = PaTHCrossAttention(d_model, nhead, dropout)
        elif attention_type == 'differential':
            self.self_attn = DifferentialAttention(d_model, nhead, dropout)
            self.cross_attn = DifferentialAttention(d_model, nhead, dropout)
        else:  # 'standard' (default baseline)
            self.self_attn = StandardAttention(d_model, nhead, dropout)
            self.cross_attn = StandardAttention(d_model, nhead, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # FFN (standard or MoE)
        if use_moe:
            self.ffn = MoEFFN(d_model, d_ffn, num_experts, moe_top_k, dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ffn, d_model),
            )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.use_moe = use_moe

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        Args:
            tgt: [B, L_tgt, D]
            memory: [B, L_mem, D]
        """
        # Self attention
        tgt2 = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        tgt2 = self.cross_attn(
            tgt, memory, memory,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def get_aux_loss(self):
        """Get MoE auxiliary loss if using MoE."""
        if self.use_moe and hasattr(self.ffn, 'aux_loss'):
            return self.ffn.aux_loss
        return 0.0


# ============================================================================
# HMEHeadV2 Main Class
# ============================================================================

class HMEHeadV2(nn.Layer):
    """
    HME Recognition Head V2 with proper autoregressive generation.

    Key improvements over HMEHead:
    1. Correct SOS/EOS handling (EOS=0, SOS=1)
    2. Proper label shifting for teacher forcing
    3. Working greedy decode for inference
    4. Modern architectures: Differential Attention, Mamba, MoE

    Args:
        in_channels: Input feature dimension from backbone
        vocab_size: Vocabulary size (default 113 for latex_symbol_dict.txt)
        d_model: Model hidden dimension
        nhead: Number of attention heads
        num_decoder_layers: Number of decoder layers
        d_ffn: FFN hidden dimension
        dropout: Dropout rate
        use_moe: Use Mixture of Experts FFN
        num_experts: Number of MoE experts
        moe_top_k: Top-k experts to route to
        use_mamba: Use Mamba layers (experimental)
        use_path: Use PaTH position encoding (Householder transforms)
        attention_type: 'differential' (default) or 'path' for attention layers
        max_len: Maximum sequence length
    """

    def __init__(
        self,
        in_channels,
        vocab_size=113,
        d_model=192,
        nhead=4,
        num_decoder_layers=2,
        d_ffn=512,
        dropout=0.1,
        use_moe=True,
        num_experts=2,
        moe_top_k=1,
        use_mamba=False,
        use_path=False,
        attention_type='differential',
        max_len=256,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.use_moe = use_moe
        self.use_mamba = use_mamba
        self.use_path = use_path

        # Debug: verify use_moe is being passed correctly
        print(f"[HMEHeadV2] use_moe={use_moe}, use_mamba={use_mamba}, attention_type={attention_type}")

        # Feature projection
        if in_channels != d_model:
            self.feature_proj = nn.Conv2D(in_channels, d_model, kernel_size=1)
        else:
            self.feature_proj = nn.Identity()

        # Position encodings
        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)

        # Use PaTH or standard sinusoidal position encoding for 1D
        if use_path:
            self.pos_enc_1d = PaTHPositionEncoding(d_model, max_len=max_len, dropout=dropout)
        else:
            self.pos_enc_1d = WordPosEnc(d_model, max_len=max_len)

        # Word embedding with LayerNorm
        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.LayerNorm(d_model),
        )

        # Decoder layers
        self.layers = nn.LayerList([
            DecoderLayerV2(
                d_model, nhead, d_ffn, dropout,
                use_moe=use_moe, num_experts=num_experts, moe_top_k=moe_top_k,
                attention_type=attention_type
            )
            for _ in range(num_decoder_layers)
        ])

        # Optional Mamba layer for efficient sequence modeling
        if use_mamba:
            self.mamba = MambaLayer(d_model, d_state=16, d_conv=4, expand=2, dropout=dropout)

        # Output projection
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                trunc_normal_(m.weight)

    def _generate_causal_mask(self, sz):
        """Generate causal attention mask."""
        mask = paddle.triu(paddle.ones([sz, sz]), diagonal=1) * (-1e9)
        return mask

    def _encode_image(self, features, image_mask=None):
        """
        Encode image features to memory.

        Args:
            features: [B, C, H, W]
            image_mask: [B, 1, H, W] optional
        Returns:
            memory: [B, H*W, D]
            memory_key_padding_mask: [B, H*W]
            H, W: spatial dimensions
        """
        B, C, H, W = features.shape

        # Project features
        features = self.feature_proj(features)

        # Handle mask
        if image_mask is not None:
            mask_down = F.adaptive_avg_pool2d(image_mask, output_size=[H, W])
            mask_down = (mask_down > 0.5).astype('float32')
            mask_2d = mask_down.squeeze(1)
            padding_mask_2d = (1 - mask_2d).astype('bool')
        else:
            padding_mask_2d = paddle.zeros([B, H, W], dtype='bool')

        # Reshape and add position encoding
        features = features.transpose([0, 2, 3, 1])  # [B, H, W, D]
        features = self.pos_enc_2d(features, padding_mask_2d)
        features = self.norm(features)

        # Flatten to sequence
        memory = features.reshape([B, H * W, self.d_model])
        memory_key_padding_mask = padding_mask_2d.reshape([B, H * W])

        return memory, memory_key_padding_mask, H, W

    def _decode(self, tgt, memory, memory_key_padding_mask, tgt_mask=None):
        """
        Run decoder on embedded target sequence.

        Args:
            tgt: [B, L, D] embedded targets
            memory: [B, H*W, D] encoded image
            memory_key_padding_mask: [B, H*W]
            tgt_mask: [L, L] causal mask
        """
        # Optional Mamba processing
        if self.use_mamba:
            tgt = tgt + self.mamba(tgt)

        # Decoder layers
        for layer in self.layers:
            tgt = layer(
                tgt, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        return tgt

    def forward(self, inputs, targets=None):
        """
        Forward pass for training or inference.

        Args:
            inputs: Either:
                - (features, image_mask, decoder_inputs) for training
                - features tensor [B, C, H, W] for inference
            targets: Optional dict with 'decoder_targets' for loss computation

        Training expects:
            - decoder_inputs: [B, L] with [SOS, tok1, tok2, ...]
            - decoder_targets: [B, L] with [tok1, tok2, ..., EOS]

        Returns:
            Training: dict with 'logits', 'aux_loss'
            Inference: logits from greedy decode
        """
        if isinstance(inputs, (list, tuple)):
            features, image_mask, decoder_inputs = inputs
            is_training = True
        else:
            features = inputs
            image_mask = None
            decoder_inputs = None
            is_training = False

        # Encode image
        memory, memory_key_padding_mask, H, W = self._encode_image(features, image_mask)

        if is_training:
            # Training: teacher forcing with decoder_inputs
            B, L = decoder_inputs.shape

            # Clamp to valid vocab range
            decoder_inputs = paddle.clip(decoder_inputs, min=0, max=self.vocab_size - 1)

            # Embed and add position encoding
            tgt = self.word_embed(decoder_inputs)
            tgt = self.pos_enc_1d(tgt)

            # Causal mask
            tgt_mask = self._generate_causal_mask(L)

            # Decode
            out = self._decode(tgt, memory, memory_key_padding_mask, tgt_mask)

            # Apply output norm and project to vocabulary
            out = self.norm(out)
            logits = self.proj(out)  # [B, L, vocab_size]

            # Collect auxiliary losses from MoE layers (only if MoE enabled)
            if self.use_moe:
                aux_loss = 0.0
                for layer in self.layers:
                    aux_loss = aux_loss + layer.get_aux_loss()
            else:
                aux_loss = 0.0

            # Debug: verify aux_loss computation (only print first batch)
            if not hasattr(self, '_aux_debug_printed'):
                print(f"[HMEHeadV2.forward] use_moe={self.use_moe}, aux_loss={aux_loss}")
                self._aux_debug_printed = True

            return {'logits': logits, 'aux_loss': aux_loss}
        else:
            # Inference: greedy decode
            return self._greedy_decode(memory, memory_key_padding_mask)

    def _greedy_decode(self, memory, memory_key_padding_mask):
        """
        Greedy autoregressive decoding for inference.

        Args:
            memory: [B, H*W, D] encoded image
            memory_key_padding_mask: [B, H*W]
        Returns:
            [B, max_len, vocab_size] logits
        """
        B = memory.shape[0]

        # Start with SOS token
        ys = paddle.full([B, 1], SOS_IDX, dtype='int64')

        for step in range(self.max_len - 1):
            # Embed current sequence
            tgt = self.word_embed(ys)
            tgt = self.pos_enc_1d(tgt)

            # Causal mask
            curr_len = ys.shape[1]
            tgt_mask = self._generate_causal_mask(curr_len)

            # Decode
            out = self._decode(tgt, memory, memory_key_padding_mask, tgt_mask)

            # Get next token prediction (apply norm before projection)
            out_normed = self.norm(out[:, -1:, :])
            logits = self.proj(out_normed)  # [B, 1, vocab_size]
            next_token = paddle.argmax(logits, axis=-1)  # [B, 1]

            # Append to sequence
            ys = paddle.concat([ys, next_token], axis=1)

            # Check if all sequences have generated EOS
            if paddle.all(next_token == EOS_IDX):
                break

        # Return final logits for full sequence
        tgt = self.word_embed(ys)
        tgt = self.pos_enc_1d(tgt)
        tgt_mask = self._generate_causal_mask(ys.shape[1])
        out = self._decode(tgt, memory, memory_key_padding_mask, tgt_mask)
        out = self.norm(out)
        logits = self.proj(out)

        return logits
