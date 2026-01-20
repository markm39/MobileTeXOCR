"""
Unified Text Spotting Decoder

An autoregressive transformer decoder that generates both bounding boxes
and LaTeX text in a single sequence. Inspired by VISTA-OCR and HunyuanOCR.

Output format: <loc>x1,y1,x2,y2</loc> \\LaTeX_content <sep> <loc>...</loc> ...
"""

from typing import Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.tokenizer import LaTeXTokenizer, TokenizerConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, D]
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CrossAttention(nn.Module):
    """Cross-attention layer for attending to encoder features."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, T, D] decoder states
            key_value: [B, N, D] encoder features
            key_padding_mask: [B, N] True for positions to mask

        Returns:
            Attended features [B, T, D]
        """
        B, T, D = query.shape
        _, N, _ = key_value.shape

        q = self.q_proj(query).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, T, N]

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention, cross-attention, and FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention to encoder
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input [B, T, D]
            encoder_out: Encoder features [B, N, D]
            tgt_mask: Causal mask [T, T]
            tgt_key_padding_mask: Padding mask for targets [B, T]
            memory_key_padding_mask: Padding mask for encoder [B, N]

        Returns:
            Decoder output [B, T, D]
        """
        # Self-attention with causal masking
        x2, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            is_causal=True if tgt_mask is None else False,
        )
        x = self.norm1(x + self.dropout(x2))

        # Cross-attention to encoder
        x2 = self.cross_attn(x, encoder_out, memory_key_padding_mask)
        x = self.norm2(x + self.dropout(x2))

        # Feed-forward
        x = self.norm3(x + self.ffn(x))

        return x


class UnifiedDecoder(nn.Module):
    """Unified decoder for text spotting.

    Generates sequences of the form:
    <bos> <loc> x1 y1 x2 y2 </loc> latex_token_1 ... latex_token_n <sep> ... <eos>

    Features:
    - Autoregressive generation with causal masking
    - Cross-attention to encoder features for spatial grounding
    - Separate embeddings for location tokens vs text tokens
    - Supports multiple regions per image via <sep> token
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        num_location_bins: int = 1000,
    ):
        """
        Args:
            vocab_size: Total vocabulary size (location + latex tokens)
            d_model: Model dimension
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            num_location_bins: Number of bins for coordinate discretization
        """
        super().__init__()

        self.d_model = d_model
        self.num_location_bins = num_location_bins
        self.max_seq_length = max_seq_length

        # Token embedding (unified for both location and latex tokens)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_length, dropout)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return mask

    def forward(
        self,
        encoder_out: torch.Tensor,
        target_ids: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training.

        Args:
            encoder_out: Encoder features [B, N, D]
            target_ids: Target token IDs [B, T]
            encoder_padding_mask: [B, N] True for padding positions
            target_padding_mask: [B, T] True for padding positions

        Returns:
            Logits [B, T, vocab_size]
        """
        B, T = target_ids.shape

        # Embed tokens
        x = self.token_embedding(target_ids)  # [B, T, D]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Generate causal mask
        causal_mask = self._generate_causal_mask(T, x.device)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_out,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=target_padding_mask,
                memory_key_padding_mask=encoder_padding_mask,
            )

        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)  # [B, T, vocab_size]

        return logits

    @torch.no_grad()
    def generate(
        self,
        encoder_out: torch.Tensor,
        tokenizer: LaTeXTokenizer,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Autoregressive generation.

        Args:
            encoder_out: Encoder features [B, N, D]
            tokenizer: Tokenizer instance
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            encoder_padding_mask: [B, N] True for padding positions

        Returns:
            List of generated token sequences
        """
        B = encoder_out.shape[0]
        device = encoder_out.device

        # Start with BOS token
        generated = torch.full(
            (B, 1), tokenizer.bos_token_id,
            dtype=torch.long, device=device
        )

        # Track which sequences have finished
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            # Forward pass
            logits = self.forward(encoder_out, generated, encoder_padding_mask)
            next_logits = logits[:, -1, :] / temperature  # [B, vocab_size]

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Replace with EOS for finished sequences
            next_token[finished] = tokenizer.eos_token_id

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            # Update finished status
            finished = finished | (next_token.squeeze(-1) == tokenizer.eos_token_id)

            # Stop if all sequences have finished
            if finished.all():
                break

        return generated.tolist()

    @torch.no_grad()
    def greedy_decode(
        self,
        encoder_out: torch.Tensor,
        tokenizer: LaTeXTokenizer,
        max_length: int = 256,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Greedy decoding (faster than sampling).

        Args:
            encoder_out: Encoder features [B, N, D]
            tokenizer: Tokenizer instance
            max_length: Maximum generation length
            encoder_padding_mask: [B, N] True for padding positions

        Returns:
            List of generated token sequences
        """
        B = encoder_out.shape[0]
        device = encoder_out.device

        # Start with BOS token
        generated = torch.full(
            (B, 1), tokenizer.bos_token_id,
            dtype=torch.long, device=device
        )

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            logits = self.forward(encoder_out, generated, encoder_padding_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]

            next_token[finished] = tokenizer.eos_token_id
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(-1) == tokenizer.eos_token_id)

            if finished.all():
                break

        return generated.tolist()


def create_decoder(
    d_model: int = 384,
    num_layers: int = 6,
    num_heads: int = 8,
    tokenizer: Optional[LaTeXTokenizer] = None,
) -> UnifiedDecoder:
    """Create a unified decoder with default settings.

    Args:
        d_model: Model dimension (should match encoder output_dim)
        num_layers: Number of decoder layers
        num_heads: Number of attention heads
        tokenizer: Optional tokenizer (creates default if None)

    Returns:
        UnifiedDecoder instance
    """
    if tokenizer is None:
        tokenizer = LaTeXTokenizer()

    return UnifiedDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=d_model * 4,
        num_location_bins=tokenizer.num_location_bins,
    )
