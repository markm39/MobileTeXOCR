"""
Unified Text Spotting Decoder with KV-Caching

An autoregressive transformer decoder that generates both bounding boxes
and LaTeX text in a single sequence. Inspired by VISTA-OCR and HunyuanOCR.

Features:
- KV-caching for fast autoregressive generation (2-3x speedup)
- Beam search decoding with length normalization
- Cross-attention to encoder features for spatial grounding

Output format: <loc>x1,y1,x2,y2</loc> \\LaTeX_content <sep> <loc>...</loc> ...
"""

from typing import Optional, Tuple, List, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.tokenizer import LaTeXTokenizer, TokenizerConfig


# Type alias for KV cache: tuple of (key, value) tensors per layer
KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


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

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, D]
            offset: Position offset for cached generation
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, offset:offset + seq_len]
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """Causal self-attention with KV-caching support.

    This implementation uses separate Q, K, V projections instead of
    nn.MultiheadAttention to enable efficient KV-caching during generation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor [B, T, D]
            past_kv: Cached (key, value) from previous steps, each [B, H, T_past, head_dim]
            use_cache: Whether to return updated cache
            attention_mask: Optional attention mask [T, T] or [B, T, T]

        Returns:
            output: Attended features [B, T, D]
            present_kv: Updated (key, value) cache if use_cache=True, else None
        """
        B, T, D = x.shape

        # Compute Q, K, V for current input
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [B, H, T, head_dim]

        # Concatenate with past cache if available
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # [B, H, T_past + T, head_dim]
            v = torch.cat([past_v, v], dim=2)

        # Prepare cache to return
        present_kv = (k, v) if use_cache else None

        # Compute attention scores
        # q: [B, H, T, head_dim], k: [B, H, T_total, head_dim]
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, T, T_total]

        # Apply causal mask
        T_total = k.size(2)
        if attention_mask is None:
            # Create causal mask: new tokens can attend to all past + current
            # When using cache, T=1 (new token), T_total = past + 1
            # The new token at position T_total-1 can attend to all positions [0, T_total)
            causal_mask = torch.triu(
                torch.ones(T, T_total, device=x.device, dtype=torch.bool),
                diagonal=T_total - T + 1
            )
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            attn_weights = attn_weights.masked_fill(attention_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = (attn_weights @ v).transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)

        return out, present_kv


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
    """Single decoder layer with self-attention, cross-attention, and FFN.

    Supports KV-caching for efficient autoregressive generation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention (causal) with KV-caching
        self.self_attn = CausalSelfAttention(d_model, num_heads, dropout)
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
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Decoder input [B, T, D]
            encoder_out: Encoder features [B, N, D]
            past_kv: Cached (key, value) for self-attention
            use_cache: Whether to return updated cache
            memory_key_padding_mask: Padding mask for encoder [B, N]

        Returns:
            output: Decoder output [B, T, D]
            present_kv: Updated cache if use_cache=True
        """
        # Self-attention with caching
        attn_out, present_kv = self.self_attn(
            x, past_kv=past_kv, use_cache=use_cache
        )
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention to encoder
        cross_out = self.cross_attn(x, encoder_out, memory_key_padding_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # Feed-forward
        x = self.norm3(x + self.ffn(x))

        return x, present_kv


class UnifiedDecoder(nn.Module):
    """Unified decoder for text spotting with KV-caching support.

    Generates sequences of the form:
    <bos> <loc> x1 y1 x2 y2 </loc> latex_token_1 ... latex_token_n <sep> ... <eos>

    Features:
    - Autoregressive generation with causal masking
    - KV-caching for 2-3x faster generation
    - Cross-attention to encoder features for spatial grounding
    - Beam search with length normalization
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
        self.num_layers = num_layers
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

    def forward(
        self,
        encoder_out: torch.Tensor,
        target_ids: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
        past_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """Forward pass with optional KV-caching.

        Args:
            encoder_out: Encoder features [B, N, D]
            target_ids: Target token IDs [B, T]
            encoder_padding_mask: [B, N] True for padding positions
            target_padding_mask: [B, T] True for padding positions (unused with cache)
            past_cache: List of (key, value) tuples from previous forward passes
            use_cache: Whether to return updated cache for incremental generation

        Returns:
            logits: [B, T, vocab_size]
            present_cache: Updated cache if use_cache=True, else None
        """
        B, T = target_ids.shape

        # Calculate position offset for positional encoding
        if past_cache is not None and past_cache[0] is not None:
            pos_offset = past_cache[0][0].size(2)  # T_past from cached keys
        else:
            pos_offset = 0

        # Embed tokens
        x = self.token_embedding(target_ids)  # [B, T, D]

        # Add positional encoding with offset
        x = self.pos_encoding(x, offset=pos_offset)

        # Pass through decoder layers with caching
        present_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_cache[i] if past_cache is not None else None
            x, layer_present = layer(
                x, encoder_out,
                past_kv=layer_past,
                use_cache=use_cache,
                memory_key_padding_mask=encoder_padding_mask,
            )
            if use_cache:
                present_cache.append(layer_present)

        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)  # [B, T, vocab_size]

        return logits, present_cache

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
        use_cache: bool = True,
    ) -> List[List[int]]:
        """Autoregressive generation with sampling.

        Args:
            encoder_out: Encoder features [B, N, D]
            tokenizer: Tokenizer instance
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            encoder_padding_mask: [B, N] True for padding positions
            use_cache: Whether to use KV-caching (recommended)

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
        cache = None

        for _ in range(max_length - 1):
            # Get input for this step
            if use_cache and cache is not None:
                input_ids = generated[:, -1:]  # Only last token when using cache
            else:
                input_ids = generated

            # Forward pass
            logits, cache = self.forward(
                encoder_out, input_ids, encoder_padding_mask,
                past_cache=cache if use_cache else None,
                use_cache=use_cache,
            )
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
        use_cache: bool = True,
    ) -> List[List[int]]:
        """Greedy decoding with KV-caching.

        Args:
            encoder_out: Encoder features [B, N, D]
            tokenizer: Tokenizer instance
            max_length: Maximum generation length
            encoder_padding_mask: [B, N] True for padding positions
            use_cache: Whether to use KV-caching (recommended for speed)

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
        cache = None

        for _ in range(max_length - 1):
            # Get input for this step
            if use_cache and cache is not None:
                input_ids = generated[:, -1:]  # Only last token when using cache
            else:
                input_ids = generated

            # Forward pass
            logits, cache = self.forward(
                encoder_out, input_ids, encoder_padding_mask,
                past_cache=cache if use_cache else None,
                use_cache=use_cache,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]

            next_token[finished] = tokenizer.eos_token_id
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(-1) == tokenizer.eos_token_id)

            if finished.all():
                break

        return generated.tolist()

    @torch.no_grad()
    def beam_search(
        self,
        encoder_out: torch.Tensor,
        tokenizer: LaTeXTokenizer,
        beam_size: int = 5,
        max_length: int = 256,
        length_penalty: float = 1.0,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> List[List[int]]:
        """Beam search decoding with KV-caching for better quality.

        Args:
            encoder_out: Encoder features [B, N, D]
            tokenizer: Tokenizer instance
            beam_size: Number of beams to keep
            max_length: Maximum generation length
            length_penalty: Penalty for longer sequences (>1 favors longer)
            encoder_padding_mask: [B, N] True for padding positions
            use_cache: Whether to use KV-caching (recommended for speed)

        Returns:
            List of best token sequences (one per batch item)
        """
        B = encoder_out.shape[0]
        device = encoder_out.device
        vocab_size = self.output_proj.out_features

        # Process one batch item at a time for simplicity
        all_best_sequences = []

        for b in range(B):
            # Get encoder output for this item [1, N, D]
            enc_out = encoder_out[b:b+1]
            enc_mask = encoder_padding_mask[b:b+1] if encoder_padding_mask is not None else None

            # Expand for beam search [beam_size, N, D]
            enc_out_expanded = enc_out.expand(beam_size, -1, -1)
            enc_mask_expanded = enc_mask.expand(beam_size, -1) if enc_mask is not None else None

            # Initialize beams
            sequences = torch.full((beam_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
            scores = torch.zeros(beam_size, device=device)
            scores[1:] = float('-inf')  # Only first beam is active initially

            finished_seqs = []
            finished_scores = []
            cache = None

            for step in range(max_length - 1):
                # Get input for this step
                if use_cache and cache is not None:
                    input_ids = sequences[:, -1:]
                else:
                    input_ids = sequences

                # Forward pass
                logits, cache = self.forward(
                    enc_out_expanded, input_ids, enc_mask_expanded,
                    past_cache=cache if use_cache else None,
                    use_cache=use_cache,
                )
                next_logits = logits[:, -1, :]  # [beam_size, vocab_size]
                log_probs = F.log_softmax(next_logits, dim=-1)

                # Add current scores [beam_size, vocab_size]
                next_scores = scores.unsqueeze(-1) + log_probs

                # Flatten for top-k selection [beam_size * vocab_size]
                next_scores_flat = next_scores.view(-1)

                # Get top beam_size candidates
                top_scores, top_indices = torch.topk(next_scores_flat, beam_size, dim=-1)

                # Convert flat indices to beam and token indices
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size

                # Build new sequences
                new_sequences = torch.cat([
                    sequences[beam_indices],
                    token_indices.unsqueeze(-1)
                ], dim=1)

                # Reorder cache for new beam arrangement
                if use_cache and cache is not None:
                    cache = [
                        (k[:, :, beam_indices], v[:, :, beam_indices])
                        for k, v in cache
                    ]
                    # Reshape back: cache tensors are [B, H, T, head_dim]
                    # After indexing beam_indices, we need [beam_size, H, T, head_dim]
                    cache = [
                        (k.transpose(0, 2).contiguous().transpose(0, 2),
                         v.transpose(0, 2).contiguous().transpose(0, 2))
                        for k, v in cache
                    ]

                # Check for finished sequences (EOS token)
                is_eos = token_indices == tokenizer.eos_token_id

                for i in range(beam_size):
                    if is_eos[i]:
                        seq_len = new_sequences[i].shape[0]
                        final_score = top_scores[i] / (seq_len ** length_penalty)
                        finished_seqs.append(new_sequences[i].tolist())
                        finished_scores.append(final_score.item())

                # Continue with non-finished beams
                continuing = ~is_eos
                if continuing.sum() == 0:
                    break

                # Keep only continuing beams (refill if needed)
                if continuing.sum() < beam_size:
                    continuing_indices = continuing.nonzero(as_tuple=True)[0]
                    n_continuing = len(continuing_indices)
                    if n_continuing > 0:
                        pad_indices = continuing_indices[0].expand(beam_size - n_continuing)
                        all_indices = torch.cat([continuing_indices, pad_indices])
                        sequences = new_sequences[all_indices]
                        scores = top_scores[all_indices]
                        scores[n_continuing:] = float('-inf')

                        # Reorder cache for continuing beams
                        if use_cache and cache is not None:
                            cache = [
                                (k[all_indices], v[all_indices])
                                for k, v in cache
                            ]
                    else:
                        break
                else:
                    sequences = new_sequences
                    scores = top_scores

            # Add any remaining sequences
            for i in range(beam_size):
                if scores[i] > float('-inf'):
                    seq_len = sequences[i].shape[0]
                    final_score = scores[i] / (seq_len ** length_penalty)
                    finished_seqs.append(sequences[i].tolist())
                    finished_scores.append(final_score.item())

            # Select best sequence
            if finished_seqs:
                best_idx = max(range(len(finished_scores)), key=lambda i: finished_scores[i])
                all_best_sequences.append(finished_seqs[best_idx])
            else:
                # Fallback to first beam
                all_best_sequences.append(sequences[0].tolist())

        return all_best_sequences


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
