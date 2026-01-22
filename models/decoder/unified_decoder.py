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
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Self-attention projections (for KV caching)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
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
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Decoder input [B, T, D]
            encoder_out: Encoder features [B, N, D]
            tgt_mask: Causal mask [T, T]
            tgt_key_padding_mask: Padding mask for targets [B, T]
            memory_key_padding_mask: Padding mask for encoder [B, N]
            past_kv: Cached (key, value) from previous steps
            use_cache: Whether to return updated cache

        Returns:
            (output, new_cache) where new_cache is (key, value) if use_cache=True
        """
        B, T, D = x.shape

        # Self-attention with KV caching
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # If we have cached KV, concatenate
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        new_cache = (k, v) if use_cache else None

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores [B, H, T, S] where S is total sequence length
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Causal masking - only mask if we're processing more than one token
        if T > 1:
            S = k.shape[2]
            causal_mask = torch.triu(
                torch.ones(T, S, device=x.device, dtype=torch.bool),
                diagonal=S - T + 1
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x2 = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x2 = self.out_proj(x2)
        x = self.norm1(x + self.dropout(x2))

        # Cross-attention to encoder
        x2 = self.cross_attn(x, encoder_out, memory_key_padding_mask)
        x = self.norm2(x + self.dropout(x2))

        # Feed-forward
        x = self.norm3(x + self.ffn(x))

        return x, new_cache


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
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass for training or cached inference.

        Args:
            encoder_out: Encoder features [B, N, D]
            target_ids: Target token IDs [B, T]
            encoder_padding_mask: [B, N] True for padding positions
            target_padding_mask: [B, T] True for padding positions
            past_kv: List of (key, value) caches for each layer
            use_cache: Whether to return updated KV caches

        Returns:
            Tuple of (logits [B, T, vocab_size], new_kv_caches or None)
        """
        B, T = target_ids.shape

        # Embed tokens
        x = self.token_embedding(target_ids)  # [B, T, D]

        # Add positional encoding - offset by cache length if using cache
        if past_kv is not None and past_kv[0] is not None:
            past_length = past_kv[0][0].shape[1]
            # Create position tensor for just the new tokens
            positions = torch.arange(past_length, past_length + T, device=x.device)
            x = x + self.pos_encoding.pe[:, positions]
            x = self.pos_encoding.dropout(x)
        else:
            x = self.pos_encoding(x)

        # Prepare past_kv for each layer
        if past_kv is None:
            past_kv = [None] * len(self.layers)

        # Pass through decoder layers with caching
        new_kv_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past_kv = past_kv[i] if past_kv else None
            x, new_cache = layer(
                x, encoder_out,
                tgt_key_padding_mask=target_padding_mask,
                memory_key_padding_mask=encoder_padding_mask,
                past_kv=layer_past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_kv_caches.append(new_cache)

        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)  # [B, T, vocab_size]

        return logits, new_kv_caches

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
        """Autoregressive generation with KV-caching.

        Args:
            encoder_out: Encoder features [B, N, D]
            tokenizer: Tokenizer instance
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            encoder_padding_mask: [B, N] True for padding positions
            use_cache: Whether to use KV-caching (much faster)

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
        past_kv = None

        for _ in range(max_length - 1):
            # Only process the last token if using cache
            if use_cache and past_kv is not None:
                input_ids = generated[:, -1:]
            else:
                input_ids = generated

            # Forward pass with caching
            logits, past_kv = self.forward(
                encoder_out, input_ids, encoder_padding_mask,
                past_kv=past_kv, use_cache=use_cache
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
        """Greedy decoding with KV-caching for fast inference.

        Args:
            encoder_out: Encoder features [B, N, D]
            tokenizer: Tokenizer instance
            max_length: Maximum generation length
            encoder_padding_mask: [B, N] True for padding positions
            use_cache: Whether to use KV-caching (much faster)

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
        past_kv = None

        for _ in range(max_length - 1):
            # Only process the last token if using cache
            if use_cache and past_kv is not None:
                input_ids = generated[:, -1:]  # Just the last token
            else:
                input_ids = generated

            logits, past_kv = self.forward(
                encoder_out, input_ids, encoder_padding_mask,
                past_kv=past_kv, use_cache=use_cache
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
        """Beam search decoding with KV-caching for fast, high-quality generation.

        Args:
            encoder_out: Encoder features [B, N, D]
            tokenizer: Tokenizer instance
            beam_size: Number of beams to keep
            max_length: Maximum generation length
            length_penalty: Penalty for longer sequences (>1 favors longer)
            encoder_padding_mask: [B, N] True for padding positions
            use_cache: Whether to use KV-caching (much faster)

        Returns:
            List of best token sequences (one per batch item)
        """
        B = encoder_out.shape[0]
        device = encoder_out.device
        vocab_size = self.output_proj.out_features
        num_layers = len(self.layers)

        # Process one batch item at a time for simplicity
        all_best_sequences = []

        for b in range(B):
            # Get encoder output for this item [1, N, D]
            enc_out = encoder_out[b:b+1]
            enc_mask = encoder_padding_mask[b:b+1] if encoder_padding_mask is not None else None

            # Expand for beam search [beam_size, N, D]
            enc_out_expanded = enc_out.expand(beam_size, -1, -1)
            enc_mask_expanded = enc_mask.expand(beam_size, -1) if enc_mask is not None else None

            # Initialize beams with BOS token
            sequences = torch.full((beam_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
            scores = torch.zeros(beam_size, device=device)

            # Only first beam is active initially
            scores[1:] = float('-inf')

            # Initialize KV cache - None until first forward pass
            past_kv = None

            finished_seqs = []
            finished_scores = []

            for step in range(max_length - 1):
                # Determine input: full sequence on first step, just last token after
                if use_cache and past_kv is not None:
                    input_ids = sequences[:, -1:]  # [beam_size, 1]
                else:
                    input_ids = sequences  # [beam_size, seq_len]

                # Forward pass with caching
                logits, past_kv = self.forward(
                    enc_out_expanded, input_ids, enc_mask_expanded,
                    past_kv=past_kv, use_cache=use_cache
                )
                next_logits = logits[:, -1, :]  # [beam_size, vocab_size]
                log_probs = F.log_softmax(next_logits, dim=-1)

                # Add current scores [beam_size, vocab_size]
                next_scores = scores.unsqueeze(-1) + log_probs

                # Flatten for top-k selection
                next_scores_flat = next_scores.view(-1)

                # Get top 2*beam_size candidates to handle EOS tokens
                top_scores, top_indices = torch.topk(next_scores_flat, 2 * beam_size, dim=-1)

                # Convert flat indices to beam and token indices
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size

                # Process candidates
                new_beam_indices = []
                new_token_indices = []
                new_scores = []

                for i in range(2 * beam_size):
                    beam_idx = beam_indices[i].item()
                    token_idx = token_indices[i].item()
                    score = top_scores[i].item()

                    if token_idx == tokenizer.eos_token_id:
                        # Finished sequence
                        seq = sequences[beam_idx].tolist() + [token_idx]
                        final_score = score / (len(seq) ** length_penalty)
                        finished_seqs.append(seq)
                        finished_scores.append(final_score)
                    else:
                        # Continuing beam
                        if len(new_beam_indices) < beam_size:
                            new_beam_indices.append(beam_idx)
                            new_token_indices.append(token_idx)
                            new_scores.append(score)

                    if len(new_beam_indices) >= beam_size:
                        break

                # Stop if no continuing beams
                if len(new_beam_indices) == 0:
                    break

                # Convert to tensors
                new_beam_indices = torch.tensor(new_beam_indices, device=device)
                new_token_indices = torch.tensor(new_token_indices, device=device)
                scores = torch.tensor(new_scores, device=device)

                # Pad if fewer than beam_size continuing
                if len(new_beam_indices) < beam_size:
                    pad_size = beam_size - len(new_beam_indices)
                    new_beam_indices = torch.cat([
                        new_beam_indices,
                        new_beam_indices[0].expand(pad_size)
                    ])
                    new_token_indices = torch.cat([
                        new_token_indices,
                        torch.full((pad_size,), tokenizer.pad_token_id, device=device)
                    ])
                    scores = torch.cat([
                        scores,
                        torch.full((pad_size,), float('-inf'), device=device)
                    ])

                # Reorder sequences based on selected beams
                sequences = torch.cat([
                    sequences[new_beam_indices],
                    new_token_indices.unsqueeze(-1)
                ], dim=1)

                # Reorder KV cache based on selected beams
                if use_cache and past_kv is not None:
                    new_past_kv = []
                    for layer_cache in past_kv:
                        k, v = layer_cache
                        # Reorder along batch dimension (beam dimension)
                        new_k = k[new_beam_indices]
                        new_v = v[new_beam_indices]
                        new_past_kv.append((new_k, new_v))
                    past_kv = new_past_kv

            # Add any remaining sequences
            for i in range(min(beam_size, len(scores))):
                if scores[i] > float('-inf'):
                    seq_len = sequences[i].shape[0]
                    final_score = scores[i].item() / (seq_len ** length_penalty)
                    finished_seqs.append(sequences[i].tolist())
                    finished_scores.append(final_score)

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
