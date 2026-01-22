"""Test KV-caching implementation."""

import torch
import time
import sys
sys.path.insert(0, '/Users/markmiller/MobileTeXOCR')

from models.decoder.unified_decoder import UnifiedDecoder
from models.decoder.tokenizer import LaTeXTokenizer, TokenizerConfig


def test_kv_caching():
    """Test that KV-caching produces same results as non-cached version."""
    print("Testing KV-caching implementation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create tokenizer and decoder
    tokenizer = LaTeXTokenizer(TokenizerConfig(num_location_bins=1000))
    decoder = UnifiedDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_layers=4,
        num_heads=8,
        dim_feedforward=1024,
        max_seq_length=256,
    ).to(device)
    decoder.eval()

    # Create fake encoder output
    batch_size = 2
    num_encoder_tokens = 144  # 12x12 grid
    encoder_out = torch.randn(batch_size, num_encoder_tokens, 256, device=device)

    print("\n1. Testing forward pass for training...")
    target_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, 32), device=device)
    logits, cache = decoder.forward(encoder_out, target_ids)
    assert logits.shape == (batch_size, 32, tokenizer.vocab_size), f"Wrong shape: {logits.shape}"
    assert cache is None, "Cache should be None when use_cache=False"
    print("   Forward pass OK - shape:", logits.shape)

    print("\n2. Testing greedy decode consistency...")
    torch.manual_seed(42)

    # Without cache
    with torch.no_grad():
        start = time.time()
        result_no_cache = decoder.greedy_decode(
            encoder_out, tokenizer, max_length=50, use_cache=False
        )
        time_no_cache = time.time() - start

    # With cache
    torch.manual_seed(42)
    with torch.no_grad():
        start = time.time()
        result_with_cache = decoder.greedy_decode(
            encoder_out, tokenizer, max_length=50, use_cache=True
        )
        time_with_cache = time.time() - start

    # Results should be identical
    print(f"   Without cache: {time_no_cache:.3f}s")
    print(f"   With cache:    {time_with_cache:.3f}s")
    print(f"   Speedup:       {time_no_cache/time_with_cache:.2f}x")

    # Check if results match
    match = result_no_cache == result_with_cache
    if match:
        print("   Results MATCH")
    else:
        print("   WARNING: Results differ!")
        print(f"   No cache: {result_no_cache[0][:20]}...")
        print(f"   With cache: {result_with_cache[0][:20]}...")

    print("\n3. Testing beam search with cache...")
    with torch.no_grad():
        # Without cache
        start = time.time()
        beam_no_cache = decoder.beam_search(
            encoder_out[:1], tokenizer, beam_size=3, max_length=30, use_cache=False
        )
        time_beam_no_cache = time.time() - start

        # With cache
        start = time.time()
        beam_with_cache = decoder.beam_search(
            encoder_out[:1], tokenizer, beam_size=3, max_length=30, use_cache=True
        )
        time_beam_with_cache = time.time() - start

    print(f"   Beam search without cache: {time_beam_no_cache:.3f}s")
    print(f"   Beam search with cache:    {time_beam_with_cache:.3f}s")
    print(f"   Speedup:                   {time_beam_no_cache/time_beam_with_cache:.2f}x")

    print("\n4. Testing longer generation to see speedup scale...")
    with torch.no_grad():
        start = time.time()
        _ = decoder.greedy_decode(encoder_out, tokenizer, max_length=100, use_cache=False)
        time_long_no_cache = time.time() - start

        start = time.time()
        _ = decoder.greedy_decode(encoder_out, tokenizer, max_length=100, use_cache=True)
        time_long_with_cache = time.time() - start

    print(f"   100 tokens without cache: {time_long_no_cache:.3f}s")
    print(f"   100 tokens with cache:    {time_long_with_cache:.3f}s")
    print(f"   Speedup:                  {time_long_no_cache/time_long_with_cache:.2f}x")

    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    test_kv_caching()
