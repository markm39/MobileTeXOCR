#!/usr/bin/env python3
"""
Decoding Benchmark Script

Benchmarks KV-caching speedup for greedy and beam search decoding.
Verifies that cached and non-cached decoding produce identical results.

Usage:
    python scripts/benchmark_decode.py
    python scripts/benchmark_decode.py --checkpoint outputs/latex_ocr_15m/checkpoints/best.pt
    python scripts/benchmark_decode.py --max_length 100 --batch_size 4
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.decoder.unified_decoder import UnifiedDecoder
from models.decoder.tokenizer import LaTeXTokenizer, TokenizerConfig
from models import HandwrittenLaTeXOCR, ModelConfig


def benchmark_decoder_standalone(
    d_model: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    batch_size: int = 2,
    num_encoder_tokens: int = 144,
    max_length: int = 50,
    device: torch.device = torch.device('cpu'),
):
    """Benchmark decoder with synthetic encoder output."""
    print("\n" + "=" * 60)
    print("DECODER BENCHMARK (Standalone)")
    print("=" * 60)

    tokenizer = LaTeXTokenizer(TokenizerConfig(num_location_bins=1000))
    decoder = UnifiedDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=d_model * 4,
        max_seq_length=256,
    ).to(device)

    # Set to inference mode
    decoder.train(False)

    params = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder parameters: {params:,}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Encoder tokens: {num_encoder_tokens}")
    print(f"Max generation length: {max_length}")

    # Create synthetic encoder output
    encoder_out = torch.randn(batch_size, num_encoder_tokens, d_model, device=device)

    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        _ = decoder.greedy_decode(encoder_out, tokenizer, max_length=10, use_cache=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Benchmark greedy decoding
    print("\n--- Greedy Decoding ---")

    # Without cache
    torch.manual_seed(42)
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        result_no_cache = decoder.greedy_decode(
            encoder_out, tokenizer, max_length=max_length, use_cache=False
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_no_cache = time.time() - start

    # With cache
    torch.manual_seed(42)
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        result_with_cache = decoder.greedy_decode(
            encoder_out, tokenizer, max_length=max_length, use_cache=True
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_with_cache = time.time() - start

    speedup = time_no_cache / time_with_cache if time_with_cache > 0 else 0
    print(f"Without cache: {time_no_cache:.3f}s ({max_length/time_no_cache:.0f} tokens/s)")
    print(f"With cache:    {time_with_cache:.3f}s ({max_length/time_with_cache:.0f} tokens/s)")
    print(f"Speedup:       {speedup:.2f}x")

    # Check consistency
    match = result_no_cache == result_with_cache
    if match:
        print("Results: MATCH (cached = non-cached)")
    else:
        print("Results: DIFFER (warning: cache may have bug)")
        print(f"  No cache first 20: {result_no_cache[0][:20]}")
        print(f"  With cache first 20: {result_with_cache[0][:20]}")

    # Benchmark beam search
    print("\n--- Beam Search (beam_size=3) ---")

    # Without cache
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        beam_no_cache = decoder.beam_search(
            encoder_out[:1], tokenizer, beam_size=3, max_length=max_length, use_cache=False
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_beam_no_cache = time.time() - start

    # With cache
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        beam_with_cache = decoder.beam_search(
            encoder_out[:1], tokenizer, beam_size=3, max_length=max_length, use_cache=True
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_beam_with_cache = time.time() - start

    speedup_beam = time_beam_no_cache / time_beam_with_cache if time_beam_with_cache > 0 else 0
    print(f"Without cache: {time_beam_no_cache:.3f}s")
    print(f"With cache:    {time_beam_with_cache:.3f}s")
    print(f"Speedup:       {speedup_beam:.2f}x")

    # Longer sequence test
    print("\n--- Longer Sequences (100 tokens) ---")
    long_length = 100

    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        _ = decoder.greedy_decode(encoder_out, tokenizer, max_length=long_length, use_cache=False)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_long_no_cache = time.time() - start

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        _ = decoder.greedy_decode(encoder_out, tokenizer, max_length=long_length, use_cache=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_long_with_cache = time.time() - start

    speedup_long = time_long_no_cache / time_long_with_cache if time_long_with_cache > 0 else 0
    print(f"Without cache: {time_long_no_cache:.3f}s")
    print(f"With cache:    {time_long_with_cache:.3f}s")
    print(f"Speedup:       {speedup_long:.2f}x")

    return {
        'greedy_speedup': speedup,
        'beam_speedup': speedup_beam,
        'long_speedup': speedup_long,
        'results_match': match,
    }


def benchmark_full_model(
    checkpoint_path: Optional[str],
    batch_size: int = 2,
    max_length: int = 50,
    device: torch.device = torch.device('cpu'),
):
    """Benchmark full model with checkpoint."""
    print("\n" + "=" * 60)
    print("FULL MODEL BENCHMARK")
    print("=" * 60)

    if checkpoint_path:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {}).get('model', {})

        model = HandwrittenLaTeXOCR(ModelConfig(
            encoder_type=config.get('encoder_type', 'fastvithd'),
            encoder_size=config.get('encoder_size', 'small'),
            image_size=config.get('image_size', 384),
            d_model=config.get('d_model', 384),
            num_decoder_layers=config.get('num_decoder_layers', 6),
            num_heads=config.get('num_heads', 8),
        ))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("\nUsing default model config")
        model = HandwrittenLaTeXOCR(ModelConfig(
            encoder_type='fastvithd',
            encoder_size='small',
            d_model=256,
            num_decoder_layers=4,
        ))

    model = model.to(device)
    model.train(False)

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")

    # Create synthetic images
    image_size = 384
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)

    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        _ = model(images[:1])
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Benchmark greedy
    print("\n--- Greedy Decoding (Full Model) ---")

    with torch.no_grad():
        # Get encoder output for fair comparison
        encoder_out = model.encoder(images)
        encoder_features = encoder_out.features

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        result_no_cache = model.decoder.greedy_decode(
            encoder_features,
            model.tokenizer,
            max_length=max_length,
            use_cache=False,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_no_cache = time.time() - start

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        result_with_cache = model.decoder.greedy_decode(
            encoder_features,
            model.tokenizer,
            max_length=max_length,
            use_cache=True,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_with_cache = time.time() - start

    speedup = time_no_cache / time_with_cache if time_with_cache > 0 else 0
    print(f"Without cache: {time_no_cache:.3f}s")
    print(f"With cache:    {time_with_cache:.3f}s")
    print(f"Speedup:       {speedup:.2f}x")

    # Benchmark beam search
    print("\n--- Beam Search (Full Model, beam_size=5) ---")

    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        _ = model.beam_search(images[:1], beam_size=5, max_length=max_length, use_cache=False)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_beam_no_cache = time.time() - start

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        _ = model.beam_search(images[:1], beam_size=5, max_length=max_length, use_cache=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_beam_with_cache = time.time() - start

    speedup_beam = time_beam_no_cache / time_beam_with_cache if time_beam_with_cache > 0 else 0
    print(f"Without cache: {time_beam_no_cache:.3f}s")
    print(f"With cache:    {time_beam_with_cache:.3f}s")
    print(f"Speedup:       {speedup_beam:.2f}x")

    return {
        'greedy_speedup': speedup,
        'beam_speedup': speedup_beam,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark KV-caching speedup')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint (optional)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--max_length', type=int, default=50, help='Max generation length')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension (standalone)')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers (standalone)')
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Run standalone decoder benchmark
    standalone_results = benchmark_decoder_standalone(
        d_model=args.d_model,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    # Run full model benchmark if checkpoint provided or just with defaults
    full_model_results = benchmark_full_model(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nStandalone Decoder:")
    print(f"  Greedy speedup: {standalone_results['greedy_speedup']:.2f}x")
    print(f"  Beam speedup:   {standalone_results['beam_speedup']:.2f}x")
    print(f"  Long seq speedup: {standalone_results['long_speedup']:.2f}x")
    print(f"  Results match:  {standalone_results['results_match']}")

    print("\nFull Model:")
    print(f"  Greedy speedup: {full_model_results['greedy_speedup']:.2f}x")
    print(f"  Beam speedup:   {full_model_results['beam_speedup']:.2f}x")

    expected_speedup = 2.0
    if standalone_results['greedy_speedup'] >= expected_speedup:
        print(f"\nKV-caching is working well (>= {expected_speedup}x speedup expected)")
    else:
        print(f"\nWARNING: Speedup is lower than expected ({expected_speedup}x)")
        print("This may indicate a problem with the implementation or measurement noise on CPU")


if __name__ == '__main__':
    main()
