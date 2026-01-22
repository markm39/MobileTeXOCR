#!/usr/bin/env python3
"""
Evaluation Script for Handwritten LaTeX OCR

Tests a trained model on the test/validation set and reports metrics.

Usage:
    python scripts/eval.py --checkpoint outputs/latex_ocr_15m/checkpoints/best.pt
    python scripts/eval.py --checkpoint outputs/latex_ocr_15m/checkpoints/best.pt --decode beam --beam_size 5
    python scripts/eval.py --checkpoint outputs/latex_ocr_15m/checkpoints/best.pt --num_samples 100
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Optional

import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import HandwrittenLaTeXOCR
from models.decoder.tokenizer import LaTeXTokenizer
from data import DatasetConfig, CombinedDataset, get_eval_transforms
from training.metrics import compute_metrics


def load_model(checkpoint_path: str, device: torch.device) -> HandwrittenLaTeXOCR:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try to load config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_config = config.get('model', {})
    else:
        # Default config
        model_config = {}

    from models import ModelConfig
    model = HandwrittenLaTeXOCR(ModelConfig(
        encoder_type=model_config.get('encoder_type', 'fastvithd'),
        encoder_size=model_config.get('encoder_size', 'small'),
        image_size=model_config.get('image_size', 384),
        d_model=model_config.get('d_model', 384),
        num_decoder_layers=model_config.get('num_decoder_layers', 6),
        num_heads=model_config.get('num_heads', 8),
    ))

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Set to inference mode
    model.train(False)

    return model


def check_prediction_diversity(predictions: List[str]) -> Dict[str, Any]:
    """Analyze prediction diversity to detect mode collapse."""
    if not predictions:
        return {'unique_ratio': 0.0, 'is_diverse': False}

    unique_preds = set(predictions)
    unique_ratio = len(unique_preds) / len(predictions)

    counter = Counter(predictions)
    most_common = counter.most_common(5)

    return {
        'unique_ratio': unique_ratio,
        'num_unique': len(unique_preds),
        'total': len(predictions),
        'most_common': most_common,
        'is_diverse': unique_ratio >= 0.5,
    }


def run_test(
    model: HandwrittenLaTeXOCR,
    dataloader,
    decode_method: str = 'greedy',
    beam_size: int = 5,
    length_penalty: float = 1.0,
    num_samples: Optional[int] = None,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run model test on dataset.

    Args:
        model: Trained model
        dataloader: Data loader for testing
        decode_method: 'greedy' or 'beam'
        beam_size: Beam size for beam search
        length_penalty: Length penalty for beam search
        num_samples: Limit number of samples (None for all)
        device: Device to run on
        verbose: Whether to print progress

    Returns:
        Dictionary of test metrics
    """
    model.train(False)

    all_predictions = []
    all_targets = []
    all_token_ids = []
    total_time = 0.0
    num_processed = 0

    iterator = tqdm(dataloader, desc="Testing") if verbose else dataloader

    with torch.no_grad():
        for batch in iterator:
            if num_samples and num_processed >= num_samples:
                break

            images = batch['images'].to(device)
            targets = batch['latex']

            batch_size = images.size(0)

            # Generate predictions
            start_time = time.time()

            if decode_method == 'beam':
                output = model.beam_search(
                    images,
                    beam_size=beam_size,
                    max_length=256,
                    length_penalty=length_penalty,
                    use_cache=True,
                )
            else:
                output = model(images)  # Uses greedy by default

            elapsed = time.time() - start_time
            total_time += elapsed

            # Extract predictions
            for pred_regions in output.predictions:
                if pred_regions:
                    pred_latex = ' '.join(region[1] for region in pred_regions)
                else:
                    pred_latex = ""
                all_predictions.append(pred_latex)

            if output.token_ids:
                all_token_ids.extend(output.token_ids)

            all_targets.extend(targets[:batch_size])
            num_processed += batch_size

            if num_samples and num_processed >= num_samples:
                break

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)

    # Add timing info
    metrics['total_time'] = total_time
    metrics['samples_per_second'] = num_processed / total_time if total_time > 0 else 0
    metrics['avg_time_per_sample'] = total_time / num_processed if num_processed > 0 else 0
    metrics['num_samples'] = num_processed

    # Check diversity
    diversity = check_prediction_diversity(all_predictions)
    metrics['prediction_diversity'] = diversity['unique_ratio']
    metrics['is_diverse'] = diversity['is_diverse']

    return {
        'metrics': metrics,
        'diversity': diversity,
        'predictions': all_predictions,
        'targets': all_targets,
    }


def print_results(results: Dict[str, Any], decode_method: str):
    """Print test results."""
    metrics = results['metrics']
    diversity = results['diversity']

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"\nDecode method: {decode_method}")
    print(f"Samples tested: {metrics['num_samples']}")
    print(f"Total time: {metrics['total_time']:.2f}s")
    print(f"Speed: {metrics['samples_per_second']:.1f} samples/sec")
    print(f"Avg time per sample: {metrics['avg_time_per_sample']*1000:.1f}ms")

    print("\n--- Accuracy Metrics ---")
    print(f"Expression Rate (ExpRate): {metrics.get('exp_rate', 0)*100:.2f}%")
    print(f"Symbol Accuracy: {metrics.get('symbol_accuracy', 0)*100:.2f}%")

    print("\n--- Diversity Check ---")
    print(f"Unique predictions: {diversity['num_unique']}/{diversity['total']} ({diversity['unique_ratio']*100:.1f}%)")
    if not diversity['is_diverse']:
        print("WARNING: Predictions lack diversity - possible mode collapse!")

    if diversity['most_common']:
        print("\nMost common predictions:")
        for pred, count in diversity['most_common'][:3]:
            pct = count / diversity['total'] * 100
            display = pred[:60] + "..." if len(pred) > 60 else pred
            print(f"  ({pct:.1f}%) {display}")

    print("\n--- Sample Predictions ---")
    predictions = results['predictions']
    targets = results['targets']
    for i in range(min(5, len(predictions))):
        print(f"\n[{i}] Target: {targets[i][:80]}...")
        print(f"[{i}] Pred:   {predictions[i][:80]}...")
        match = "MATCH" if predictions[i] == targets[i] else "DIFF"
        print(f"[{i}] Result: {match}")


def main():
    parser = argparse.ArgumentParser(description='Test Handwritten LaTeX OCR Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--datasets', type=str, nargs='+', default=['crohme'], help='Datasets to test on')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Data split')
    parser.add_argument('--decode', type=str, default='greedy', choices=['greedy', 'beam'], help='Decode method')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
    parser.add_argument('--length_penalty', type=float, default=1.0, help='Length penalty for beam search')
    parser.add_argument('--num_samples', type=int, help='Limit number of samples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress bar')
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Create dataset
    tokenizer = LaTeXTokenizer()
    dataset_config = DatasetConfig(
        data_dir=args.data_dir,
        image_size=384,
    )
    transform = get_eval_transforms(image_size=384)

    try:
        dataset = CombinedDataset(
            dataset_config,
            split=args.split,
            transform=transform,
            tokenizer=tokenizer,
            datasets=args.datasets,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print(f"Dataset: {args.datasets} ({args.split})")
    print(f"Samples: {len(dataset)}")

    # Create dataloader
    dataloader = dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        use_weighted_sampling=False,
    )

    # Run test
    results = run_test(
        model=model,
        dataloader=dataloader,
        decode_method=args.decode,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        num_samples=args.num_samples,
        device=device,
        verbose=not args.quiet,
    )

    # Print results
    print_results(results, args.decode)

    # Save results if requested
    if args.output:
        output_data = {
            'checkpoint': args.checkpoint,
            'datasets': args.datasets,
            'split': args.split,
            'decode_method': args.decode,
            'beam_size': args.beam_size if args.decode == 'beam' else None,
            'metrics': results['metrics'],
            'diversity': results['diversity'],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
