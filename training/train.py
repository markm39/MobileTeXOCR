#!/usr/bin/env python3
"""
Main training script for Handwritten LaTeX OCR.

Usage:
    python -m training.train --config configs/base.yaml
    python -m training.train --encoder fastvithd --batch_size 32
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import HandwrittenLaTeXOCR, ModelConfig
from models.decoder.tokenizer import LaTeXTokenizer, TokenizerConfig
from data import DatasetConfig, CombinedDataset, get_train_transforms, get_eval_transforms
from training import Trainer, TrainingConfig


def setup_logging(output_dir: Path, level: int = logging.INFO):
    """Setup logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(output_dir / 'train.log')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Handwritten LaTeX OCR')

    # Config file
    parser.add_argument('--config', type=str, help='Path to YAML config file')

    # Model settings
    parser.add_argument('--encoder', type=str, default='fastvithd',
                       choices=['fastvithd', 'perception'],
                       help='Encoder type')
    parser.add_argument('--encoder_size', type=str, default='base',
                       choices=['small', 'base', 'tiny'],
                       help='Encoder size variant')
    parser.add_argument('--image_size', type=int, default=384,
                       help='Input image size')
    parser.add_argument('--d_model', type=int, default=384,
                       help='Model dimension')
    parser.add_argument('--num_decoder_layers', type=int, default=6,
                       help='Number of decoder layers')

    # Data settings
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['mathwriting', 'crohme', 'hme100k'],
                       help='Datasets to use')

    # Training settings
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='latex_ocr',
                       help='Experiment name')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')

    # Mixed precision
    parser.add_argument('--no_amp', action='store_true',
                       help='Disable automatic mixed precision')
    parser.add_argument('--amp_dtype', type=str, default='float16',
                       choices=['float16', 'bfloat16'],
                       help='AMP dtype')

    # Logging
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='latex-ocr',
                       help='Wandb project name')

    # Checkpointing
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')

    # Hardware
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # Load config file if provided
    config_dict = {}
    if args.config:
        config_dict = load_config(args.config)

    # Override with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config_dict[key] = value

    # Create output directory
    output_dir = Path(config_dict.get('output_dir', './outputs'))
    experiment_name = config_dict.get('experiment_name', 'latex_ocr')
    experiment_dir = output_dir / experiment_name

    # Setup logging
    setup_logging(experiment_dir)
    logger = logging.getLogger(__name__)

    logger.info("Starting training with config:")
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")

    # Create tokenizer
    tokenizer = LaTeXTokenizer()
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Create model
    model_config = ModelConfig(
        encoder_type=config_dict.get('encoder', 'fastvithd'),
        encoder_size=config_dict.get('encoder_size', 'base'),
        image_size=config_dict.get('image_size', 384),
        d_model=config_dict.get('d_model', 384),
        num_decoder_layers=config_dict.get('num_decoder_layers', 6),
    )
    model = HandwrittenLaTeXOCR(model_config)
    logger.info(f"Created model with {model.count_parameters():,} parameters")

    # Create datasets
    dataset_config = DatasetConfig(
        data_dir=config_dict.get('data_dir', './data'),
        image_size=config_dict.get('image_size', 384),
    )

    train_transform = get_train_transforms(
        image_size=dataset_config.image_size,
        augment_strength='medium',
    )
    valid_transform = get_eval_transforms(image_size=dataset_config.image_size)

    train_dataset = CombinedDataset(
        dataset_config,
        split='train',
        transform=train_transform,
        tokenizer=tokenizer,
        datasets=config_dict.get('datasets', ['mathwriting', 'crohme', 'hme100k']),
    )

    val_dataset = CombinedDataset(
        dataset_config,
        split='val',
        transform=valid_transform,
        tokenizer=tokenizer,
        datasets=config_dict.get('datasets', ['mathwriting', 'crohme', 'hme100k']),
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = train_dataset.get_dataloader(
        batch_size=config_dict.get('batch_size', 32),
        num_workers=config_dict.get('num_workers', 4),
        use_weighted_sampling=True,
    )
    val_loader = val_dataset.get_dataloader(
        batch_size=config_dict.get('batch_size', 32),
        shuffle=False,
        num_workers=config_dict.get('num_workers', 4),
        use_weighted_sampling=False,
    )

    # Create training config
    training_config = TrainingConfig(
        output_dir=str(output_dir),
        experiment_name=experiment_name,
        num_epochs=config_dict.get('num_epochs', 20),
        batch_size=config_dict.get('batch_size', 32),
        learning_rate=config_dict.get('learning_rate', 1e-4),
        weight_decay=config_dict.get('weight_decay', 0.01),
        warmup_steps=config_dict.get('warmup_steps', 1000),
        gradient_accumulation_steps=config_dict.get('gradient_accumulation_steps', 1),
        use_amp=not config_dict.get('no_amp', False),
        amp_dtype=config_dict.get('amp_dtype', 'float16'),
        use_wandb=config_dict.get('wandb', False),
        wandb_project=config_dict.get('wandb_project', 'latex-ocr'),
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        tokenizer=tokenizer,
    )

    # Resume from checkpoint if specified
    if config_dict.get('resume'):
        trainer.load_checkpoint(config_dict['resume'])

    # Train
    best_metric = trainer.train()
    logger.info(f"Training complete! Best metric: {best_metric:.4f}")

    # Save final model
    model.save_pretrained(str(experiment_dir / "final_model"))
    logger.info(f"Saved final model to {experiment_dir / 'final_model'}")


if __name__ == '__main__':
    main()
