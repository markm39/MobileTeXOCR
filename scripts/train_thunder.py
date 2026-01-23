#!/usr/bin/env python3
"""
Standalone Training Script for Thunder Compute A100

This script is designed for training on Thunder Compute with:
- Local data storage (not Google Drive)
- Better monitoring to catch mode collapse early
- A100-optimized batch sizes and mixed precision
- Checkpoint resumption support

Usage:
    python scripts/train_thunder.py --config configs/15m_base.yaml
    python scripts/train_thunder.py --config configs/15m_base.yaml --resume outputs/latex_ocr_15m/checkpoints/latest.pt
"""

import argparse
import logging
import json
import sys
import time
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import HandwrittenLaTeXOCR, ModelConfig
from models.decoder.tokenizer import LaTeXTokenizer, TokenizerConfig
from data import DatasetConfig, CombinedDataset, get_train_transforms, get_eval_transforms
from training.metrics import compute_metrics


def setup_logging(output_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Setup logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(output_dir / 'train.log')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger('train_thunder')
    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


@dataclass
class TrainingState:
    """Mutable training state for checkpointing."""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    patience_counter: int = 0
    history: List[Dict] = field(default_factory=list)


class ThunderTrainer:
    """Training loop optimized for Thunder Compute A100."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        tokenizer: LaTeXTokenizer,
        output_dir: Path,
        logger: logging.Logger,
        lr_override: Optional[float] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.logger = logger
        self.lr_override = lr_override

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")

        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Training config shortcuts
        self.train_config = config.get('training', {})
        self.opt_config = config.get('optimizer', {})
        self.amp_config = config.get('amp', {})
        self.monitor_config = config.get('monitoring', {})

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Mixed precision
        amp_enabled = self.amp_config.get('enabled', True)
        amp_dtype_str = self.amp_config.get('dtype', 'bfloat16')
        self.amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16
        self.use_amp = amp_enabled
        self.scaler = GradScaler('cuda') if amp_enabled and self.amp_dtype == torch.float16 else None

        # Training state
        self.state = TrainingState()

        # Checkpointing
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {'params': decay_params, 'weight_decay': self.opt_config.get('weight_decay', 0.01)},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=self.opt_config.get('learning_rate', 1e-4),
            betas=(0.9, 0.98),
            eps=1e-8,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = len(self.train_loader) * self.train_config.get('num_epochs', 20)
        warmup_steps = self.opt_config.get('warmup_steps', 2000)

        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.opt_config.get('learning_rate', 1e-4) * 0.01,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    def check_prediction_diversity(self, predictions: List[str]) -> Dict[str, Any]:
        """Check for mode collapse by analyzing prediction diversity.

        Returns metrics about prediction uniqueness.
        """
        if not predictions:
            return {'unique_ratio': 0.0, 'most_common': None, 'is_collapsed': True}

        unique_preds = set(predictions)
        unique_ratio = len(unique_preds) / len(predictions)

        counter = Counter(predictions)
        most_common = counter.most_common(3)

        min_unique_ratio = self.monitor_config.get('min_unique_ratio', 0.5)
        is_collapsed = unique_ratio < min_unique_ratio

        return {
            'unique_ratio': unique_ratio,
            'num_unique': len(unique_preds),
            'total': len(predictions),
            'most_common': most_common,
            'is_collapsed': is_collapsed,
        }

    def train(self) -> float:
        """Run full training loop."""
        num_epochs = self.train_config.get('num_epochs', 20)
        freeze_epochs = self.config.get('encoder_training', {}).get('freeze_encoder_epochs', 1)

        start_epoch = self.state.epoch
        self.logger.info(f"Starting training: epochs {start_epoch}-{num_epochs-1}")
        self.logger.info(f"  Batches/epoch: {len(self.train_loader)}")
        self.logger.info(f"  Starting step: {self.state.global_step}")

        for epoch in range(start_epoch, num_epochs):
            self.state.epoch = epoch
            epoch_start = time.time()

            # Handle encoder freezing
            if epoch < freeze_epochs:
                self.model.freeze_encoder()
                self.logger.info(f"Epoch {epoch}: Encoder frozen")
            elif epoch == freeze_epochs:
                self.model.unfreeze_encoder()
                self.logger.info(f"Epoch {epoch}: Encoder unfrozen")

            # Train epoch
            train_metrics = self._train_epoch()
            epoch_time = time.time() - epoch_start

            self.logger.info(
                f"Epoch {epoch} train: loss={train_metrics['loss']:.4f}, "
                f"time={epoch_time:.1f}s"
            )

            # Validation
            if self.val_loader is not None:
                val_metrics = self._validate()
                self.logger.info(
                    f"Epoch {epoch} val: loss={val_metrics['loss']:.4f}, "
                    f"exp_rate={val_metrics.get('exp_rate', 0):.4f}"
                )

                # Save history
                self.state.history.append({
                    'epoch': epoch,
                    'step': self.state.global_step,
                    'train': train_metrics,
                    'val': val_metrics,
                })
                self._save_history()

                # Check for improvement
                metric = self.config.get('early_stopping', {}).get('metric', 'exp_rate')
                current = val_metrics.get(metric, 0)
                min_delta = self.config.get('early_stopping', {}).get('min_delta', 0.001)

                if current > self.state.best_metric + min_delta:
                    self.state.best_metric = current
                    self.state.patience_counter = 0
                    self._save_checkpoint("best")
                    self.logger.info(f"New best {metric}: {current:.4f}")
                else:
                    self.state.patience_counter += 1

                # Early stopping
                patience = self.config.get('early_stopping', {}).get('patience', 5)
                if self.state.patience_counter >= patience:
                    self.logger.info(f"Early stopping after {epoch + 1} epochs")
                    break

            # Save epoch checkpoint
            self._save_checkpoint(f"epoch_{epoch}")
            self._save_checkpoint("latest")  # Always keep latest

        self.logger.info(f"Training complete! Best metric: {self.state.best_metric:.4f}")
        return self.state.best_metric

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with mode collapse monitoring."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        log_steps = self.train_config.get('log_steps', 100)
        diversity_steps = self.monitor_config.get('diversity_check_steps', 500)
        grad_accum = self.train_config.get('gradient_accumulation_steps', 1)
        max_grad_norm = self.train_config.get('max_grad_norm', 1.0)

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self._train_step(batch, grad_accum, max_grad_norm)
            total_loss += loss
            num_batches += 1

            # Logging
            if self.state.global_step % log_steps == 0 and self.state.global_step > 0:
                lr = self.optimizer.param_groups[0]['lr']
                avg_loss = total_loss / num_batches
                self.logger.info(
                    f"Step {self.state.global_step}: loss={loss:.4f}, "
                    f"avg_loss={avg_loss:.4f}, lr={lr:.2e}"
                )

            # Diversity check (mode collapse detection)
            if (self.state.global_step % diversity_steps == 0 and
                self.state.global_step > 0 and
                self.monitor_config.get('log_sample_predictions', True)):
                self._check_diversity_on_batch(batch)

            # Periodic checkpointing
            save_steps = self.config.get('checkpointing', {}).get('save_steps', 2000)
            if self.state.global_step % save_steps == 0 and self.state.global_step > 0:
                self._save_checkpoint(f"step_{self.state.global_step}")
                self._cleanup_checkpoints()

        return {'loss': total_loss / max(num_batches, 1)}

    def _train_step(self, batch: Dict[str, torch.Tensor], grad_accum: int, max_grad_norm: float) -> float:
        """Single training step."""
        images = batch['images'].to(self.device)
        token_ids = batch['token_ids'].to(self.device)
        padding_mask = batch.get('padding_mask')
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)

        # Forward pass with mixed precision
        with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            output = self.model(images, token_ids, target_padding_mask=padding_mask)
            loss = output.loss / grad_accum

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation step
        if (self.state.global_step + 1) % grad_accum == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            if self.scheduler:
                self.scheduler.step()

        self.state.global_step += 1

        return loss.item() * grad_accum

    @torch.no_grad()
    def _check_diversity_on_batch(self, batch: Dict[str, torch.Tensor]):
        """Check prediction diversity on a batch to detect mode collapse."""
        self.model.eval()

        images = batch['images'][:8].to(self.device)  # Check first 8 samples
        targets = batch['latex'][:8]

        # Generate predictions
        output = self.model(images)
        predictions = []
        for pred_regions in output.predictions:
            if pred_regions:
                pred_latex = ' '.join(region[1] for region in pred_regions)
            else:
                pred_latex = ""
            predictions.append(pred_latex)

        # Check diversity
        diversity = self.check_prediction_diversity(predictions)

        self.logger.info(f"Diversity check at step {self.state.global_step}:")
        self.logger.info(f"  Unique predictions: {diversity['num_unique']}/{diversity['total']} ({diversity['unique_ratio']:.2%})")

        if diversity['is_collapsed']:
            self.logger.warning("  MODE COLLAPSE DETECTED! Predictions are not diverse.")
            self.logger.warning(f"  Most common: {diversity['most_common']}")

        # Log sample predictions
        self.logger.info("  Sample predictions:")
        for i, (pred, target) in enumerate(zip(predictions[:3], targets[:3])):
            self.logger.info(f"    [{i}] Pred: {pred[:80]}...")
            self.logger.info(f"    [{i}] True: {target[:80]}...")

        self.model.train()

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            images = batch['images'].to(self.device)
            token_ids = batch['token_ids'].to(self.device)
            targets = batch['latex']

            # Compute loss
            with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(images, token_ids)
                total_loss += output.loss.item()

            # Generate predictions (using KV-cache for speed)
            pred_output = self.model(images)
            for pred_regions in pred_output.predictions:
                if pred_regions:
                    pred_latex = ' '.join(region[1] for region in pred_regions)
                else:
                    pred_latex = ""
                all_predictions.append(pred_latex)

            all_targets.extend(targets)
            num_batches += 1

        # Compute metrics
        metrics = compute_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / max(num_batches, 1)

        # Check diversity across validation set
        diversity = self.check_prediction_diversity(all_predictions)
        metrics['pred_diversity'] = diversity['unique_ratio']

        return metrics

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'state': {
                'epoch': self.state.epoch,
                'global_step': self.state.global_step,
                'best_metric': self.state.best_metric,
                'patience_counter': self.state.patience_counter,
            },
            'config': self.config,
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(f"Saved checkpoint: {checkpoint_path}")

    def _cleanup_checkpoints(self):
        """Keep only recent step checkpoints."""
        limit = self.config.get('checkpointing', {}).get('save_total_limit', 5)
        checkpoints = sorted(
            self.checkpoint_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split('_')[1]),
        )

        if len(checkpoints) > limit:
            for checkpoint in checkpoints[:-limit]:
                checkpoint.unlink()
                self.logger.debug(f"Removed old checkpoint: {checkpoint}")

    def _save_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.state.history, f, indent=2)

    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # If LR was overridden, don't restore scheduler (use fresh schedule with new LR)
        if self.lr_override:
            self.logger.info(f"LR override: {self.lr_override} - using fresh scheduler")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr_override
            # Recreate scheduler with new LR
            self.scheduler = self._create_scheduler()
        elif self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        state = checkpoint.get('state', {})
        self.state.epoch = state.get('epoch', 0) + 1  # Resume from next epoch
        self.state.global_step = state.get('global_step', 0)
        self.state.best_metric = state.get('best_metric', 0.0)
        self.state.patience_counter = state.get('patience_counter', 0)

        # Load history if available
        history_path = self.output_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.state.history = json.load(f)

        self.logger.info(
            f"Loaded checkpoint: epoch {self.state.epoch}, "
            f"step {self.state.global_step}, best_metric {self.state.best_metric:.4f}"
        )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train Handwritten LaTeX OCR on Thunder Compute')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--num_epochs', type=int, help='Override number of epochs')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--lr', type=float, help='Override learning rate (useful when resuming)')
    parser.add_argument('--max_grad_norm', type=float, help='Override gradient clipping threshold')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.data_dir:
        config.setdefault('data', {})['data_dir'] = args.data_dir
    if args.output_dir:
        config.setdefault('training', {})['output_dir'] = args.output_dir
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.num_epochs:
        config.setdefault('training', {})['num_epochs'] = args.num_epochs
    if args.wandb:
        config.setdefault('logging', {})['use_wandb'] = True
    if args.lr:
        config.setdefault('optimizer', {})['learning_rate'] = args.lr
    if args.max_grad_norm:
        config.setdefault('training', {})['max_grad_norm'] = args.max_grad_norm

    # Setup output directory
    train_config = config.get('training', {})
    output_dir = Path(train_config.get('output_dir', './outputs'))
    experiment_name = train_config.get('experiment_name', 'latex_ocr')
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(experiment_dir)
    logger.info("Thunder Compute Training Script")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {experiment_dir}")

    # Save config for reproducibility
    with open(experiment_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Create tokenizer
    tokenizer = LaTeXTokenizer()
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Create model
    model_config = config.get('model', {})
    model = HandwrittenLaTeXOCR(ModelConfig(
        encoder_type=model_config.get('encoder_type', 'fastvithd'),
        encoder_size=model_config.get('encoder_size', 'small'),
        image_size=model_config.get('image_size', 384),
        d_model=model_config.get('d_model', 384),
        num_decoder_layers=model_config.get('num_decoder_layers', 6),
        num_heads=model_config.get('num_heads', 8),
        decoder_dropout=model_config.get('dropout', 0.1),
        label_smoothing=train_config.get('label_smoothing', 0.1),
    ))
    logger.info(f"Created model with {model.count_parameters():,} parameters")

    # Create datasets
    data_config = config.get('data', {})
    dataset_config = DatasetConfig(
        data_dir=data_config.get('data_dir', './data'),
        image_size=data_config.get('image_size', 384),
    )

    train_transform = get_train_transforms(
        image_size=dataset_config.image_size,
        augment_strength=data_config.get('augment_strength', 'medium'),
    )
    valid_transform = get_eval_transforms(image_size=dataset_config.image_size)

    try:
        train_dataset = CombinedDataset(
            dataset_config,
            split='train',
            transform=train_transform,
            tokenizer=tokenizer,
            datasets=data_config.get('datasets', ['mathwriting', 'crohme', 'hme100k']),
        )

        val_dataset = CombinedDataset(
            dataset_config,
            split='val',
            transform=valid_transform,
            tokenizer=tokenizer,
            datasets=data_config.get('datasets', ['mathwriting', 'crohme', 'hme100k']),
        )
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        logger.error("Make sure data is downloaded. Run: python scripts/download_data.py")
        sys.exit(1)

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Create dataloaders
    batch_size = train_config.get('batch_size', 128)
    num_workers = 4  # Thunder Compute has good CPU count

    train_loader = train_dataset.get_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampling=True,
    )
    val_loader = val_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        use_weighted_sampling=False,
    )

    # Create trainer
    trainer = ThunderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        tokenizer=tokenizer,
        output_dir=experiment_dir,
        logger=logger,
        lr_override=args.lr,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    best_metric = trainer.train()

    # Save final model
    model.save_pretrained(str(experiment_dir / "final_model"))
    logger.info(f"Saved final model to {experiment_dir / 'final_model'}")
    logger.info(f"Best metric achieved: {best_metric:.4f}")


if __name__ == '__main__':
    main()
