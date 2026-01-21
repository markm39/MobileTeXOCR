"""
Training Loop for Handwritten LaTeX OCR

Supports:
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Learning rate scheduling
- Checkpoint saving/resuming
- Wandb logging
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from training.metrics import compute_metrics, ExpRate, SymbolAccuracy

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Output
    output_dir: str = "./outputs"
    experiment_name: str = "latex_ocr"

    # Training
    num_epochs: int = 20
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"  # "cosine", "linear", "constant"

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"

    # Checkpointing
    save_steps: int = 1000
    validation_steps: int = 500
    save_total_limit: int = 3

    # Logging
    log_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "latex-ocr"

    # Encoder training
    freeze_encoder_epochs: int = 1  # Freeze encoder for first N epochs

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_metric: str = "exp_rate"


class Trainer:
    """Training loop for the OCR model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            tokenizer: Tokenizer for decoding predictions
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.tokenizer = tokenizer or model.tokenizer

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Mixed precision
        if self.config.use_amp:
            dtype = torch.float16 if self.config.amp_dtype == "float16" else torch.bfloat16
            self.scaler = GradScaler('cuda') if dtype == torch.float16 else None
            self.amp_dtype = dtype
        else:
            self.scaler = None
            self.amp_dtype = torch.float32

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0

        # Output directory
        self.output_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Wandb
        self.wandb_run = None
        if self.config.use_wandb:
            self._setup_wandb()

        logger.info(f"Training on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters that should/shouldn't have weight decay
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
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = self.config.warmup_steps

        if self.config.lr_scheduler == "cosine":
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
                eta_min=self.config.learning_rate * 0.01,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        elif self.config.lr_scheduler == "linear":
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps,
            )
        else:
            return None

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=self.config.__dict__,
            )
        except ImportError:
            logger.warning("wandb not installed, disabling wandb logging")
            self.config.use_wandb = False

    def train(self):
        """Run the full training loop."""
        start_epoch = self.epoch  # Resume from saved epoch
        print(f"Starting training: epochs {start_epoch}-{self.config.num_epochs-1}, {len(self.train_loader)} batches/epoch, step {self.global_step}", flush=True)

        for epoch in range(start_epoch, self.config.num_epochs):
            self.epoch = epoch

            # Handle encoder freezing
            if epoch < self.config.freeze_encoder_epochs:
                self.model.freeze_encoder()
                print(f"Epoch {epoch}: Encoder frozen", flush=True)
            elif epoch == self.config.freeze_encoder_epochs:
                self.model.unfreeze_encoder()
                print(f"Epoch {epoch}: Encoder unfrozen", flush=True)
            else:
                # Ensure encoder is unfrozen for epochs after freeze period
                self.model.unfreeze_encoder()

            # Train epoch
            train_metrics = self._train_epoch()
            print(f"Epoch {epoch} train: {train_metrics}", flush=True)

            # Run validation
            if self.val_loader is not None:
                val_metrics = self.run_validation()
                print(f"Epoch {epoch} val: {val_metrics}", flush=True)

                # Check for improvement
                current_metric = val_metrics.get(self.config.early_stopping_metric, 0)
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    self._save_checkpoint("best")
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping after {epoch + 1} epochs", flush=True)
                    break

            # Save checkpoint
            self._save_checkpoint(f"epoch_{epoch}")

        print(f"Training complete! Best metric: {self.best_metric:.4f}", flush=True)
        return self.best_metric

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1

            # Logging
            if self.global_step % self.config.log_steps == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Step {self.global_step}: loss={loss:.4f}, lr={lr:.2e}", flush=True)
                logger.info(
                    f"Step {self.global_step}: loss={loss:.4f}, lr={lr:.2e}"
                )
                if self.wandb_run:
                    import wandb
                    wandb.log({
                        'train/loss': loss,
                        'train/lr': lr,
                        'train/epoch': self.epoch,
                    }, step=self.global_step)

            # Periodic validation
            if self.val_loader and self.global_step % self.config.validation_steps == 0:
                val_metrics = self.run_validation()
                if self.wandb_run:
                    import wandb
                    wandb.log({f'val/{k}': v for k, v in val_metrics.items()}, step=self.global_step)
                self.model.train()

            # Checkpointing
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint(f"step_{self.global_step}")

        return {'loss': total_loss / num_batches}

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        # Move to device
        images = batch['images'].to(self.device)
        token_ids = batch['token_ids'].to(self.device)
        padding_mask = batch.get('padding_mask')
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)

        # Forward pass with mixed precision
        with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.config.use_amp):
            output = self.model(images, token_ids, target_padding_mask=padding_mask)
            loss = output.loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            if self.scheduler:
                self.scheduler.step()

        self.global_step += 1

        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def run_validation(self) -> Dict[str, float]:
        """Run validation on the validation set."""
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
            with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.config.use_amp):
                output = self.model(images, token_ids)
                total_loss += output.loss.item()

            # Generate predictions
            pred_output = self.model(images)
            for pred_regions in pred_output.predictions:
                # Take first region's LaTeX (or combine if multi-region)
                if pred_regions:
                    pred_latex = ' '.join(region[1] for region in pred_regions)
                else:
                    pred_latex = ""
                all_predictions.append(pred_latex)

            all_targets.extend(targets)
            num_batches += 1

        # Compute metrics
        metrics = compute_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / num_batches

        return metrics

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{name}.pt"

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'config': self.config.__dict__,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Clean up old checkpoints
        self._cleanup_checkpoints(checkpoint_dir)

    def _cleanup_checkpoints(self, checkpoint_dir: Path):
        """Keep only the most recent checkpoints."""
        checkpoints = sorted(
            checkpoint_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split('_')[1]),
        )

        if len(checkpoints) > self.config.save_total_limit:
            for checkpoint in checkpoints[:-self.config.save_total_limit]:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")

    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', 0.0)

        print(f"Loaded checkpoint: epoch {self.epoch}, step {self.global_step}, best_metric {self.best_metric:.4f}", flush=True)
