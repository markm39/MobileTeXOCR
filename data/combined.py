"""
Combined Dataset

Combines multiple datasets (MathWriting, CROHME, HME100K) for training.
Supports weighted sampling and curriculum learning.
"""

from typing import Optional, List, Dict, Any, Callable, Tuple
import random
import logging

import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler

from data.base import BaseDataset, DatasetConfig, Sample, collate_fn
from data.mathwriting import MathWritingDataset
from data.crohme import CROHMEDataset
from data.hme100k import HME100KDataset

logger = logging.getLogger(__name__)


class CombinedDataset(Dataset):
    """Combined dataset from multiple sources.

    Provides unified access to MathWriting, CROHME, and HME100K datasets
    with support for weighted sampling.
    """

    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        datasets: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            config: Dataset configuration
            split: Data split
            transform: Image transforms
            tokenizer: Tokenizer for encoding
            datasets: List of dataset names to include
                     ["mathwriting", "crohme", "hme100k"]
            weights: Sampling weights per dataset (default: proportional to size)
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer

        # Default to all datasets
        if datasets is None:
            datasets = ["mathwriting", "crohme", "hme100k"]
        self.dataset_names = datasets

        # Default weights (proportional sampling)
        if weights is None:
            weights = {name: 1.0 for name in datasets}
        self.weights = weights

        # Load individual datasets
        self.datasets: Dict[str, BaseDataset] = {}
        self._load_datasets()

        # Create combined sample list with source tracking
        self.samples: List[Tuple[str, int]] = []  # (dataset_name, local_idx)
        self._build_sample_list()

        logger.info(f"Combined dataset: {len(self)} total samples")
        for name, ds in self.datasets.items():
            logger.info(f"  {name}: {len(ds)} samples")

    def _load_datasets(self):
        """Load individual datasets."""
        for name in self.dataset_names:
            try:
                if name == "mathwriting":
                    ds = MathWritingDataset(
                        self.config, self.split, self.transform, self.tokenizer
                    )
                elif name == "crohme":
                    ds = CROHMEDataset(
                        self.config, self.split, self.transform, self.tokenizer
                    )
                elif name == "hme100k":
                    ds = HME100KDataset(
                        self.config, self.split, self.transform, self.tokenizer
                    )
                else:
                    logger.warning(f"Unknown dataset: {name}")
                    continue

                if len(ds) > 0:
                    self.datasets[name] = ds
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")

    def _build_sample_list(self):
        """Build combined sample list."""
        for name, ds in self.datasets.items():
            for i in range(len(ds)):
                self.samples.append((name, i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        """Get a sample by index."""
        dataset_name, local_idx = self.samples[idx]
        return self.datasets[dataset_name][local_idx]

    def get_sampler(self, num_samples: Optional[int] = None) -> WeightedRandomSampler:
        """Get weighted sampler for balanced training.

        Args:
            num_samples: Number of samples per epoch (default: dataset size)

        Returns:
            WeightedRandomSampler
        """
        # Compute sample weights
        sample_weights = []

        for dataset_name, _ in self.samples:
            dataset_size = len(self.datasets[dataset_name])
            # Weight = dataset_weight / dataset_size (normalizes by size)
            weight = self.weights.get(dataset_name, 1.0) / dataset_size
            sample_weights.append(weight)

        if num_samples is None:
            num_samples = len(self)

        return WeightedRandomSampler(
            sample_weights,
            num_samples=num_samples,
            replacement=True,
        )

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        use_weighted_sampling: bool = True,
    ):
        """Create DataLoader with optional weighted sampling.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle (ignored if weighted sampling)
            num_workers: Number of workers
            use_weighted_sampling: Use weighted sampler

        Returns:
            DataLoader
        """
        from torch.utils.data import DataLoader

        if use_weighted_sampling and self.split == "train":
            sampler = self.get_sampler()
            return DataLoader(
                self,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
        else:
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )


def create_dataloaders(
    config: DatasetConfig,
    tokenizer: Any,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[Any, Any, Any]:
    """Create train/val/test dataloaders.

    Args:
        config: Dataset configuration
        tokenizer: LaTeX tokenizer
        train_transform: Transforms for training
        eval_transform: Transforms for evaluation
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Training dataset
    train_dataset = CombinedDataset(
        config,
        split="train",
        transform=train_transform,
        tokenizer=tokenizer,
    )
    train_loader = train_dataset.get_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampling=True,
    )

    # Validation dataset
    val_dataset = CombinedDataset(
        config,
        split="val",
        transform=eval_transform,
        tokenizer=tokenizer,
    )
    val_loader = val_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        use_weighted_sampling=False,
    )

    # Test dataset
    test_dataset = CombinedDataset(
        config,
        split="test",
        transform=eval_transform,
        tokenizer=tokenizer,
    )
    test_loader = test_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        use_weighted_sampling=False,
    )

    return train_loader, val_loader, test_loader
