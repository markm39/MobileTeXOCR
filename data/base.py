"""
Base dataset class and utilities for handwritten LaTeX OCR.
"""

from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for datasets."""
    # Paths
    data_dir: str = "./data"

    # Image settings
    image_size: int = 384
    max_seq_length: int = 512

    # Tokenizer settings
    num_location_bins: int = 1000

    # Augmentation
    augment: bool = True

    # Caching
    cache_images: bool = False

    # Workers
    num_workers: int = 4


@dataclass
class Sample:
    """A single training sample."""
    image: torch.Tensor  # [3, H, W]
    latex: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # Normalized (x1, y1, x2, y2)
    token_ids: Optional[torch.Tensor] = None  # Tokenized sequence
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDataset(Dataset):
    """Base class for handwritten math datasets.

    All datasets should inherit from this and implement:
    - __len__: Return dataset size
    - _load_sample: Load a single sample (image, latex, optional bbox)
    """

    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Args:
            config: Dataset configuration
            split: Data split ("train", "val", "test")
            transform: Image transforms to apply
            tokenizer: LaTeXTokenizer instance for encoding labels
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer

        # Will be populated by subclasses
        self.samples: List[Dict[str, Any]] = []

        # Image cache (optional)
        self._cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        """Get a single sample.

        Returns:
            Sample with image, latex, and optional bbox
        """
        # Check cache first
        if self.config.cache_images and idx in self._cache:
            image = self._cache[idx]
        else:
            image, latex, bbox, metadata = self._load_sample(idx)

            # Cache if enabled
            if self.config.cache_images:
                self._cache[idx] = image.clone()

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        # Tokenize if tokenizer is provided
        token_ids = None
        if self.tokenizer is not None:
            sample_info = self.samples[idx]
            latex = sample_info.get('latex', '')
            bbox = sample_info.get('bbox')
            token_ids = torch.tensor(
                self.tokenizer.encode_sample(latex, bbox, add_special_tokens=True),
                dtype=torch.long
            )

        return Sample(
            image=image,
            latex=self.samples[idx].get('latex', ''),
            bbox=self.samples[idx].get('bbox'),
            token_ids=token_ids,
            metadata=self.samples[idx].get('metadata', {}),
        )

    def _load_sample(self, idx: int) -> Tuple[torch.Tensor, str, Optional[Tuple], Dict]:
        """Load a single sample from disk.

        Must be implemented by subclasses.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, latex, bbox, metadata)
        """
        raise NotImplementedError("Subclasses must implement _load_sample")

    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess an image.

        Args:
            path: Path to image file

        Returns:
            Image tensor [3, H, W] normalized to [0, 1]
        """
        from PIL import Image
        import torchvision.transforms.functional as TF

        img = Image.open(path).convert('RGB')

        # Resize to target size while maintaining aspect ratio
        w, h = img.size
        target_size = self.config.image_size

        # Compute scaling factor
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Pad to square
        padded = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        padded.paste(img, (paste_x, paste_y))

        # Convert to tensor
        tensor = TF.to_tensor(padded)  # [3, H, W] in [0, 1]

        return tensor

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> DataLoader:
        """Create a DataLoader for this dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of data loading workers

        Returns:
            DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )


def collate_fn(samples: List[Sample]) -> Dict[str, torch.Tensor]:
    """Collate samples into a batch.

    Args:
        samples: List of Sample objects

    Returns:
        Batched tensors
    """
    # Stack images
    images = torch.stack([s.image for s in samples])

    batch = {
        'images': images,
        'latex': [s.latex for s in samples],
    }

    # Pad token sequences if present
    if samples[0].token_ids is not None:
        max_len = max(len(s.token_ids) for s in samples)
        padded_ids = []
        padding_masks = []

        for s in samples:
            ids = s.token_ids
            pad_len = max_len - len(ids)

            if pad_len > 0:
                # Assuming pad_token_id is the first token (index 0 after location bins)
                # This will be set correctly by the tokenizer
                pad_id = 1000  # Default, will be overridden
                ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])

            padded_ids.append(ids)
            mask = torch.zeros(max_len, dtype=torch.bool)
            mask[len(s.token_ids):] = True
            padding_masks.append(mask)

        batch['token_ids'] = torch.stack(padded_ids)
        batch['padding_mask'] = torch.stack(padding_masks)

    # Include bboxes if present
    if samples[0].bbox is not None:
        batch['bboxes'] = [s.bbox for s in samples]

    return batch


def create_dataloader(
    dataset: BaseDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with proper collation.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        pin_memory: Whether to pin memory

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=shuffle,  # Drop incomplete batches during training
    )
