"""
HME100K Dataset Loader

Handwritten Mathematical Expression 100K dataset.
100K offline handwritten math images with LaTeX labels.
"""

from typing import Optional, Tuple, Dict, Any, Callable, List
from pathlib import Path
import json
import logging
import csv

import torch
import numpy as np

from data.base import BaseDataset, DatasetConfig

logger = logging.getLogger(__name__)


class HME100KDataset(BaseDataset):
    """HME100K dataset for handwritten math recognition.

    Large-scale offline handwritten math dataset (100K samples).
    Contains actual photographed handwritten expressions.

    Directory structure expected:
        hme100k/
            train/
                images/
                    *.jpg
                labels.txt  (or labels.json)
            test/
                images/
                    *.jpg
                labels.txt

    Labels format (txt):
        image_name.jpg\tLaTeX_expression
        or
        image_name.jpg,LaTeX_expression
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
            split: Data split
            transform: Image transforms
            tokenizer: LaTeXTokenizer for encoding
        """
        super().__init__(config, split, transform, tokenizer)
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from disk."""
        data_dir = Path(self.config.data_dir) / "hme100k"

        # Try different directory structures
        possible_structures = [
            (data_dir / self.split / "images", data_dir / self.split / "labels.txt"),
            (data_dir / self.split / "images", data_dir / self.split / "labels.json"),
            (data_dir / "images" / self.split, data_dir / "labels" / f"{self.split}.txt"),
            (data_dir / "images" / self.split, data_dir / "labels" / f"{self.split}.json"),
            (data_dir / self.split, data_dir / f"{self.split}_labels.txt"),
        ]

        for image_dir, label_file in possible_structures:
            if image_dir.exists() and label_file.exists():
                self._load_from_files(image_dir, label_file)
                break

        logger.info(f"Loaded {len(self.samples)} HME100K samples for {self.split}")

    def _load_from_files(self, image_dir: Path, label_file: Path):
        """Load images and labels from files."""
        if label_file.suffix == '.json':
            self._load_json_labels(image_dir, label_file)
        else:
            self._load_txt_labels(image_dir, label_file)

    def _load_json_labels(self, image_dir: Path, label_file: Path):
        """Load labels from JSON file."""
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)

        for image_name, latex in labels.items():
            # Handle both dict format and string format
            if isinstance(latex, dict):
                latex = latex.get('latex', latex.get('label', ''))

            image_path = image_dir / image_name
            if not image_path.exists():
                # Try common extensions
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    candidate = image_dir / f"{Path(image_name).stem}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break

            if image_path.exists():
                self.samples.append({
                    'image_path': str(image_path),
                    'latex': self._clean_latex(latex),
                    'bbox': None,
                    'metadata': {'source': 'hme100k', 'id': Path(image_name).stem},
                })

    def _load_txt_labels(self, image_dir: Path, label_file: Path):
        """Load labels from text file (tab or comma separated)."""
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Try tab separator first, then comma
                if '\t' in line:
                    parts = line.split('\t', 1)
                else:
                    parts = line.split(',', 1)

                if len(parts) != 2:
                    continue

                image_name, latex = parts
                image_name = image_name.strip()
                latex = latex.strip()

                image_path = image_dir / image_name
                if not image_path.exists():
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        candidate = image_dir / f"{Path(image_name).stem}{ext}"
                        if candidate.exists():
                            image_path = candidate
                            break

                if image_path.exists():
                    self.samples.append({
                        'image_path': str(image_path),
                        'latex': self._clean_latex(latex),
                        'bbox': None,
                        'metadata': {'source': 'hme100k', 'id': Path(image_name).stem},
                    })

    def _clean_latex(self, latex: str) -> str:
        """Clean and normalize LaTeX string."""
        if not latex:
            return ""

        latex = latex.strip()
        latex = latex.strip('$')
        latex = ' '.join(latex.split())

        return latex

    def _load_sample(self, idx: int) -> Tuple[torch.Tensor, str, Optional[Tuple], Dict]:
        """Load a single sample."""
        sample = self.samples[idx]
        image = self._load_image(sample['image_path'])

        return (
            image,
            sample['latex'],
            sample.get('bbox'),
            sample.get('metadata', {}),
        )


def download_hme100k(output_dir: str):
    """Download HME100K dataset instructions."""
    print(f"""
    HME100K Dataset Download Instructions:

    1. The dataset is available from:
       - Original paper: https://arxiv.org/abs/1910.05171
       - GitHub releases from the paper authors

    2. Alternative sources:
       - Hugging Face: search for "hme100k" or "handwritten math"
       - Kaggle: https://www.kaggle.com/datasets

    3. After downloading, organize as:
       {output_dir}/hme100k/
           train/
               images/
                   *.jpg
               labels.txt  (format: image.jpg\\tLaTeX)
           test/
               images/
                   *.jpg
               labels.txt

    4. Label format (tab-separated):
       image001.jpg\t\\frac{{a}}{{b}}
       image002.jpg\tx^2 + y^2 = z^2
    """)
