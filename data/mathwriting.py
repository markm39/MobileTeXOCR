"""
MathWriting Dataset Loader

Google's large-scale online handwritten math dataset (630K samples).
Paper: https://arxiv.org/abs/2404.10690
Format: InkML files with stroke data that must be rendered to images.
"""

from typing import Optional, Tuple, Dict, Any, Callable, List
from pathlib import Path
import json
import logging
from xml.etree import ElementTree as ET

import torch
import numpy as np
from PIL import Image, ImageDraw

from data.base import BaseDataset, DatasetConfig

logger = logging.getLogger(__name__)


class MathWritingDataset(BaseDataset):
    """MathWriting dataset from Google.

    The dataset contains online handwritten math expressions with stroke data.
    We render strokes to images for training the vision model.

    Directory structure expected:
        mathwriting/
            train/
                *.inkml  (or organized in subdirs)
            val/
            test/

    Or with pre-rendered images:
        mathwriting/
            images/
                train/
                    *.png
            labels/
                train.json  (mapping image_id -> latex)
    """

    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        use_prerendered: bool = True,
    ):
        """
        Args:
            config: Dataset configuration
            split: Data split
            transform: Image transforms
            tokenizer: LaTeXTokenizer for encoding
            use_prerendered: Use pre-rendered images if available
        """
        super().__init__(config, split, transform, tokenizer)
        self.use_prerendered = use_prerendered

        # Rendering settings for InkML -> Image
        self.stroke_width = 2
        self.margin = 10

        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from disk."""
        data_dir = Path(self.config.data_dir) / "mathwriting"

        # Check for pre-rendered images first
        prerendered_dir = data_dir / "images" / self.split
        labels_file = data_dir / "labels" / f"{self.split}.json"

        if self.use_prerendered and prerendered_dir.exists() and labels_file.exists():
            self._load_prerendered(prerendered_dir, labels_file)
        else:
            # Try the requested split directory
            split_dir = data_dir / self.split

            # MathWriting uses 'valid' instead of 'val' - map accordingly
            if not split_dir.exists() and self.split == "val":
                # Try 'valid' directory first (MathWriting naming)
                split_dir = data_dir / "valid"
                if split_dir.exists():
                    logger.info("Using 'valid' split as validation set")
                else:
                    # Try 'test' as fallback
                    split_dir = data_dir / "test"
                    if split_dir.exists():
                        logger.info("Using 'test' split as validation set")

            if split_dir.exists():
                self._load_inkml(split_dir)
            else:
                # No dedicated split dir - create splits from train
                train_dir = data_dir / "train"
                if train_dir.exists():
                    # Check if we need to create a val split from train
                    val_dir = data_dir / "val"
                    valid_dir = data_dir / "valid"
                    test_dir = data_dir / "test"
                    needs_split = not val_dir.exists() and not valid_dir.exists() and not test_dir.exists()

                    if needs_split:
                        if self.split == "train":
                            logger.info("No val/test split found, using 90% of data for training")
                            self._load_inkml_with_split(train_dir, val_ratio=0.1, use_val=False)
                        else:
                            logger.info(f"No {self.split} split found, using 10% of train data")
                            self._load_inkml_with_split(train_dir, val_ratio=0.1, use_val=True)
                    else:
                        logger.warning(f"Split directory not found: {split_dir}")
                else:
                    logger.warning(f"No data found in {data_dir}")

        logger.info(f"Loaded {len(self.samples)} MathWriting samples for {self.split}")

    def _load_prerendered(self, image_dir: Path, labels_file: Path):
        """Load pre-rendered images and labels."""
        with open(labels_file, 'r') as f:
            labels = json.load(f)

        for image_id, latex in labels.items():
            image_path = image_dir / f"{image_id}.png"
            if image_path.exists():
                self.samples.append({
                    'image_path': str(image_path),
                    'latex': latex,
                    'bbox': None,  # Full image
                    'metadata': {'source': 'mathwriting', 'id': image_id},
                })

    def _load_inkml_with_split(self, inkml_dir: Path, val_ratio: float = 0.1, use_val: bool = True):
        """Load InkML files and split into train/val.

        Args:
            inkml_dir: Directory with InkML files
            val_ratio: Fraction of data to use for validation
            use_val: If True, return val portion; if False, return train portion
        """
        if not inkml_dir.exists():
            logger.warning(f"InkML directory not found: {inkml_dir}")
            return

        # Find all InkML files
        inkml_files = sorted(list(inkml_dir.rglob("*.inkml")) + list(inkml_dir.rglob("*.xml")))

        # Deterministic split based on hash
        val_files = []
        train_files = []
        for f in inkml_files:
            # Use hash for deterministic split
            if hash(f.stem) % 100 < val_ratio * 100:
                val_files.append(f)
            else:
                train_files.append(f)

        files_to_load = val_files if use_val else train_files
        logger.info(f"Split: {len(train_files)} train, {len(val_files)} val files")

        for inkml_path in files_to_load:
            try:
                strokes, latex = self._parse_inkml(inkml_path)
                if strokes and latex:
                    self.samples.append({
                        'inkml_path': str(inkml_path),
                        'strokes': strokes,
                        'latex': latex,
                        'bbox': None,
                        'metadata': {'source': 'mathwriting', 'id': inkml_path.stem},
                    })
            except Exception as e:
                logger.debug(f"Failed to parse {inkml_path}: {e}")

    def _load_inkml(self, inkml_dir: Path):
        """Load InkML files and prepare for rendering."""
        if not inkml_dir.exists():
            logger.warning(f"InkML directory not found: {inkml_dir}")
            return

        # Find all InkML files
        inkml_files = list(inkml_dir.rglob("*.inkml")) + list(inkml_dir.rglob("*.xml"))

        for inkml_path in inkml_files:
            try:
                strokes, latex = self._parse_inkml(inkml_path)
                if strokes and latex:
                    self.samples.append({
                        'inkml_path': str(inkml_path),
                        'strokes': strokes,
                        'latex': latex,
                        'bbox': None,
                        'metadata': {'source': 'mathwriting', 'id': inkml_path.stem},
                    })
            except Exception as e:
                logger.debug(f"Failed to parse {inkml_path}: {e}")

    def _parse_inkml(self, path: Path) -> Tuple[List[np.ndarray], str]:
        """Parse InkML file to extract strokes and LaTeX.

        Args:
            path: Path to InkML file

        Returns:
            Tuple of (list of stroke arrays, latex string)
        """
        tree = ET.parse(path)
        root = tree.getroot()

        # Handle namespace
        ns = {'ink': 'http://www.w3.org/2003/InkML'}

        strokes = []
        latex = ""

        # Extract LaTeX annotation
        for annotation in root.findall('.//ink:annotation', ns):
            if annotation.get('type') == 'truth' or annotation.get('type') == 'label':
                latex = annotation.text or ""
                break

        # Also check without namespace
        if not latex:
            for annotation in root.findall('.//annotation'):
                if annotation.get('type') in ('truth', 'label', 'normalizedLabel'):
                    latex = annotation.text or ""
                    break

        # Extract strokes
        for trace in root.findall('.//ink:trace', ns):
            stroke = self._parse_trace(trace.text)
            if stroke is not None:
                strokes.append(stroke)

        # Also check without namespace
        if not strokes:
            for trace in root.findall('.//trace'):
                stroke = self._parse_trace(trace.text)
                if stroke is not None:
                    strokes.append(stroke)

        return strokes, latex

    def _parse_trace(self, trace_text: str) -> Optional[np.ndarray]:
        """Parse a trace element into array of points.

        Args:
            trace_text: Trace data as string "x1 y1, x2 y2, ..."

        Returns:
            numpy array of shape [N, 2] or None
        """
        if not trace_text:
            return None

        points = []
        for point_str in trace_text.strip().split(','):
            coords = point_str.strip().split()
            if len(coords) >= 2:
                try:
                    x, y = float(coords[0]), float(coords[1])
                    points.append([x, y])
                except ValueError:
                    continue

        if len(points) < 2:
            return None

        return np.array(points)

    def _render_strokes(
        self,
        strokes: List[np.ndarray],
        target_size: int = 384
    ) -> torch.Tensor:
        """Render stroke data to image tensor.

        Args:
            strokes: List of stroke arrays, each [N, 2]
            target_size: Output image size

        Returns:
            Image tensor [3, H, W]
        """
        if not strokes:
            # Return white image
            return torch.ones(3, target_size, target_size)

        # Concatenate all points to find bounds
        all_points = np.concatenate(strokes, axis=0)
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)

        # Add margins
        width = max_x - min_x + 2 * self.margin
        height = max_y - min_y + 2 * self.margin

        # Compute scale to fit target size
        scale = min((target_size - 2 * self.margin) / max(width, 1),
                    (target_size - 2 * self.margin) / max(height, 1))

        # Create white image
        img = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw strokes
        for stroke in strokes:
            # Transform points
            points = (stroke - [min_x, min_y]) * scale + self.margin
            # Center in image
            offset_x = (target_size - (max_x - min_x) * scale) / 2
            offset_y = (target_size - (max_y - min_y) * scale) / 2
            points += [offset_x - self.margin, offset_y - self.margin]

            # Draw polyline
            if len(points) >= 2:
                point_tuples = [tuple(p) for p in points]
                draw.line(point_tuples, fill=(0, 0, 0), width=self.stroke_width)

        # Convert to tensor
        img_array = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        return tensor

    def _load_sample(self, idx: int) -> Tuple[torch.Tensor, str, Optional[Tuple], Dict]:
        """Load a single sample.

        Args:
            idx: Sample index

        Returns:
            (image, latex, bbox, metadata)
        """
        sample = self.samples[idx]

        if 'image_path' in sample:
            # Pre-rendered image
            image = self._load_image(sample['image_path'])
        else:
            # Render from strokes
            image = self._render_strokes(
                sample['strokes'],
                self.config.image_size
            )

        return (
            image,
            sample['latex'],
            sample.get('bbox'),
            sample.get('metadata', {}),
        )


def download_mathwriting(output_dir: str):
    """Download MathWriting dataset.

    Note: The dataset must be downloaded from the official source.
    This function provides instructions.
    """
    print("""
    MathWriting Dataset Download Instructions:

    1. Visit: https://github.com/google-research/google-research/tree/master/mathwriting

    2. The dataset is available on Google Cloud Storage:
       gs://mathwriting_data/

    3. Download using gsutil:
       gsutil -m cp -r gs://mathwriting_data/train {output_dir}/mathwriting/
       gsutil -m cp -r gs://mathwriting_data/val {output_dir}/mathwriting/
       gsutil -m cp -r gs://mathwriting_data/test {output_dir}/mathwriting/

    4. Alternatively, check the paper for download links:
       https://arxiv.org/abs/2404.10690
    """.format(output_dir=output_dir))
