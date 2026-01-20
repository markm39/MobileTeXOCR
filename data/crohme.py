"""
CROHME Dataset Loader

Competition on Recognition of Online Handwritten Mathematical Expressions.
Standard benchmark for handwritten math recognition (10K samples).
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


class CROHMEDataset(BaseDataset):
    """CROHME dataset for handwritten math recognition.

    Competition dataset with online handwritten expressions.
    Commonly used for benchmarking (CROHME 2014, 2016, 2019, etc.).

    Directory structure expected:
        crohme/
            CROHME_2019/
                train/
                    *.inkml
                test/
                    *.inkml
            or
            train/
                *.inkml
            test/
                *.inkml

    Or with pre-rendered images:
        crohme/
            images/
                train/
                    *.png
            labels/
                train.json
    """

    CROHME_VERSIONS = ['2014', '2016', '2019', '2023']

    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        version: str = "2019",
        use_prerendered: bool = True,
    ):
        """
        Args:
            config: Dataset configuration
            split: Data split ("train", "val", "test")
            transform: Image transforms
            tokenizer: LaTeXTokenizer for encoding
            version: CROHME version year
            use_prerendered: Use pre-rendered images if available
        """
        super().__init__(config, split, transform, tokenizer)
        self.version = version
        self.use_prerendered = use_prerendered
        self.stroke_width = 2
        self.margin = 10

        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from disk."""
        data_dir = Path(self.config.data_dir) / "crohme"

        # Try different directory structures
        possible_dirs = [
            data_dir / f"CROHME_{self.version}" / self.split,
            data_dir / f"CROHME{self.version}" / self.split,
            data_dir / self.split,
        ]

        # Check for pre-rendered first
        prerendered_dir = data_dir / "images" / self.split
        labels_file = data_dir / "labels" / f"{self.split}.json"

        if self.use_prerendered and prerendered_dir.exists() and labels_file.exists():
            self._load_prerendered(prerendered_dir, labels_file)
        else:
            for inkml_dir in possible_dirs:
                if inkml_dir.exists():
                    self._load_inkml(inkml_dir)
                    break

        logger.info(f"Loaded {len(self.samples)} CROHME samples for {self.split}")

    def _load_prerendered(self, image_dir: Path, labels_file: Path):
        """Load pre-rendered images."""
        with open(labels_file, 'r') as f:
            labels = json.load(f)

        for image_id, label_info in labels.items():
            image_path = image_dir / f"{image_id}.png"
            if image_path.exists():
                latex = label_info if isinstance(label_info, str) else label_info.get('latex', '')
                self.samples.append({
                    'image_path': str(image_path),
                    'latex': latex,
                    'bbox': None,
                    'metadata': {'source': 'crohme', 'version': self.version, 'id': image_id},
                })

    def _load_inkml(self, inkml_dir: Path):
        """Load InkML files."""
        if not inkml_dir.exists():
            logger.warning(f"CROHME directory not found: {inkml_dir}")
            return

        inkml_files = list(inkml_dir.rglob("*.inkml"))

        for inkml_path in inkml_files:
            try:
                strokes, latex = self._parse_inkml(inkml_path)
                if strokes and latex:
                    self.samples.append({
                        'inkml_path': str(inkml_path),
                        'strokes': strokes,
                        'latex': latex,
                        'bbox': None,
                        'metadata': {
                            'source': 'crohme',
                            'version': self.version,
                            'id': inkml_path.stem
                        },
                    })
            except Exception as e:
                logger.debug(f"Failed to parse {inkml_path}: {e}")

    def _parse_inkml(self, path: Path) -> Tuple[List[np.ndarray], str]:
        """Parse CROHME InkML file."""
        tree = ET.parse(path)
        root = tree.getroot()

        # Namespace handling
        ns = {'ink': 'http://www.w3.org/2003/InkML'}

        strokes = []
        latex = ""

        # Find LaTeX annotation - CROHME uses various annotation types
        for annotation in root.findall('.//ink:annotation', ns):
            ann_type = annotation.get('type', '')
            if ann_type in ('truth', 'normalizedLabel', 'label', 'equationInTeX'):
                latex = annotation.text or ""
                if latex:
                    break

        # Without namespace fallback
        if not latex:
            for annotation in root.findall('.//annotation'):
                ann_type = annotation.get('type', '')
                if ann_type in ('truth', 'normalizedLabel', 'label', 'equationInTeX'):
                    latex = annotation.text or ""
                    if latex:
                        break

        # Parse traces
        for trace in root.findall('.//ink:trace', ns):
            stroke = self._parse_trace(trace.text)
            if stroke is not None:
                strokes.append(stroke)

        if not strokes:
            for trace in root.findall('.//trace'):
                stroke = self._parse_trace(trace.text)
                if stroke is not None:
                    strokes.append(stroke)

        # Clean LaTeX
        latex = self._clean_latex(latex)

        return strokes, latex

    def _parse_trace(self, trace_text: str) -> Optional[np.ndarray]:
        """Parse trace text to points array."""
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

    def _clean_latex(self, latex: str) -> str:
        """Clean and normalize LaTeX string."""
        if not latex:
            return ""

        # Remove leading/trailing whitespace
        latex = latex.strip()

        # CROHME sometimes uses $ delimiters
        latex = latex.strip('$')

        # Normalize spaces
        latex = ' '.join(latex.split())

        return latex

    def _render_strokes(
        self,
        strokes: List[np.ndarray],
        target_size: int = 384
    ) -> torch.Tensor:
        """Render strokes to image tensor."""
        if not strokes:
            return torch.ones(3, target_size, target_size)

        all_points = np.concatenate(strokes, axis=0)
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)

        width = max_x - min_x + 2 * self.margin
        height = max_y - min_y + 2 * self.margin

        scale = min((target_size - 2 * self.margin) / max(width, 1),
                    (target_size - 2 * self.margin) / max(height, 1))

        img = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        for stroke in strokes:
            points = (stroke - [min_x, min_y]) * scale + self.margin
            offset_x = (target_size - (max_x - min_x) * scale) / 2
            offset_y = (target_size - (max_y - min_y) * scale) / 2
            points += [offset_x - self.margin, offset_y - self.margin]

            if len(points) >= 2:
                point_tuples = [tuple(p) for p in points]
                draw.line(point_tuples, fill=(0, 0, 0), width=self.stroke_width)

        img_array = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        return tensor

    def _load_sample(self, idx: int) -> Tuple[torch.Tensor, str, Optional[Tuple], Dict]:
        """Load a single sample."""
        sample = self.samples[idx]

        if 'image_path' in sample:
            image = self._load_image(sample['image_path'])
        else:
            image = self._render_strokes(sample['strokes'], self.config.image_size)

        return (
            image,
            sample['latex'],
            sample.get('bbox'),
            sample.get('metadata', {}),
        )


def download_crohme(output_dir: str, version: str = "2019"):
    """Download CROHME dataset instructions."""
    print(f"""
    CROHME {version} Dataset Download Instructions:

    1. Official CROHME website: https://www.isical.ac.in/~crohme/

    2. For CROHME 2019:
       - Register and download from the competition page
       - Or use the TC-11 repository: http://tc11.cvc.uab.es/

    3. Extract to: {output_dir}/crohme/CROHME_{version}/

    4. Directory structure should be:
       {output_dir}/crohme/CROHME_{version}/
           train/
               *.inkml
           test/
               *.inkml

    5. Alternative: Use pre-processed versions from Kaggle:
       https://www.kaggle.com/datasets/rtatman/handwritten-mathematical-expressions
    """)
