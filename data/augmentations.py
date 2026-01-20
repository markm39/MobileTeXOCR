"""
Data Augmentations for Handwritten Math OCR

Provides augmentation transforms optimized for handwritten mathematical expressions.
Key considerations:
- Preserve mathematical structure
- Simulate real-world variations (camera, lighting, paper)
- Handle variable-length expressions
"""

from typing import Optional, Tuple, List, Dict, Any
import random
import math

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class RandomAffine:
    """Random affine transformation with controlled parameters."""

    def __init__(
        self,
        degrees: float = 5.0,
        translate: Tuple[float, float] = (0.05, 0.05),
        scale: Tuple[float, float] = (0.95, 1.05),
        shear: float = 3.0,
        fill: int = 255,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.fill = fill

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Convert to PIL for transformation
        from PIL import Image
        import numpy as np

        # [C, H, W] -> PIL
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # Apply affine
        pil_img = TF.affine(
            pil_img,
            angle=random.uniform(-self.degrees, self.degrees),
            translate=(
                random.uniform(-self.translate[0], self.translate[0]) * pil_img.width,
                random.uniform(-self.translate[1], self.translate[1]) * pil_img.height,
            ),
            scale=random.uniform(self.scale[0], self.scale[1]),
            shear=random.uniform(-self.shear, self.shear),
            fill=self.fill,
        )

        # PIL -> tensor
        return TF.to_tensor(pil_img)


class RandomPerspective:
    """Mild perspective distortion to simulate camera angles."""

    def __init__(self, distortion_scale: float = 0.1, p: float = 0.3, fill: int = 255):
        self.distortion_scale = distortion_scale
        self.p = p
        self.fill = fill

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        from PIL import Image
        import numpy as np

        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        pil_img = TF.perspective(
            pil_img,
            startpoints=self._get_corners(pil_img.width, pil_img.height),
            endpoints=self._get_distorted_corners(pil_img.width, pil_img.height),
            fill=self.fill,
        )

        return TF.to_tensor(pil_img)

    def _get_corners(self, w: int, h: int) -> List[Tuple[int, int]]:
        return [(0, 0), (w, 0), (w, h), (0, h)]

    def _get_distorted_corners(self, w: int, h: int) -> List[Tuple[int, int]]:
        d = self.distortion_scale
        return [
            (int(random.uniform(0, d * w)), int(random.uniform(0, d * h))),
            (int(random.uniform(w * (1 - d), w)), int(random.uniform(0, d * h))),
            (int(random.uniform(w * (1 - d), w)), int(random.uniform(h * (1 - d), h))),
            (int(random.uniform(0, d * w)), int(random.uniform(h * (1 - d), h))),
        ]


class RandomNoise:
    """Add random noise to simulate scan/camera artifacts."""

    def __init__(self, std: float = 0.02, p: float = 0.3):
        self.std = std
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        noise = torch.randn_like(img) * self.std
        return torch.clamp(img + noise, 0, 1)


class RandomBrightness:
    """Random brightness adjustment."""

    def __init__(self, factor_range: Tuple[float, float] = (0.9, 1.1), p: float = 0.3):
        self.factor_range = factor_range
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        factor = random.uniform(*self.factor_range)
        return torch.clamp(img * factor, 0, 1)


class RandomContrast:
    """Random contrast adjustment."""

    def __init__(self, factor_range: Tuple[float, float] = (0.9, 1.1), p: float = 0.3):
        self.factor_range = factor_range
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        factor = random.uniform(*self.factor_range)
        mean = img.mean()
        return torch.clamp((img - mean) * factor + mean, 0, 1)


class RandomGaussianBlur:
    """Random Gaussian blur to simulate focus variations."""

    def __init__(self, kernel_size: int = 3, sigma_range: Tuple[float, float] = (0.1, 1.0), p: float = 0.2):
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        sigma = random.uniform(*self.sigma_range)
        return TF.gaussian_blur(img, self.kernel_size, sigma)


class RandomErosionDilation:
    """Simulate stroke width variations through morphological operations."""

    def __init__(self, kernel_size: int = 3, p: float = 0.2):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        # Add batch dimension for unfold
        img = img.unsqueeze(0)

        if random.random() < 0.5:
            # Erosion (makes strokes thinner)
            img = -F.max_pool2d(-img, self.kernel_size, stride=1, padding=self.kernel_size // 2)
        else:
            # Dilation (makes strokes thicker)
            img = F.max_pool2d(img, self.kernel_size, stride=1, padding=self.kernel_size // 2)

        return img.squeeze(0)


class RandomInvert:
    """Random inversion (white on black vs black on white)."""

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        return 1.0 - img


class ElasticDistortion:
    """Elastic distortion to simulate natural handwriting variations."""

    def __init__(self, alpha: float = 50.0, sigma: float = 5.0, p: float = 0.2):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        C, H, W = img.shape

        # Generate random displacement fields
        dx = torch.randn(1, H, W) * self.alpha
        dy = torch.randn(1, H, W) * self.alpha

        # Smooth with Gaussian
        dx = TF.gaussian_blur(dx, int(self.sigma * 6) | 1, self.sigma)
        dy = TF.gaussian_blur(dy, int(self.sigma * 6) | 1, self.sigma)

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )

        # Apply displacement
        grid_x = grid_x + dx.squeeze() * 2 / W
        grid_y = grid_y + dy.squeeze() * 2 / H

        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        # Resample image
        img = img.unsqueeze(0)
        warped = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return warped.squeeze(0)


class Normalize:
    """Normalize image to standard range."""

    def __init__(self, mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            img = t(img)
        return img


def get_train_transforms(
    image_size: int = 384,
    augment_strength: str = "medium",
) -> Compose:
    """Get training transforms.

    Args:
        image_size: Target image size
        augment_strength: "light", "medium", or "heavy"

    Returns:
        Composed transforms
    """
    transforms = []

    if augment_strength == "light":
        transforms.extend([
            RandomBrightness((0.95, 1.05), p=0.3),
            RandomContrast((0.95, 1.05), p=0.3),
            RandomNoise(0.01, p=0.2),
        ])
    elif augment_strength == "medium":
        transforms.extend([
            RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.97, 1.03), shear=2),
            RandomBrightness((0.9, 1.1), p=0.3),
            RandomContrast((0.9, 1.1), p=0.3),
            RandomGaussianBlur(3, (0.1, 0.5), p=0.2),
            RandomNoise(0.02, p=0.3),
            RandomErosionDilation(3, p=0.2),
        ])
    elif augment_strength == "heavy":
        transforms.extend([
            RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=3),
            RandomPerspective(0.1, p=0.3),
            ElasticDistortion(30, 4, p=0.2),
            RandomBrightness((0.8, 1.2), p=0.4),
            RandomContrast((0.8, 1.2), p=0.4),
            RandomGaussianBlur(3, (0.1, 1.0), p=0.3),
            RandomNoise(0.03, p=0.3),
            RandomErosionDilation(3, p=0.3),
            RandomInvert(p=0.05),
        ])

    # Always normalize at the end
    transforms.append(Normalize())

    return Compose(transforms)


def get_eval_transforms(image_size: int = 384) -> Compose:
    """Get evaluation transforms (no augmentation).

    Args:
        image_size: Target image size

    Returns:
        Composed transforms (normalization only)
    """
    return Compose([Normalize()])
