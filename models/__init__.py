"""
Handwritten LaTeX OCR Models

This module provides encoder-agnostic vision models for handwritten LaTeX recognition.
Supports both FastViTHD (Apple CVPR 2025) and Perception Encoder (Meta NeurIPS 2025).
"""

from models.encoder.base import BaseEncoder
from models.encoder.fastvithd import FastViTHD
from models.encoder.perception_encoder import PerceptionEncoder
from models.decoder.unified_decoder import UnifiedDecoder
from models.full_model import HandwrittenLaTeXOCR, ModelConfig

__all__ = [
    'BaseEncoder',
    'FastViTHD',
    'PerceptionEncoder',
    'UnifiedDecoder',
    'HandwrittenLaTeXOCR',
    'ModelConfig',
]
