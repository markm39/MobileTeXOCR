"""
Evaluation Metrics for Handwritten Math OCR

Standard metrics:
- Expression Recognition Rate (ExpRate): Exact match accuracy
- Symbol Accuracy: Character/token-level accuracy
- BLEU Score: N-gram overlap for partial credit
"""

from typing import List, Tuple, Dict, Optional
from collections import Counter
import math


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertions, deletions, substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalize_latex(latex: str) -> str:
    """Normalize LaTeX for comparison.

    Handles common variations that should be considered equivalent.
    """
    # Remove extra whitespace
    latex = ' '.join(latex.split())

    # Common normalizations
    replacements = [
        ('\\left(', '('),
        ('\\right)', ')'),
        ('\\left[', '['),
        ('\\right]', ']'),
        ('\\left\\{', '\\{'),
        ('\\right\\}', '\\}'),
        (' ', ''),  # Remove all spaces for comparison
    ]

    for old, new in replacements:
        latex = latex.replace(old, new)

    return latex


class ExpRate:
    """Expression Recognition Rate metric.

    Measures exact match accuracy between predicted and ground truth.
    """

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Whether to normalize LaTeX before comparison
        """
        self.normalize = normalize
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.correct = 0
        self.total = 0

    def update(self, predictions: List[str], targets: List[str]):
        """Update with a batch of predictions.

        Args:
            predictions: List of predicted LaTeX strings
            targets: List of ground truth LaTeX strings
        """
        for pred, target in zip(predictions, targets):
            if self.normalize:
                pred = normalize_latex(pred)
                target = normalize_latex(target)

            if pred == target:
                self.correct += 1
            self.total += 1

    def compute(self) -> float:
        """Compute the metric.

        Returns:
            Expression recognition rate (0-1)
        """
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class SymbolAccuracy:
    """Symbol-level accuracy using edit distance.

    Computes 1 - (edit_distance / max_length) averaged over samples.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.reset()

    def reset(self):
        self.total_accuracy = 0.0
        self.count = 0

    def update(self, predictions: List[str], targets: List[str]):
        """Update with a batch of predictions."""
        for pred, target in zip(predictions, targets):
            if self.normalize:
                pred = normalize_latex(pred)
                target = normalize_latex(target)

            if len(target) == 0 and len(pred) == 0:
                self.total_accuracy += 1.0
            elif len(target) == 0:
                self.total_accuracy += 0.0
            else:
                distance = levenshtein_distance(pred, target)
                max_len = max(len(pred), len(target))
                accuracy = 1.0 - (distance / max_len)
                self.total_accuracy += max(0.0, accuracy)

            self.count += 1

    def compute(self) -> float:
        """Compute the metric."""
        if self.count == 0:
            return 0.0
        return self.total_accuracy / self.count


class BLEU:
    """BLEU score for partial credit on LaTeX generation."""

    def __init__(self, max_n: int = 4, weights: Optional[Tuple[float, ...]] = None):
        """
        Args:
            max_n: Maximum n-gram size
            weights: Weights for each n-gram (default: uniform)
        """
        self.max_n = max_n
        self.weights = weights or tuple(1.0 / max_n for _ in range(max_n))
        self.reset()

    def reset(self):
        self.scores = []

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-gram counts."""
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def _tokenize(self, latex: str) -> List[str]:
        """Simple tokenization for BLEU."""
        # Split on spaces and special characters
        tokens = []
        current = ""
        for char in latex:
            if char in ' {}[]()^_':
                if current:
                    tokens.append(current)
                    current = ""
                if char != ' ':
                    tokens.append(char)
            else:
                current += char
        if current:
            tokens.append(current)
        return tokens

    def update(self, predictions: List[str], targets: List[str]):
        """Update with a batch."""
        for pred, target in zip(predictions, targets):
            pred_tokens = self._tokenize(pred)
            target_tokens = self._tokenize(target)

            if len(pred_tokens) == 0 or len(target_tokens) == 0:
                self.scores.append(0.0)
                continue

            # Compute n-gram precisions
            precisions = []
            for n in range(1, self.max_n + 1):
                pred_ngrams = self._get_ngrams(pred_tokens, n)
                target_ngrams = self._get_ngrams(target_tokens, n)

                if len(pred_ngrams) == 0:
                    precisions.append(0.0)
                    continue

                overlap = sum((pred_ngrams & target_ngrams).values())
                total = sum(pred_ngrams.values())
                precisions.append(overlap / total if total > 0 else 0.0)

            # Brevity penalty
            bp = min(1.0, math.exp(1 - len(target_tokens) / max(len(pred_tokens), 1)))

            # Weighted geometric mean
            if all(p > 0 for p in precisions):
                log_precision = sum(w * math.log(p) for w, p in zip(self.weights, precisions))
                score = bp * math.exp(log_precision)
            else:
                score = 0.0

            self.scores.append(score)

    def compute(self) -> float:
        """Compute average BLEU score."""
        if len(self.scores) == 0:
            return 0.0
        return sum(self.scores) / len(self.scores)


def compute_metrics(
    predictions: List[str],
    targets: List[str],
    normalize: bool = True,
) -> Dict[str, float]:
    """Compute all metrics for a batch.

    Args:
        predictions: Predicted LaTeX strings
        targets: Ground truth LaTeX strings
        normalize: Whether to normalize LaTeX

    Returns:
        Dictionary of metric names to values
    """
    exp_rate = ExpRate(normalize)
    symbol_acc = SymbolAccuracy(normalize)
    bleu = BLEU()

    exp_rate.update(predictions, targets)
    symbol_acc.update(predictions, targets)
    bleu.update(predictions, targets)

    return {
        'exp_rate': exp_rate.compute(),
        'symbol_accuracy': symbol_acc.compute(),
        'bleu': bleu.compute(),
    }


def compute_bbox_metrics(
    pred_bboxes: List[Optional[Tuple[float, ...]]],
    target_bboxes: List[Optional[Tuple[float, ...]]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute bounding box metrics.

    Args:
        pred_bboxes: Predicted bounding boxes (x1, y1, x2, y2)
        target_bboxes: Ground truth bounding boxes
        iou_threshold: IoU threshold for match

    Returns:
        Dictionary with bbox metrics
    """
    correct = 0
    total = 0

    for pred, target in zip(pred_bboxes, target_bboxes):
        if target is None:
            continue

        total += 1

        if pred is None:
            continue

        # Compute IoU
        x1 = max(pred[0], target[0])
        y1 = max(pred[1], target[1])
        x2 = min(pred[2], target[2])
        y2 = min(pred[3], target[3])

        if x1 < x2 and y1 < y2:
            intersection = (x2 - x1) * (y2 - y1)
            pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
            target_area = (target[2] - target[0]) * (target[3] - target[1])
            union = pred_area + target_area - intersection
            iou = intersection / union if union > 0 else 0
        else:
            iou = 0

        if iou >= iou_threshold:
            correct += 1

    return {
        'bbox_accuracy': correct / total if total > 0 else 0.0,
        'bbox_total': total,
    }
