"""
LaTeX Tokenizer for Handwritten Math OCR

Handles tokenization of:
1. LaTeX commands and symbols (~2000 tokens)
2. Location tokens for bounding boxes (~1000 tokens for coordinates)
3. Special tokens (BOS, EOS, PAD, SEP, LOC_START, LOC_END)
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Configuration for LaTeX tokenizer."""
    num_location_bins: int = 1000  # Discretization for coordinates
    max_seq_length: int = 512
    vocab_file: Optional[str] = None


# Core LaTeX vocabulary - common math symbols and commands
# This is a minimal set; the full vocabulary will be loaded from file
CORE_LATEX_VOCAB = [
    # Special tokens
    '<pad>', '<bos>', '<eos>', '<unk>', '<sep>',
    '<loc>', '</loc>',  # Bounding box markers

    # Basic characters
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',

    # Basic operators and punctuation
    '+', '-', '=', '<', '>', '(', ')', '[', ']', '{', '}',
    ',', '.', ':', ';', '!', '?', "'", '"', '/', '\\', '|',
    '*', '^', '_', '&', '#', '@', '%', '~', '`',

    # Greek letters
    '\\alpha', '\\beta', '\\gamma', '\\delta', '\\epsilon', '\\zeta',
    '\\eta', '\\theta', '\\iota', '\\kappa', '\\lambda', '\\mu',
    '\\nu', '\\xi', '\\pi', '\\rho', '\\sigma', '\\tau',
    '\\upsilon', '\\phi', '\\chi', '\\psi', '\\omega',
    '\\Alpha', '\\Beta', '\\Gamma', '\\Delta', '\\Epsilon', '\\Zeta',
    '\\Eta', '\\Theta', '\\Iota', '\\Kappa', '\\Lambda', '\\Mu',
    '\\Nu', '\\Xi', '\\Pi', '\\Rho', '\\Sigma', '\\Tau',
    '\\Upsilon', '\\Phi', '\\Chi', '\\Psi', '\\Omega',
    '\\varepsilon', '\\vartheta', '\\varpi', '\\varrho', '\\varsigma', '\\varphi',

    # Common math commands
    '\\frac', '\\sqrt', '\\sum', '\\prod', '\\int', '\\oint',
    '\\partial', '\\nabla', '\\infty', '\\pm', '\\mp', '\\times', '\\div',
    '\\cdot', '\\circ', '\\bullet', '\\star', '\\dagger', '\\ddagger',
    '\\leq', '\\geq', '\\neq', '\\approx', '\\equiv', '\\sim', '\\simeq',
    '\\subset', '\\supset', '\\subseteq', '\\supseteq', '\\in', '\\notin',
    '\\cup', '\\cap', '\\setminus', '\\emptyset',
    '\\forall', '\\exists', '\\neg', '\\wedge', '\\vee', '\\rightarrow',
    '\\leftarrow', '\\Rightarrow', '\\Leftarrow', '\\leftrightarrow',
    '\\to', '\\mapsto', '\\implies', '\\iff',

    # Delimiters
    '\\left', '\\right', '\\langle', '\\rangle', '\\lfloor', '\\rfloor',
    '\\lceil', '\\rceil', '\\lvert', '\\rvert', '\\lVert', '\\rVert',

    # Accents and modifiers
    '\\hat', '\\bar', '\\dot', '\\ddot', '\\tilde', '\\vec', '\\overline',
    '\\underline', '\\overbrace', '\\underbrace', '\\overrightarrow',

    # Spacing
    '\\quad', '\\qquad', '\\,', '\\:', '\\;', '\\ ', '\\!',

    # Text and formatting
    '\\text', '\\textbf', '\\textit', '\\mathrm', '\\mathbf', '\\mathit',
    '\\mathcal', '\\mathbb', '\\mathfrak', '\\boldsymbol',

    # Environments
    '\\begin', '\\end', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix',
    'cases', 'align', 'aligned', 'array', 'equation',

    # Matrix/array commands
    '\\\\', '\\hline', '\\cline', '\\multicolumn', '\\multirow',

    # Limits and subscripts/superscripts
    '\\lim', '\\limsup', '\\liminf', '\\max', '\\min', '\\sup', '\\inf',
    '\\log', '\\ln', '\\exp', '\\sin', '\\cos', '\\tan', '\\cot',
    '\\sec', '\\csc', '\\arcsin', '\\arccos', '\\arctan',
    '\\sinh', '\\cosh', '\\tanh', '\\coth',

    # Other common symbols
    '\\ldots', '\\cdots', '\\vdots', '\\ddots',
    '\\prime', '\\angle', '\\triangle', '\\square', '\\diamond',
    '\\perp', '\\parallel', '\\cong', '\\propto',
]


class LaTeXTokenizer:
    """Tokenizer for LaTeX with location tokens for bounding boxes.

    Vocabulary structure:
    - [0, num_location_bins): Location tokens for coordinates
    - [num_location_bins, num_location_bins + len(latex_vocab)): LaTeX tokens
    """

    # Special token indices (relative to latex vocab start)
    PAD_TOKEN = '<pad>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'
    SEP_TOKEN = '<sep>'
    LOC_START_TOKEN = '<loc>'
    LOC_END_TOKEN = '</loc>'

    def __init__(self, config: Optional[TokenizerConfig] = None):
        """
        Args:
            config: Tokenizer configuration
        """
        self.config = config or TokenizerConfig()
        self.num_location_bins = self.config.num_location_bins

        # Build vocabulary
        if self.config.vocab_file and Path(self.config.vocab_file).exists():
            self._load_vocab(self.config.vocab_file)
        else:
            self._build_default_vocab()

        # Compile regex for tokenization
        self._compile_patterns()

    def _build_default_vocab(self):
        """Build vocabulary from core LaTeX tokens."""
        # Remove duplicates while preserving order
        seen = set()
        unique_vocab = []
        for token in CORE_LATEX_VOCAB:
            if token not in seen:
                seen.add(token)
                unique_vocab.append(token)

        self.latex_vocab = unique_vocab
        self.latex_to_id = {tok: i for i, tok in enumerate(self.latex_vocab)}
        self.id_to_latex = {i: tok for i, tok in enumerate(self.latex_vocab)}

    def _load_vocab(self, vocab_file: str):
        """Load vocabulary from JSON file."""
        with open(vocab_file, 'r') as f:
            data = json.load(f)
        self.latex_vocab = data['tokens']
        self.latex_to_id = {tok: i for i, tok in enumerate(self.latex_vocab)}
        self.id_to_latex = {i: tok for i, tok in enumerate(self.latex_vocab)}

    def _compile_patterns(self):
        """Compile regex patterns for LaTeX tokenization."""
        # Sort by length (longest first) to match longer tokens first
        latex_commands = sorted(
            [t for t in self.latex_vocab if t.startswith('\\')],
            key=len, reverse=True
        )
        # Escape special regex chars
        escaped = [re.escape(cmd) for cmd in latex_commands]
        command_pattern = '|'.join(escaped) if escaped else r'(?!)'

        # Full pattern: LaTeX commands OR single characters
        self.tokenize_pattern = re.compile(
            f'({command_pattern}|[a-zA-Z0-9]|[^a-zA-Z0-9\\s])'
        )

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including location tokens."""
        return self.num_location_bins + len(self.latex_vocab)

    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self.num_location_bins + self.latex_to_id[self.PAD_TOKEN]

    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID."""
        return self.num_location_bins + self.latex_to_id[self.BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return self.num_location_bins + self.latex_to_id[self.EOS_TOKEN]

    @property
    def unk_token_id(self) -> int:
        """Unknown token ID."""
        return self.num_location_bins + self.latex_to_id[self.UNK_TOKEN]

    @property
    def sep_token_id(self) -> int:
        """Separator token ID."""
        return self.num_location_bins + self.latex_to_id[self.SEP_TOKEN]

    @property
    def loc_start_id(self) -> int:
        """Location start token ID."""
        return self.num_location_bins + self.latex_to_id[self.LOC_START_TOKEN]

    @property
    def loc_end_id(self) -> int:
        """Location end token ID."""
        return self.num_location_bins + self.latex_to_id[self.LOC_END_TOKEN]

    def coordinate_to_token(self, coord: float) -> int:
        """Convert normalized coordinate [0, 1] to location token ID.

        Args:
            coord: Coordinate value in [0, 1]

        Returns:
            Token ID in [0, num_location_bins)
        """
        coord = max(0.0, min(1.0, coord))  # Clamp to [0, 1]
        return int(coord * (self.num_location_bins - 1))

    def token_to_coordinate(self, token_id: int) -> float:
        """Convert location token ID back to coordinate.

        Args:
            token_id: Token ID in [0, num_location_bins)

        Returns:
            Coordinate value in [0, 1]
        """
        return token_id / (self.num_location_bins - 1)

    def is_location_token(self, token_id: int) -> bool:
        """Check if token ID is a location token."""
        return 0 <= token_id < self.num_location_bins

    def encode_latex(self, latex: str) -> List[int]:
        """Tokenize LaTeX string to token IDs.

        Args:
            latex: LaTeX string

        Returns:
            List of token IDs (excluding location tokens)
        """
        tokens = self.tokenize_pattern.findall(latex)
        ids = []
        for tok in tokens:
            if tok in self.latex_to_id:
                ids.append(self.num_location_bins + self.latex_to_id[tok])
            else:
                ids.append(self.unk_token_id)
        return ids

    def decode_latex(self, token_ids: List[int]) -> str:
        """Decode token IDs back to LaTeX string.

        Args:
            token_ids: List of token IDs

        Returns:
            LaTeX string
        """
        parts = []
        for tid in token_ids:
            if self.is_location_token(tid):
                # Skip location tokens in LaTeX decoding
                continue
            latex_id = tid - self.num_location_bins
            if 0 <= latex_id < len(self.latex_vocab):
                tok = self.id_to_latex[latex_id]
                if tok not in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.SEP_TOKEN]:
                    parts.append(tok)
        return ''.join(parts)

    def encode_bbox(self, bbox: Tuple[float, float, float, float]) -> List[int]:
        """Encode bounding box as token sequence.

        Args:
            bbox: (x1, y1, x2, y2) normalized coordinates in [0, 1]

        Returns:
            Token sequence: [<loc>, x1, y1, x2, y2, </loc>]
        """
        x1, y1, x2, y2 = bbox
        return [
            self.loc_start_id,
            self.coordinate_to_token(x1),
            self.coordinate_to_token(y1),
            self.coordinate_to_token(x2),
            self.coordinate_to_token(y2),
            self.loc_end_id,
        ]

    def decode_bbox(self, tokens: List[int]) -> Optional[Tuple[float, float, float, float]]:
        """Decode bounding box from token sequence.

        Args:
            tokens: Token sequence containing bbox

        Returns:
            (x1, y1, x2, y2) or None if invalid
        """
        # Find <loc> and </loc>
        try:
            start_idx = tokens.index(self.loc_start_id)
            end_idx = tokens.index(self.loc_end_id)
            if end_idx - start_idx != 5:  # Must have exactly 4 coords between markers
                return None
            coords = tokens[start_idx + 1:end_idx]
            return tuple(self.token_to_coordinate(c) for c in coords)
        except (ValueError, IndexError):
            return None

    def encode_sample(
        self,
        latex: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode a complete sample (bbox + LaTeX).

        Args:
            latex: LaTeX string
            bbox: Optional bounding box (x1, y1, x2, y2)
            add_special_tokens: Whether to add BOS/EOS

        Returns:
            Complete token sequence
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.bos_token_id)

        if bbox is not None:
            tokens.extend(self.encode_bbox(bbox))

        tokens.extend(self.encode_latex(latex))

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return tokens

    def encode_multi_region(
        self,
        regions: List[Tuple[Tuple[float, float, float, float], str]],
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode multiple regions (bbox, latex) pairs.

        Args:
            regions: List of (bbox, latex) tuples
            add_special_tokens: Whether to add BOS/EOS

        Returns:
            Complete token sequence with <sep> between regions
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.bos_token_id)

        for i, (bbox, latex) in enumerate(regions):
            if i > 0:
                tokens.append(self.sep_token_id)
            tokens.extend(self.encode_bbox(bbox))
            tokens.extend(self.encode_latex(latex))

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return tokens

    def decode_sequence(
        self,
        token_ids: List[int]
    ) -> List[Tuple[Optional[Tuple[float, float, float, float]], str]]:
        """Decode full sequence back to list of (bbox, latex) pairs.

        Args:
            token_ids: Token sequence

        Returns:
            List of (bbox, latex) tuples
        """
        results = []

        # Split by separator
        current_tokens = []
        for tid in token_ids:
            if tid == self.sep_token_id:
                if current_tokens:
                    results.append(self._decode_region(current_tokens))
                    current_tokens = []
            elif tid not in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                current_tokens.append(tid)

        # Don't forget last region
        if current_tokens:
            results.append(self._decode_region(current_tokens))

        return results

    def _decode_region(
        self,
        tokens: List[int]
    ) -> Tuple[Optional[Tuple[float, float, float, float]], str]:
        """Decode a single region."""
        bbox = self.decode_bbox(tokens)

        # Extract LaTeX tokens (everything after </loc>)
        try:
            loc_end_idx = tokens.index(self.loc_end_id)
            latex_tokens = tokens[loc_end_idx + 1:]
        except ValueError:
            latex_tokens = tokens

        latex = self.decode_latex(latex_tokens)
        return (bbox, latex)

    def save_vocab(self, path: str):
        """Save vocabulary to JSON file."""
        data = {'tokens': self.latex_vocab}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_file(cls, vocab_file: str, **kwargs) -> 'LaTeXTokenizer':
        """Load tokenizer from vocabulary file."""
        config = TokenizerConfig(vocab_file=vocab_file, **kwargs)
        return cls(config)
