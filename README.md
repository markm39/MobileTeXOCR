# Handwritten LaTeX OCR

A unified VLM-OCR system for recognizing handwritten mathematical expressions and converting them to LaTeX. Designed for on-device inference with CoreML support.

## Architecture

This implementation uses the **unified text spotting** paradigm (Jan 2026 SOTA), where a single model outputs both bounding boxes and LaTeX text in one forward pass:

```
[Image] -> [Vision Encoder] -> [Unified Decoder] -> [(bbox, LaTeX), ...]
```

### Key Features

- **Encoder-Agnostic Design**: Supports both FastViTHD (Apple CVPR 2025) and Perception Encoder (Meta NeurIPS 2025)
- **Unified Output**: Generates bounding boxes AND LaTeX in a single autoregressive sequence
- **Mobile-Ready**: CoreML export with INT8/INT4 quantization for iOS deployment
- **~300-400M Parameters**: Balanced for accuracy and on-device performance

### Model Variants

| Encoder | Size | Params | Best For |
|---------|------|--------|----------|
| FastViTHD-Base | 384px | ~300M | iOS/CoreML deployment |
| FastViTHD-Small | 384px | ~150M | Real-time inference |
| Perception Encoder-Tiny | 384px | ~200M | Maximum accuracy |

## Installation

```bash
# Basic installation
pip install -e .

# With training dependencies
pip install -e ".[train]"

# With export dependencies
pip install -e ".[export]"

# All dependencies
pip install -e ".[all]"
```

## Quick Start

### Inference

```python
from models import HandwrittenLaTeXOCR

# Load model
model = HandwrittenLaTeXOCR.from_pretrained("path/to/checkpoint")

# Inference
from PIL import Image
import torchvision.transforms.functional as TF

img = Image.open("handwritten_math.png")
img_tensor = TF.to_tensor(img).unsqueeze(0)

output = model(img_tensor)
for bbox, latex in output.predictions[0]:
    print(f"Box: {bbox}, LaTeX: {latex}")
```

### Training

```bash
# Train with default settings
python -m training.train --data_dir ./data --output_dir ./outputs

# Train with custom config
python -m training.train --config configs/base.yaml
```

Or use the Colab notebook for training on H100/A100: `training/colab_train.ipynb`

## Project Structure

```
handwritten-latex-ocr/
├── models/
│   ├── encoder/
│   │   ├── base.py              # Encoder-agnostic base class
│   │   ├── fastvithd.py         # FastViTHD (Apple CVPR 2025)
│   │   └── perception_encoder.py # Perception Encoder (Meta NeurIPS 2025)
│   ├── decoder/
│   │   ├── unified_decoder.py   # Autoregressive text spotting decoder
│   │   └── tokenizer.py         # LaTeX + location tokenizer
│   └── full_model.py            # End-to-end model
├── data/
│   ├── mathwriting.py           # MathWriting dataset (630K)
│   ├── crohme.py                # CROHME benchmark (10K)
│   ├── hme100k.py               # HME100K dataset (100K)
│   ├── combined.py              # Combined dataset loader
│   └── augmentations.py         # Data augmentations
├── training/
│   ├── trainer.py               # Training loop with AMP
│   ├── metrics.py               # ExpRate, Symbol accuracy, BLEU
│   ├── train.py                 # CLI training script
│   └── colab_train.ipynb        # Colab notebook
├── export/
│   ├── to_onnx.py               # ONNX export
│   └── to_coreml.py             # CoreML export with quantization
├── ios/
│   ├── LaTeXOCR.swift           # Swift wrapper for CoreML
│   └── LaTeXTokenizer.swift     # iOS tokenizer
└── configs/                      # YAML configurations
```

## Datasets

| Dataset | Size | Type | Source |
|---------|------|------|--------|
| MathWriting | 630K | Online handwritten | Google 2024 |
| CROHME | 10K | Online handwritten | Competition benchmark |
| HME100K | 100K | Offline handwritten | Academic dataset |

Download instructions:

```python
from data.mathwriting import download_mathwriting
from data.crohme import download_crohme
from data.hme100k import download_hme100k

download_mathwriting("./data")
download_crohme("./data")
download_hme100k("./data")
```

## Export

### ONNX

```python
from export import export_to_onnx
from models import HandwrittenLaTeXOCR

model = HandwrittenLaTeXOCR.from_pretrained("checkpoint/")
export_to_onnx(model, "model.onnx")
```

### CoreML (iOS)

```python
from export import export_to_coreml

export_to_coreml(
    model,
    "model.mlpackage",
    quantize="int8",  # Options: None, "float16", "int8", "int4"
)
```

## iOS Integration

```swift
import Foundation

// Initialize model
let ocr = try LaTeXOCR()

// Recognize from image
let result = try ocr.recognize(image: uiImage)
print("LaTeX: \(result.latex)")
print("BBox: \(result.boundingBox)")
```

## Training Tips

1. **Stage 1 - Encoder warmup** (1 epoch): Freeze encoder, train decoder only
2. **Stage 2 - Full training** (10-20 epochs): Unfreeze encoder, train end-to-end
3. **Stage 3 - Fine-tune** (5 epochs): Fine-tune on CROHME for benchmark

**Expected metrics on CROHME:**
- Expression Recognition Rate (ExpRate): >60%
- Symbol Accuracy: >90%

## References

### Vision Encoders
- [FastVLM - Apple CVPR 2025](https://github.com/apple/ml-fastvlm)
- [Perception Encoder - Meta NeurIPS 2025](https://github.com/facebookresearch/perception_models)

### OCR Models
- [HunyuanOCR - Tencent 2025](https://github.com/Tencent-Hunyuan/HunyuanOCR)
- [VISTA-OCR - 2025](https://arxiv.org/html/2504.03621v1)

### Datasets
- [MathWriting - Google 2024](https://arxiv.org/abs/2404.10690)

## License

Apache License 2.0
