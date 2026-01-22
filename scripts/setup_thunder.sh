#!/bin/bash
# Thunder Compute A100 Environment Setup Script
#
# This script sets up the environment for training on Thunder Compute.
# Run this once when starting a new instance.
#
# Usage:
#   chmod +x scripts/setup_thunder.sh
#   ./scripts/setup_thunder.sh
#
# After setup, run training with:
#   python scripts/train_thunder.py --config configs/15m_base.yaml

set -e  # Exit on error

echo "=========================================="
echo "MobileTeXOCR - Thunder Compute Setup"
echo "=========================================="

# Check for GPU
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi

# Create virtual environment if it doesn't exist
echo ""
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created virtual environment"
fi

# Activate virtual environment
source venv/bin/activate
echo "Activated virtual environment"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo ""
echo "Installing dependencies..."
pip install \
    pyyaml \
    pillow \
    numpy \
    tqdm \
    matplotlib \
    scipy

# Install optional dependencies
echo ""
echo "Installing optional dependencies..."
pip install wandb || echo "wandb installation failed (optional)"

# Verify PyTorch CUDA
echo ""
echo "Verifying PyTorch CUDA..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Create data directory
echo ""
echo "Setting up directories..."
mkdir -p data
mkdir -p outputs

# Download datasets if not present
echo ""
echo "Checking for datasets..."
if [ ! -d "data/mathwriting" ]; then
    echo "MathWriting dataset not found."
    echo "Please download from: https://arxiv.org/abs/2404.10690"
    echo "And extract to: data/mathwriting/"
else
    echo "MathWriting dataset found"
fi

if [ ! -d "data/crohme" ]; then
    echo "CROHME dataset not found."
    echo "Please download from: https://www.isical.ac.in/~crohme/"
    echo "And extract to: data/crohme/"
else
    echo "CROHME dataset found"
fi

if [ ! -d "data/hme100k" ]; then
    echo "HME100K dataset not found."
    echo "Please download from: https://github.com/Shylala/HME100K"
    echo "And extract to: data/hme100k/"
else
    echo "HME100K dataset found"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To start training:"
echo "  source venv/bin/activate"
echo "  python scripts/train_thunder.py --config configs/15m_base.yaml"
echo ""
echo "To resume training from checkpoint:"
echo "  python scripts/train_thunder.py --config configs/15m_base.yaml --resume outputs/latex_ocr_15m/checkpoints/latest.pt"
echo ""
