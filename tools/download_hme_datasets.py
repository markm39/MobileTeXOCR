#!/usr/bin/env python3
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Download script for HME (Handwritten Mathematical Expression) datasets.

Supported datasets:
1. CROHME - Standard HME benchmark (~10K samples)
2. HME100K - TAL AI Platform dataset (100K samples) 
3. MathWriting - Google 2024 dataset (630K samples)

Usage:
    python tools/download_hme_datasets.py --dataset crohme
    python tools/download_hme_datasets.py --dataset all
"""

import argparse
import os
import sys
import urllib.request
import tarfile
import zipfile
from pathlib import Path
import hashlib
import shutil


# Dataset URLs and info
DATASETS = {
    "crohme": {
        "url": "https://paddleocr.bj.bcebos.com/dataset/CROHME.tar",
        "filename": "CROHME.tar",
        "extract_dir": "CROHME",
        "description": "CROHME 2014/2016/2019 benchmark dataset (~10K samples)",
        "size_mb": 50,
    },
    "hme100k": {
        # Note: HME100K requires manual download from TAL AI Platform
        # This is a placeholder - user needs to download manually
        "url": None,  
        "manual_url": "https://ai.100tal.com/dataset",
        "filename": "HME100K.zip",
        "extract_dir": "HME100K",
        "description": "TAL AI Platform HME dataset (100K samples) - manual download required",
        "size_mb": 2000,
    },
    "mathwriting": {
        # MathWriting dataset from Google (paper: arxiv.org/abs/2404.10690)
        "url": None,
        "manual_url": "https://github.com/google-research/google-research/tree/master/mathwriting",
        "filename": "mathwriting.tar.gz",
        "extract_dir": "MathWriting",
        "description": "Google MathWriting dataset (630K samples) - check GitHub for download",
        "size_mb": 5000,
    },
}


def download_progress(count, block_size, total_size):
    """Show download progress."""
    percent = int(count * block_size * 100 / total_size)
    downloaded = count * block_size / (1024 * 1024)
    total = total_size / (1024 * 1024)
    sys.stdout.write(f"\rDownloading: {percent}% ({downloaded:.1f}/{total:.1f} MB)")
    sys.stdout.flush()


def download_file(url, dest_path):
    """Download file with progress."""
    print(f"Downloading from: {url}")
    urllib.request.urlretrieve(url, dest_path, download_progress)
    print("\nDownload complete!")


def extract_archive(archive_path, extract_dir):
    """Extract tar or zip archive."""
    print(f"Extracting to: {extract_dir}")
    
    if archive_path.endswith('.tar') or archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:*') as tar:
            tar.extractall(extract_dir)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")
    
    print("Extraction complete!")


def download_crohme(data_dir):
    """Download and extract CROHME dataset."""
    dataset_info = DATASETS["crohme"]
    
    # Create data directory
    crohme_dir = os.path.join(data_dir, "CROHME")
    os.makedirs(crohme_dir, exist_ok=True)
    
    archive_path = os.path.join(data_dir, dataset_info["filename"])
    
    # Download if not exists
    if not os.path.exists(archive_path):
        download_file(dataset_info["url"], archive_path)
    else:
        print(f"Archive already exists: {archive_path}")
    
    # Extract
    extract_archive(archive_path, data_dir)
    
    # Verify structure
    expected_dirs = ["training", "evaluation"]
    for d in expected_dirs:
        path = os.path.join(crohme_dir, d)
        if os.path.exists(path):
            print(f"✓ Found: {path}")
        else:
            print(f"✗ Missing: {path}")
    
    print(f"\nCROHME dataset ready at: {crohme_dir}")
    return crohme_dir


def setup_hme100k_instructions(data_dir):
    """Print instructions for HME100K manual download."""
    print("\n" + "="*60)
    print("HME100K Dataset - Manual Download Required")
    print("="*60)
    print("""
HME100K is available from TAL AI Platform.

Steps:
1. Visit: https://ai.100tal.com/dataset
2. Register/login
3. Find and download HME100K dataset
4. Extract to: {data_dir}/HME100K/

Alternatively, you can use the TAMER download link:
https://disk.pku.edu.cn/link/AAF10CCC4D539543F68847A9010C607139

Expected structure:
{data_dir}/HME100K/
├── train/
│   ├── images/
│   └── labels.txt
└── test/
    ├── images/
    └── labels.txt
""".format(data_dir=data_dir))


def setup_mathwriting_instructions(data_dir):
    """Print instructions for MathWriting dataset."""
    print("\n" + "="*60)
    print("MathWriting Dataset - Google Research")
    print("="*60)
    print("""
MathWriting is the largest public HME dataset (630K samples).

Paper: "MathWriting: A Dataset for Handwritten Mathematical Expression Recognition"
       https://arxiv.org/abs/2404.10690

GitHub: https://github.com/google-research/google-research/tree/master/mathwriting

This dataset contains:
- 230K human-written samples
- 400K synthetic samples
- 254 distinct symbols including matrices, Greek alphabet

To use:
1. Check the GitHub repository for download instructions
2. Extract to: {data_dir}/MathWriting/
""".format(data_dir=data_dir))


def main():
    parser = argparse.ArgumentParser(description="Download HME datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="crohme",
        choices=["crohme", "hme100k", "mathwriting", "all"],
        help="Dataset to download (default: crohme)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./train_data",
        help="Directory to store datasets (default: ./train_data)"
    )
    args = parser.parse_args()
    
    # Create data directory
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Data directory: {data_dir}")
    
    if args.dataset == "crohme" or args.dataset == "all":
        print("\n" + "="*60)
        print("Downloading CROHME dataset...")
        print("="*60)
        download_crohme(data_dir)
    
    if args.dataset == "hme100k" or args.dataset == "all":
        setup_hme100k_instructions(data_dir)
    
    if args.dataset == "mathwriting" or args.dataset == "all":
        setup_mathwriting_instructions(data_dir)
    
    print("\n" + "="*60)
    print("Dataset setup complete!")
    print("="*60)
    print(f"""
To train the HME model:

1. Ultra-light (fastest, ~55% accuracy):
   python tools/train.py -c configs/rec/hme_latex_ocr_ultralight.yml

2. Balanced (recommended, ~62% accuracy):
   python tools/train.py -c configs/rec/hme_latex_ocr_balanced.yml

3. Accuracy (highest quality, ~65% accuracy):
   python tools/train.py -c configs/rec/hme_latex_ocr_accuracy.yml
""")


if __name__ == "__main__":
    main()

