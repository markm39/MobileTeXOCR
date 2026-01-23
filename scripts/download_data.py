#!/usr/bin/env python3
"""
Dataset Download Script

Downloads the training datasets for handwritten LaTeX OCR:
- MathWriting (630K samples) - Google's dataset
- CROHME (10K samples) - Competition benchmark
- HME100K (100K samples) - Additional training data

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --dataset mathwriting
    python scripts/download_data.py --dataset crohme
    python scripts/download_data.py --output_dir /path/to/data
"""

import argparse
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
import urllib.request
import shutil


def run_command(cmd, desc=None):
    """Run a shell command and print output."""
    if desc:
        print(f"\n{desc}...")
    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return False
    return True


def download_file(url, output_path, desc=None):
    """Download a file with progress."""
    if desc:
        print(f"\n{desc}")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")

    try:
        urllib.request.urlretrieve(url, output_path, reporthook=download_progress)
        print()  # newline after progress
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_progress(count, block_size, total_size):
    """Progress callback for urlretrieve."""
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    sys.stdout.write(f"\r  Progress: {percent}%")
    sys.stdout.flush()


def download_mathwriting(output_dir: Path):
    """Download MathWriting dataset from Google Cloud Storage."""
    print("\n" + "=" * 60)
    print("MATHWRITING DATASET (2.9GB)")
    print("=" * 60)

    mathwriting_dir = output_dir / "mathwriting"

    # Check if already downloaded
    if (mathwriting_dir / "train").exists() and any((mathwriting_dir / "train").iterdir()):
        print(f"\nMathWriting already exists at {mathwriting_dir}")
        return True

    mathwriting_dir.mkdir(parents=True, exist_ok=True)

    # Direct download URL (doesn't require gsutil!)
    url = "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz"
    tarball = output_dir / "mathwriting-2024.tgz"

    print(f"\nDownloading from: {url}")
    print("This is ~2.9GB, may take a few minutes...")

    # Try wget first (faster, shows progress)
    if shutil.which("wget"):
        cmd = f"wget -q --show-progress {url} -O {tarball}"
        if not run_command(cmd, "Downloading with wget"):
            print("wget failed, trying curl...")
            if shutil.which("curl"):
                cmd = f"curl -L -o {tarball} {url}"
                if not run_command(cmd, "Downloading with curl"):
                    return False
            else:
                return False
    elif shutil.which("curl"):
        cmd = f"curl -L -o {tarball} {url}"
        if not run_command(cmd, "Downloading with curl"):
            return False
    else:
        print("Neither wget nor curl found. Trying Python urllib...")
        if not download_file(url, str(tarball), "Downloading MathWriting"):
            return False

    # Extract
    print(f"\nExtracting to {output_dir}...")
    cmd = f"tar -xzf {tarball} -C {output_dir}"
    if not run_command(cmd, "Extracting tarball"):
        return False

    # The tarball extracts to mathwriting-2024/, rename to mathwriting/
    extracted_dir = output_dir / "mathwriting-2024"
    if extracted_dir.exists():
        print("Reorganizing directory structure...")
        for item in extracted_dir.iterdir():
            dest = mathwriting_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        extracted_dir.rmdir()

    # Cleanup tarball
    if tarball.exists():
        tarball.unlink()
        print("Cleaned up tarball")

    print(f"\nMathWriting downloaded to: {mathwriting_dir}")

    # Show structure
    print("\nDirectory structure:")
    for item in sorted(mathwriting_dir.iterdir()):
        if item.is_dir():
            count = len(list(item.iterdir()))
            print(f"  {item.name}/  ({count} items)")
        else:
            print(f"  {item.name}")

    return True


def download_crohme(output_dir: Path):
    """Download CROHME dataset."""
    print("\n" + "=" * 60)
    print("CROHME DATASET")
    print("=" * 60)

    crohme_dir = output_dir / "crohme"
    crohme_dir.mkdir(parents=True, exist_ok=True)

    # CROHME 2019 from TC-11 or alternative sources
    # Note: Official downloads often require registration

    # Try Kaggle dataset (preprocessed version)
    print("\nCROHME requires manual download due to license restrictions.")
    print("\nOptions:")
    print("  1. Official: https://www.isical.ac.in/~crohme/")
    print("  2. TC-11 repository: http://tc11.cvc.uab.es/datasets/CROHME_2019_1")
    print("  3. Kaggle (preprocessed): https://www.kaggle.com/datasets/rtatman/handwritten-mathematical-expressions")
    print("\nAfter download, extract to:", crohme_dir)
    print("\nExpected structure:")
    print("  crohme/")
    print("    train/")
    print("      *.inkml")
    print("    test/")
    print("      *.inkml")

    # Create placeholder directories
    (crohme_dir / "train").mkdir(exist_ok=True)
    (crohme_dir / "test").mkdir(exist_ok=True)

    return False


def download_hme100k(output_dir: Path):
    """Download HME100K dataset."""
    print("\n" + "=" * 60)
    print("HME100K DATASET")
    print("=" * 60)

    hme_dir = output_dir / "hme100k"
    hme_dir.mkdir(parents=True, exist_ok=True)

    # HME100K is available on GitHub/Baidu
    print("\nHME100K download options:")
    print("  1. GitHub: https://github.com/Shylala/HME100K")
    print("  2. The dataset files are typically hosted on Baidu or Google Drive")
    print("\nAfter download, extract to:", hme_dir)

    # Try to clone the repo for metadata/labels
    if shutil.which("git"):
        repo_dir = hme_dir / "HME100K"
        if not repo_dir.exists():
            cmd = f"git clone https://github.com/Shylala/HME100K.git {repo_dir}"
            run_command(cmd, "Cloning HME100K repository")

    print("\nNote: Image files need to be downloaded separately from the links in the repo.")

    return False


def download_im2latex(output_dir: Path):
    """Download im2latex-100k dataset (rendered LaTeX for pretraining)."""
    print("\n" + "=" * 60)
    print("IM2LATEX-100K DATASET (Optional - for pretraining)")
    print("=" * 60)

    im2latex_dir = output_dir / "im2latex"
    im2latex_dir.mkdir(parents=True, exist_ok=True)

    # im2latex is often available on Harvard Dataverse
    print("\nim2latex-100k download:")
    print("  1. Harvard Dataverse: https://zenodo.org/record/56198")
    print("  2. Kaggle: https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k")
    print("\nThis dataset contains rendered (not handwritten) LaTeX")
    print("Useful for pretraining the encoder.")

    return False


def create_sample_data(output_dir: Path):
    """Create a small sample dataset for testing."""
    print("\n" + "=" * 60)
    print("CREATING SAMPLE DATA FOR TESTING")
    print("=" * 60)

    sample_dir = output_dir / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal InkML files for testing
    train_dir = sample_dir / "train"
    train_dir.mkdir(exist_ok=True)

    val_dir = sample_dir / "val"
    val_dir.mkdir(exist_ok=True)

    # Sample InkML content
    sample_inkml = '''<?xml version="1.0" encoding="UTF-8"?>
<ink xmlns="http://www.w3.org/2003/InkML">
  <annotation type="truth">x^2</annotation>
  <trace>0 0, 10 0, 10 10, 0 10, 0 0</trace>
  <trace>15 5, 20 0, 25 5</trace>
  <trace>22 2, 24 2</trace>
</ink>'''

    samples = [
        ("sample_001", "x^2"),
        ("sample_002", "y + 1"),
        ("sample_003", r"\frac{a}{b}"),
        ("sample_004", r"\sqrt{2}"),
        ("sample_005", "a + b = c"),
    ]

    for i, (name, latex) in enumerate(samples):
        inkml = f'''<?xml version="1.0" encoding="UTF-8"?>
<ink xmlns="http://www.w3.org/2003/InkML">
  <annotation type="truth">{latex}</annotation>
  <trace>{i*10} 0, {i*10+10} 0, {i*10+10} 10, {i*10} 10</trace>
</ink>'''

        # Put first 4 in train, last 1 in val
        if i < 4:
            (train_dir / f"{name}.inkml").write_text(inkml)
        else:
            (val_dir / f"{name}.inkml").write_text(inkml)

    print(f"Created sample data in: {sample_dir}")
    print(f"  Train: {len(list(train_dir.glob('*.inkml')))} samples")
    print(f"  Val: {len(list(val_dir.glob('*.inkml')))} samples")
    print("\nUse --datasets sample for quick testing")

    return True


def main():
    parser = argparse.ArgumentParser(description='Download training datasets')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for datasets')
    parser.add_argument('--dataset', type=str, nargs='+',
                        choices=['mathwriting', 'crohme', 'hme100k', 'im2latex', 'sample', 'all'],
                        default=['all'],
                        help='Which datasets to download')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DATASET DOWNLOAD SCRIPT")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")

    datasets = args.dataset
    if 'all' in datasets:
        datasets = ['mathwriting', 'crohme', 'hme100k', 'sample']

    results = {}

    if 'sample' in datasets:
        results['sample'] = create_sample_data(output_dir)

    if 'mathwriting' in datasets:
        results['mathwriting'] = download_mathwriting(output_dir)

    if 'crohme' in datasets:
        results['crohme'] = download_crohme(output_dir)

    if 'hme100k' in datasets:
        results['hme100k'] = download_hme100k(output_dir)

    if 'im2latex' in datasets:
        results['im2latex'] = download_im2latex(output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    for name, success in results.items():
        status = "Ready" if success else "Manual download required"
        print(f"  {name}: {status}")

    print("\n" + "=" * 60)
    print("QUICK START")
    print("=" * 60)

    if results.get('sample'):
        print("\nFor quick testing with sample data:")
        print("  python scripts/train_thunder.py --config configs/15m_base.yaml --data_dir ./data/sample")

    print("\nOnce datasets are downloaded:")
    print("  python scripts/train_thunder.py --config configs/15m_base.yaml")


if __name__ == '__main__':
    main()
