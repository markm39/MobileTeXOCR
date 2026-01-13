#!/usr/bin/env python3
"""
Test script for HME V2 model with proper autoregressive inference.
Usage: python tools/test_hme_model_v2.py --image path/to/image.png
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paddle
import numpy as np
import yaml
import cv2


def main():
    parser = argparse.ArgumentParser(description='Test HME V2 model on an image')
    parser.add_argument('--image', '-i', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', '-c', type=str,
                        default='./output/rec/hme_ultralight_v2/best_accuracy',
                        help='Path to checkpoint')
    parser.add_argument('--config', type=str,
                        default='./configs/rec/hme_latex_ocr_ultralight_v2.yml',
                        help='Path to config file')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['Global']['pretrained_model'] = args.checkpoint

    from ppocr.modeling.architectures import build_model
    from ppocr.utils.save_load import load_model

    print(f"Loading checkpoint: {args.checkpoint}")

    model = build_model(config['Architecture'])
    load_model(config, model, model_type='rec')

    # Set to inference mode
    for layer in model.sublayers():
        layer.training = False

    dict_path = config['Global']['character_dict_path']
    with open(dict_path, 'r') as f:
        vocab = [line.strip() for line in f]

    EOS_IDX = 0
    SOS_IDX = 1

    print(f"Vocab size: {len(vocab) + 2} (including EOS/SOS)")
    print(f"Loading image: {args.image}")

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image: {args.image}")
        sys.exit(1)

    target_height = 32
    h, w = img.shape
    ratio = target_height / h
    new_w = max(1, int(w * ratio))
    img = cv2.resize(img, (new_w, target_height))

    img = img.astype(np.float32) / 255.0
    img = 1.0 - img

    img = img[np.newaxis, np.newaxis, :, :]

    print(f"Preprocessed image shape: {img.shape}")

    image_tensor = paddle.to_tensor(img)

    with paddle.no_grad():
        output = model(image_tensor)

        if isinstance(output, dict):
            logits = output.get('logits', output.get('head_out'))
        elif isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

    preds = paddle.argmax(logits, axis=-1).numpy()[0]

    print(f"\nRaw predictions (first 20): {preds[:20].tolist()}")

    decoded_tokens = []
    for i, token in enumerate(preds):
        if token == EOS_IDX:
            print(f"EOS found at position {i}")
            break
        if token == SOS_IDX:
            continue
        decoded_tokens.append(int(token))
        if i < 10:
            tok_str = vocab[token - 2] if 2 <= token < len(vocab) + 2 else f'<special:{token}>'
            print(f"  Position {i}: token {token} = '{tok_str}'")

    print(f"\nDecoded {len(decoded_tokens)} tokens")

    latex_tokens = []
    for idx in decoded_tokens:
        vocab_idx = idx - 2
        if 0 <= vocab_idx < len(vocab):
            latex_tokens.append(vocab[vocab_idx])
        else:
            latex_tokens.append(f'<unk:{idx}>')

    latex = ' '.join(latex_tokens)
    print(f"\nPredicted LaTeX: {latex}")

    validation_config = config.get('Validation', config.get('Test', {}))
    if not validation_config:
        for key in config:
            if 'val' in key.lower() or 'test' in key.lower():
                validation_config = config[key]
                break

    label_files = []
    if validation_config and 'dataset' in validation_config:
        label_files = validation_config['dataset'].get('label_file_list', [])

    image_name = os.path.basename(args.image)
    for label_path in label_files:
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    if image_name in line:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            print(f"\nGround Truth: {parts[1]}")
                        break
        except:
            pass


if __name__ == '__main__':
    main()
