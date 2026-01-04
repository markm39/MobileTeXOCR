#!/usr/bin/env python3
"""
Simple test script for HME (Handwritten Math Expression) model.
Usage: python tools/test_hme_model.py --image path/to/image.png
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paddle
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description='Test HME model on an image')
    parser.add_argument('--image', '-i', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', '-c', type=str,
                        default='./output/rec/hme_ultralight/best_accuracy',
                        help='Path to checkpoint')
    parser.add_argument('--config', type=str,
                        default='./configs/rec/hme_latex_ocr_ultralight.yml',
                        help='Path to config file')
    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set pretrained model path
    config['Global']['pretrained_model'] = args.checkpoint

    # Import ppocr modules
    from ppocr.modeling.architectures import build_model
    from ppocr.utils.save_load import load_model
    from ppocr.data.imaug import create_operators, transform
    from ppocr.postprocess import build_post_process

    print(f"Loading checkpoint: {args.checkpoint}")

    # Build model
    model = build_model(config['Architecture'])
    load_model(config, model, model_type='rec')
    model.eval()

    # Build postprocessor
    post_process = build_post_process(config['PostProcess'], config['Global'])

    # Load and preprocess image
    print(f"Loading image: {args.image}")

    # Use eval transforms from config
    eval_transforms = config['Eval']['dataset']['transforms']
    ops = create_operators(eval_transforms, config['Global'])

    # Read image as bytes
    with open(args.image, 'rb') as f:
        img_bytes = f.read()

    data = {'image': img_bytes, 'label': ''}  # Empty label for inference

    # Apply transforms
    for op in ops:
        data = op(data)
        if data is None:
            print("Error: Transform returned None")
            sys.exit(1)

    # data is now a list: [image, label]
    image = data[0]
    label = data[1]

    print(f"Preprocessed image shape: {image.shape}")

    # Resize to height=32 maintaining aspect ratio
    import cv2
    target_height = 32
    if len(image.shape) == 3:
        c, h, w = image.shape
        ratio = target_height / h
        new_w = max(1, int(w * ratio))
        # Transpose to HWC for cv2
        img_hwc = image.transpose(1, 2, 0) if c == 1 else image.transpose(1, 2, 0)
        if img_hwc.shape[-1] == 1:
            img_hwc = img_hwc.squeeze(-1)
        img_resized = cv2.resize(img_hwc.astype(np.float32), (new_w, target_height))
        if len(img_resized.shape) == 2:
            img_resized = img_resized[np.newaxis, :, :]
        else:
            img_resized = img_resized.transpose(2, 0, 1)
        image = img_resized

    print(f"Resized image shape: {image.shape}")

    # Create batch with mask
    # Image is [1, H, W], need to add batch dim
    if len(image.shape) == 3:
        image = image[np.newaxis, :, :, :]  # [1, 1, H, W]

    image_tensor = paddle.to_tensor(image.astype(np.float32))
    mask = paddle.ones([1, 1, image.shape[2], image.shape[3]], dtype='float32')

    # For inference, we need to do autoregressive decoding
    # The model expects (image, mask, labels) tuple for teacher forcing
    # For true inference, we need to generate tokens one by one

    # Get vocab info
    dict_path = config['Global']['character_dict_path']
    with open(dict_path, 'r') as f:
        vocab = [line.strip() for line in f]

    # 'eos' is at index 0, 'sos' is at index 1 in the vocab file
    EOS_IDX = 0
    SOS_IDX = 1 if len(vocab) > 1 and vocab[1] == 'sos' else 0
    max_len = config['Global'].get('max_text_length', 256)

    print(f"Vocab size: {len(vocab)}, SOS_IDX: {SOS_IDX}, EOS_IDX: {EOS_IDX}")

    # Autoregressive decoding
    decoded_tokens = [SOS_IDX]

    with paddle.no_grad():
        for step in range(max_len - 1):
            # Pad tokens to max_len
            padded_tokens = decoded_tokens + [0] * (max_len - len(decoded_tokens))
            labels = paddle.to_tensor([padded_tokens], dtype='int64')

            # Forward pass
            output = model((image_tensor, mask, labels))

            if isinstance(output, dict):
                output = output.get('head_out', output)
            if isinstance(output, tuple):
                output = output[0]  # Get logits

            # Get prediction for current position
            # Position len(decoded_tokens)-1 predicts the next token
            current_pos = len(decoded_tokens) - 1
            next_token_logits = output[0, current_pos, :]
            next_token = int(paddle.argmax(next_token_logits).numpy())

            if next_token == EOS_IDX:
                print(f"EOS reached at step {step}")
                break

            decoded_tokens.append(next_token)

            if step < 5:
                print(f"  Step {step}: predicted token {next_token} = '{vocab[next_token] if next_token < len(vocab) else '?'}'")

    print(f"\nDecoded {len(decoded_tokens)} tokens")

    # Convert to LaTeX (skip SOS)
    latex_tokens = []
    for idx in decoded_tokens[1:]:  # Skip SOS
        if idx < len(vocab):
            latex_tokens.append(vocab[idx])
        else:
            latex_tokens.append(f'<unk:{idx}>')

    latex = ' '.join(latex_tokens)
    print(f"\nPredicted LaTeX: {latex}")

    # Also show ground truth if available
    label_file = args.image.replace('/images/', '/').replace('.jpg', '').replace('.png', '').replace('.bmp', '')
    # Try to find the label
    eval_label_path = config['Eval']['dataset']['label_file_list'][0]
    image_name = os.path.basename(args.image)
    try:
        with open(eval_label_path, 'r') as f:
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
