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


def load_vocab(dict_path):
    """Load vocabulary from dictionary file."""
    vocab = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            token = line.strip()
            vocab[idx] = token
    # Add special tokens
    vocab[0] = '<pad>'
    vocab[1] = '<sos>'
    vocab[2] = '<eos>'
    return vocab


def preprocess_image(image_path, target_height=32):
    """Load and preprocess image using ppocr transforms."""
    import cv2
    from ppocr.data.imaug import create_operators, transform

    # Define transforms matching training config
    transforms_config = [
        {'DecodeImage': {'channel_first': False}},
        {'NormalizeImage': {'mean': [0, 0, 0], 'std': [1, 1, 1], 'order': 'hwc'}},
        {'GrayImageChannelFormat': {'inverse': True}},
    ]
    ops = create_operators(transforms_config)

    # Read image as bytes (what DecodeImage expects)
    with open(image_path, 'rb') as f:
        img_bytes = f.read()

    data = {'image': img_bytes}

    # Apply transforms
    for op in ops:
        data = op(data)

    img = data['image']  # Now [1, H, W] float32

    # Resize maintaining aspect ratio
    _, h, w = img.shape
    ratio = target_height / h
    new_w = max(1, int(w * ratio))

    # Resize using cv2
    img_resized = cv2.resize(img.transpose(1, 2, 0), (new_w, target_height))
    if len(img_resized.shape) == 2:
        img_resized = img_resized[np.newaxis, :, :]
    else:
        img_resized = img_resized.transpose(2, 0, 1)

    # Add batch dimension [1, 1, H, W]
    img = img_resized[np.newaxis, :, :, :]

    return img.astype(np.float32)


def greedy_decode(model, memory, vocab, max_len=256):
    """Greedy decoding from encoder memory."""
    B = memory.shape[0]

    # Start with SOS token
    SOS_IDX = 1
    EOS_IDX = 2

    ys = paddle.full([B, 1], SOS_IDX, dtype='int64')

    for i in range(max_len - 1):
        # This would require the decoder to be exported separately
        # For now, just return the memory
        pass

    return ys


def main():
    parser = argparse.ArgumentParser(description='Test HME model on an image')
    parser.add_argument('--image', '-i', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', '-m', type=str,
                        default='./inference/hme_ultralight/inference',
                        help='Path to exported model (without extension)')
    parser.add_argument('--dict', '-d', type=str,
                        default='./ppocr/utils/dict/latex_symbol_dict.txt',
                        help='Path to vocabulary dictionary')
    parser.add_argument('--checkpoint', '-c', type=str,
                        default=None,
                        help='Path to checkpoint (use dynamic model instead of exported)')
    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Load vocabulary
    if os.path.exists(args.dict):
        vocab = load_vocab(args.dict)
        print(f"Loaded vocabulary with {len(vocab)} tokens")
    else:
        print(f"Warning: Dictionary not found: {args.dict}")
        vocab = None

    # Preprocess image
    print(f"Loading image: {args.image}")
    img = preprocess_image(args.image)
    print(f"Preprocessed image shape: {img.shape}")

    # Load model
    if args.checkpoint:
        # Use dynamic model with checkpoint
        print(f"Loading checkpoint: {args.checkpoint}")
        from ppocr.modeling.architectures import build_model
        from ppocr.utils.save_load import load_model
        import yaml

        config_path = './configs/rec/hme_latex_ocr_ultralight.yml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Set the pretrained model path in config
        config['Global']['pretrained_model'] = args.checkpoint

        model = build_model(config['Architecture'])
        load_model(config, model, model_type='rec')
        model.eval()

        input_tensor = paddle.to_tensor(img)

        # Create mask (all ones = all valid)
        mask = paddle.ones([1, 1, img.shape[2], img.shape[3]], dtype='float32')

        # Autoregressive decoding
        SOS_IDX = 1
        EOS_IDX = 2
        max_len = 256

        # Start with SOS token
        decoded_tokens = [SOS_IDX]

        with paddle.no_grad():
            for step in range(max_len - 1):
                # Create label tensor with decoded tokens so far
                labels = paddle.to_tensor([decoded_tokens + [0] * (max_len - len(decoded_tokens))], dtype='int64')

                output = model((input_tensor, mask, labels))

                if isinstance(output, dict):
                    output = output.get('head_out', output)
                if isinstance(output, tuple):
                    output = output[0]

                # Get prediction for next token (at position len(decoded_tokens)-1)
                next_token_logits = output[0, len(decoded_tokens) - 1, :]
                next_token = int(paddle.argmax(next_token_logits).numpy())

                if next_token == EOS_IDX:
                    break

                decoded_tokens.append(next_token)

        print(f"Decoded {len(decoded_tokens)} tokens")

        # Skip SOS token for output
        preds = decoded_tokens[1:]

    else:
        # Use exported static model
        print(f"Loading exported model: {args.model}")
        if not os.path.exists(args.model + '.pdmodel'):
            print(f"Error: Model not found: {args.model}.pdmodel")
            print("Make sure you've exported the model first.")
            sys.exit(1)

        model = paddle.jit.load(args.model)

        input_tensor = paddle.to_tensor(img)

        with paddle.no_grad():
            output = model(input_tensor)

        print(f"Output shape: {output.shape}")
        print(f"Output type: {type(output)}")

        # The exported model returns encoder memory, not decoded tokens
        # Full decoding would require implementing the decode loop here
        print("\nNote: Exported model returns encoder features.")
        print("For full LaTeX output, use --checkpoint with the dynamic model.")
        return

    # Convert predictions to LaTeX
    if vocab:
        latex_tokens = []
        for idx in preds:
            if idx == 2:  # EOS
                break
            if idx == 0:  # PAD
                continue
            if idx == 1:  # SOS
                continue
            token = vocab.get(idx, f'<unk:{idx}>')
            latex_tokens.append(token)

        latex = ' '.join(latex_tokens)
        print(f"\nPredicted LaTeX: {latex}")
    else:
        print(f"\nPredicted token indices: {preds[:50]}...")  # First 50


if __name__ == '__main__':
    main()
