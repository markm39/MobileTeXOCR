# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rapidfuzz.distance import Levenshtein
from difflib import SequenceMatcher

import numpy as np
import string
from .bleu import compute_bleu_score, compute_edit_distance


class RecMetric(object):
    def __init__(
        self, main_indicator="acc", is_filter=False, ignore_space=True, **kwargs
    ):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = "".join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text)
        )
        return text.lower()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        return {
            "acc": correct_num / (all_num + self.eps),
            "norm_edit_dis": 1 - norm_edit_dis / (all_num + self.eps),
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        self.reset()
        return {"acc": acc, "norm_edit_dis": norm_edit_dis}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0


class CNTMetric(object):
    def __init__(self, main_indicator="acc", **kwargs):
        self.main_indicator = main_indicator
        self.eps = 1e-5
        self.reset()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        for pred, target in zip(preds, labels):
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        return {
            "acc": correct_num / (all_num + self.eps),
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        self.reset()
        return {"acc": acc}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0


class CANMetric(object):
    def __init__(self, main_indicator="exp_rate", **kwargs):
        self.main_indicator = main_indicator
        self.word_right = []
        self.exp_right = []
        self.word_total_length = 0
        self.exp_total_num = 0
        self.word_rate = 0
        self.exp_rate = 0
        self.reset()
        self.epoch_reset()

    def __call__(self, preds, batch, **kwargs):
        for k, v in kwargs.items():
            epoch_reset = v
            if epoch_reset:
                self.epoch_reset()
        word_probs = preds
        word_label, word_label_mask = batch
        line_right = 0
        if word_probs is not None:
            word_pred = word_probs.argmax(2)
        word_pred = word_pred.cpu().detach().numpy()
        word_scores = []
        for s1, s2, s3 in zip(word_label, word_pred, word_label_mask):
            seq_len = int(np.sum(s3))
            s1_slice = s1[:seq_len] if seq_len > 0 else s1[:1]
            s2_slice = s2[:seq_len] if seq_len > 0 else s2[:1]
            
            # Handle edge cases to avoid division by zero
            if len(s1_slice) == 0:
                word_scores.append(1.0)  # Empty label = perfect match
            else:
                ratio = SequenceMatcher(None, s1_slice, s2_slice, autojunk=False).ratio()
                score = ratio * (len(s1_slice) + len(s2_slice)) / len(s1_slice) / 2
                word_scores.append(score)
        batch_size = len(word_scores)
        for i in range(batch_size):
            if word_scores[i] == 1:
                line_right += 1
        self.word_rate = np.mean(word_scores)  # float
        self.exp_rate = line_right / batch_size  # float
        exp_length, word_length = word_label.shape[:2]
        self.word_right.append(self.word_rate * word_length)
        self.exp_right.append(self.exp_rate * exp_length)
        self.word_total_length = self.word_total_length + word_length
        self.exp_total_num = self.exp_total_num + exp_length

    def get_metric(self):
        """
        return {
            'word_rate': 0,
            "exp_rate": 0,
        }
        """
        cur_word_rate = sum(self.word_right) / self.word_total_length
        cur_exp_rate = sum(self.exp_right) / self.exp_total_num
        self.reset()
        return {"word_rate": cur_word_rate, "exp_rate": cur_exp_rate}

    def reset(self):
        self.word_rate = 0
        self.exp_rate = 0

    def epoch_reset(self):
        self.word_right = []
        self.exp_right = []
        self.word_total_length = 0
        self.exp_total_num = 0


class LaTeXOCRMetric(object):
    def __init__(self, main_indicator="exp_rate", cal_bleu_score=False, **kwargs):
        self.main_indicator = main_indicator
        self.cal_bleu_score = cal_bleu_score
        self.edit_right = []
        self.exp_right = []
        self.bleu_right = []
        self.e1_right = []
        self.e2_right = []
        self.e3_right = []
        self.exp_total_num = 0
        self.edit_dist = 0
        self.exp_rate = 0
        if self.cal_bleu_score:
            self.bleu_score = 0
        self.e1 = 0
        self.e2 = 0
        self.e3 = 0
        self.reset()
        self.epoch_reset()

    def __call__(self, preds, batch, **kwargs):
        for k, v in kwargs.items():
            epoch_reset = v
            if epoch_reset:
                self.epoch_reset()
        word_pred = preds
        word_label = batch
        line_right, e1, e2, e3 = 0, 0, 0, 0
        bleu_list, lev_dist = [], []
        for labels, prediction in zip(word_label, word_pred):
            if prediction == labels:
                line_right += 1
            distance = compute_edit_distance(prediction, labels)
            bleu_list.append(compute_bleu_score([prediction], [labels]))
            lev_dist.append(Levenshtein.normalized_distance(prediction, labels))
            if distance <= 1:
                e1 += 1
            if distance <= 2:
                e2 += 1
            if distance <= 3:
                e3 += 1

        batch_size = len(lev_dist)

        self.edit_dist = sum(lev_dist)  # float
        self.exp_rate = line_right  # float
        if self.cal_bleu_score:
            self.bleu_score = sum(bleu_list)
            self.bleu_right.append(self.bleu_score)
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        exp_length = len(word_label)
        self.edit_right.append(self.edit_dist)
        self.exp_right.append(self.exp_rate)
        self.e1_right.append(self.e1)
        self.e2_right.append(self.e2)
        self.e3_right.append(self.e3)
        self.exp_total_num = self.exp_total_num + exp_length

    def get_metric(self):
        """
        return {
            'edit distance': 0,
            "bleu_score": 0,
            "exp_rate": 0,
        }
        """
        cur_edit_distance = sum(self.edit_right) / self.exp_total_num
        cur_exp_rate = sum(self.exp_right) / self.exp_total_num
        if self.cal_bleu_score:
            cur_bleu_score = sum(self.bleu_right) / self.exp_total_num
        cur_exp_1 = sum(self.e1_right) / self.exp_total_num
        cur_exp_2 = sum(self.e2_right) / self.exp_total_num
        cur_exp_3 = sum(self.e3_right) / self.exp_total_num
        self.reset()
        if self.cal_bleu_score:
            return {
                "bleu_score": cur_bleu_score,
                "edit distance": cur_edit_distance,
                "exp_rate": cur_exp_rate,
                "exp_rate<=1 ": cur_exp_1,
                "exp_rate<=2 ": cur_exp_2,
                "exp_rate<=3 ": cur_exp_3,
            }
        else:

            return {
                "edit distance": cur_edit_distance,
                "exp_rate": cur_exp_rate,
                "exp_rate<=1 ": cur_exp_1,
                "exp_rate<=2 ": cur_exp_2,
                "exp_rate<=3 ": cur_exp_3,
            }

    def reset(self):
        self.edit_dist = 0
        self.exp_rate = 0
        if self.cal_bleu_score:
            self.bleu_score = 0
        self.e1 = 0
        self.e2 = 0
        self.e3 = 0

    def epoch_reset(self):
        self.edit_right = []
        self.exp_right = []
        if self.cal_bleu_score:
            self.bleu_right = []
        self.e1_right = []
        self.e2_right = []
        self.e3_right = []
        self.editdistance_total_length = 0
        self.exp_total_num = 0


class CANMetricV2(object):
    """
    Metric for HMEHeadV2 with shifted labels.

    Expects:
        - preds: dict with 'logits' [B, L, vocab_size]
        - batch: (images, image_masks, decoder_inputs, decoder_targets, label_masks)

    Computes:
        - word_rate: Average token-level accuracy
        - exp_rate: Expression-level accuracy (exact match)
    """

    IGNORE_INDEX = -100
    EOS_IDX = 0

    def __init__(self, main_indicator="exp_rate", **kwargs):
        self.main_indicator = main_indicator
        self.word_right = []
        self.exp_right = []
        self.word_total_length = 0
        self.exp_total_num = 0
        self.word_rate = 0
        self.exp_rate = 0
        self.reset()
        self.epoch_reset()

    def __call__(self, preds, batch, **kwargs):
        for k, v in kwargs.items():
            epoch_reset = v
            if epoch_reset:
                self.epoch_reset()

        # Handle two possible input formats:
        # 1. preds = raw logits [B, L, V], batch = batch[2:] = (decoder_inputs, decoder_targets, label_masks)
        # 2. preds = post-processed [(text, probs), ...], batch = full batch

        # Determine format based on batch length
        if len(batch) == 3:
            # Sliced batch: (decoder_inputs, decoder_targets, label_masks)
            decoder_targets = batch[1]
            label_masks = batch[2]
        elif len(batch) >= 5:
            # Full batch: (images, image_masks, decoder_inputs, decoder_targets, label_masks)
            decoder_targets = batch[3]
            label_masks = batch[4]
        else:
            # Fallback - assume last two are targets and masks
            decoder_targets = batch[-2]
            label_masks = batch[-1]

        # Load vocabulary for decoding
        if not hasattr(self, '_vocab'):
            self._vocab = ['<eos>', '<sos>']
            try:
                import os
                dict_path = 'ppocr/utils/dict/latex_symbol_dict.txt'
                if os.path.exists(dict_path):
                    with open(dict_path, 'r') as f:
                        for line in f:
                            self._vocab.append(line.strip())
            except:
                pass

        # Convert paddle tensor to numpy if needed
        if hasattr(preds, 'numpy'):
            preds = preds.numpy()

        # Handle raw logits vs post-processed text
        # Raw logits: ndarray with shape [B, L, V] or [B, L]
        # Post-processed: list of [(text, probs), ...]
        if hasattr(preds, 'shape') and len(preds.shape) >= 2:
            # Raw logits - decode to token indices
            if len(preds.shape) == 3:
                pred_indices = preds.argmax(axis=2)  # [B, L]
            else:
                pred_indices = preds  # Already token indices [B, L]

            # Decode predictions to text
            decoded_preds = []
            for b in range(pred_indices.shape[0]):
                symbols = []
                for idx in pred_indices[b]:
                    idx = int(idx)
                    if idx == 0:  # EOS
                        break
                    if idx == 1:  # SOS
                        continue
                    if idx < len(self._vocab):
                        symbols.append(self._vocab[idx])
                decoded_preds.append(' '.join(symbols))
            preds = decoded_preds

        word_scores = []
        line_right = 0
        batch_size = len(preds)

        for i in range(batch_size):
            # Get predicted text
            if isinstance(preds[i], (list, tuple)):
                pred_text = preds[i][0]
            else:
                pred_text = str(preds[i])

            # Decode target tokens to text
            target_tokens = decoder_targets[i]
            mask = label_masks[i]
            seq_len = int(np.sum(mask)) if hasattr(mask, '__len__') else len(target_tokens)

            # Convert target indices to text
            target_symbols = []
            for j in range(seq_len):
                idx = int(target_tokens[j])
                if idx == 0:  # EOS
                    break
                if idx == 1:  # SOS - skip
                    continue
                if idx < len(self._vocab):
                    target_symbols.append(self._vocab[idx])
            target_text = ' '.join(target_symbols)

            # Compare predicted text with target text
            if len(target_text) == 0 and len(pred_text) == 0:
                # Both empty = match
                word_scores.append(1.0)
                line_right += 1
            elif len(target_text) == 0 or len(pred_text) == 0:
                # One empty, one not = no match
                word_scores.append(0.0)
            else:
                # Compute similarity using SequenceMatcher
                ratio = SequenceMatcher(None, target_text, pred_text, autojunk=False).ratio()
                word_scores.append(ratio)

                # Check exact match
                if pred_text.strip() == target_text.strip():
                    line_right += 1

        self.word_rate = np.mean(word_scores) if word_scores else 0.0
        self.exp_rate = line_right / batch_size if batch_size > 0 else 0.0

        exp_length = batch_size
        word_length = batch_size  # Use batch_size as proxy for word_length

        self.word_right.append(self.word_rate * word_length)
        self.exp_right.append(self.exp_rate * exp_length)
        self.word_total_length = self.word_total_length + word_length
        self.exp_total_num = self.exp_total_num + exp_length

    def get_metric(self):
        """
        return {
            'word_rate': 0,
            "exp_rate": 0,
        }
        """
        if self.word_total_length == 0:
            return {"word_rate": 0.0, "exp_rate": 0.0}

        cur_word_rate = sum(self.word_right) / self.word_total_length
        cur_exp_rate = sum(self.exp_right) / self.exp_total_num
        self.reset()
        return {"word_rate": cur_word_rate, "exp_rate": cur_exp_rate}

    def reset(self):
        self.word_rate = 0
        self.exp_rate = 0

    def epoch_reset(self):
        self.word_right = []
        self.exp_right = []
        self.word_total_length = 0
        self.exp_total_num = 0
