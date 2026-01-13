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

        # Handle dict output from HMEHeadV2
        if isinstance(preds, dict):
            logits = preds['logits']
        else:
            logits = preds

        # Get targets from batch (DyMaskCollatorV2 format)
        # batch = (images, image_masks, decoder_inputs, decoder_targets, label_masks)
        decoder_targets = batch[3]  # [B, L]
        label_masks = batch[4]  # [B, L]

        # Get predictions
        word_pred = logits.argmax(axis=2)  # [B, L]

        # Convert to numpy
        if hasattr(word_pred, 'cpu'):
            word_pred = word_pred.cpu().detach().numpy()
        elif hasattr(word_pred, 'numpy'):
            word_pred = word_pred.numpy()

        if hasattr(decoder_targets, 'cpu'):
            word_label = decoder_targets.cpu().detach().numpy()
        elif hasattr(decoder_targets, 'numpy'):
            word_label = decoder_targets.numpy()
        else:
            word_label = decoder_targets

        if hasattr(label_masks, 'cpu'):
            word_label_mask = label_masks.cpu().detach().numpy()
        elif hasattr(label_masks, 'numpy'):
            word_label_mask = label_masks.numpy()
        else:
            word_label_mask = label_masks

        word_scores = []
        line_right = 0
        batch_size = word_label.shape[0]

        for i in range(batch_size):
            target = word_label[i]
            pred = word_pred[i]
            mask = word_label_mask[i]

            # Get sequence length from mask
            seq_len = int(np.sum(mask))

            if seq_len == 0:
                # Empty sequence
                word_scores.append(1.0)
                line_right += 1
                continue

            # Get valid portions
            target_seq = target[:seq_len]
            pred_seq = pred[:seq_len]

            # Compute token-level accuracy
            correct = np.sum(target_seq == pred_seq)
            word_scores.append(correct / seq_len)

            # Check exact match (truncate at first EOS in prediction)
            # Find first EOS in prediction
            eos_positions = np.where(pred_seq == self.EOS_IDX)[0]
            if len(eos_positions) > 0:
                pred_end = eos_positions[0]
            else:
                pred_end = seq_len

            # Find first EOS in target
            target_eos_positions = np.where(target_seq == self.EOS_IDX)[0]
            if len(target_eos_positions) > 0:
                target_end = target_eos_positions[0] + 1  # Include EOS
            else:
                target_end = seq_len

            # Compare sequences
            pred_final = pred[:pred_end + 1] if pred_end < seq_len else pred[:seq_len]
            target_final = target[:target_end]

            if np.array_equal(pred_final, target_final):
                line_right += 1

        self.word_rate = np.mean(word_scores)
        self.exp_rate = line_right / batch_size

        exp_length = batch_size
        word_length = word_label.shape[1]

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
