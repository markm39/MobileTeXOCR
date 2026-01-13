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
Loss functions for HME (Handwritten Mathematical Expression) Recognition.

Based on:
- CoMER/TAMER: Cross-entropy loss for sequence prediction
- TAMER: Tree structure loss for bracket matching
- CAN: Counting loss as auxiliary task

Combined loss:
    L = L_symbol + λ1 * L_struct + λ2 * L_counting
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ["HMELoss", "HMELossV2"]


class SymbolCrossEntropyLoss(nn.Layer):
    """
    Cross-entropy loss for symbol prediction.
    Handles bidirectional sequences (L2R and R2L concatenated).
    """

    def __init__(self, ignore_index=0, label_smoothing=0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, pred, target, vocab_size=None):
        """
        Args:
            pred: [B, L, vocab_size] - model predictions
            target: [B, L] - ground truth indices
            vocab_size: int - vocabulary size for clamping
            
        Returns:
            loss: scalar
        """
        # Ensure target is int64 and clamp to valid range
        target = target.astype('int64')
        if vocab_size is not None:
            target = paddle.clip(target, min=0, max=vocab_size - 1)
        
        # Reshape for cross entropy
        pred = pred.reshape([-1, pred.shape[-1]])  # [B*L, vocab_size]
        target = target.reshape([-1])  # [B*L]

        # Create mask for non-padding positions
        mask = (target != self.ignore_index).astype('float32')

        # Compute cross entropy with label smoothing
        if self.label_smoothing > 0:
            vocab_size_actual = pred.shape[-1]
            log_probs = F.log_softmax(pred, axis=-1)

            # Smooth targets
            smooth_target = paddle.full_like(log_probs, self.label_smoothing / (vocab_size_actual - 1))
            target_one_hot = F.one_hot(target, vocab_size_actual).astype('float32')
            smooth_target = smooth_target * (1 - target_one_hot) + target_one_hot * (1 - self.label_smoothing)

            loss = -paddle.sum(smooth_target * log_probs, axis=-1)
        else:
            loss = F.cross_entropy(pred, target, reduction='none')

        # Apply mask and average
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss


class TreeStructureLoss(nn.Layer):
    """
    Tree structure loss from TAMER.
    Trains the model to predict parent-child relationships in the parse tree.
    
    For each token, predicts which previous token is its structural parent.
    This helps with bracket matching and nested structure recognition.
    """

    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, sim, struct_target):
        """
        Args:
            sim: [2B, L, L] - structure similarity predictions
            struct_target: [2B, L] - ground truth parent indices for each position
            
        Returns:
            loss: scalar
        """
        B2, L, _ = sim.shape

        # Reshape similarity to [2B*L, L] for cross entropy
        sim = sim.reshape([-1, L])  # [2B*L, L]
        struct_target = struct_target.reshape([-1])  # [2B*L]

        # Create mask for valid positions (not ignore_index)
        mask = (struct_target != self.ignore_index).astype('float32')

        # Replace ignore_index with 0 for cross_entropy (will be masked out)
        struct_target = paddle.where(
            struct_target == self.ignore_index,
            paddle.zeros_like(struct_target),
            struct_target,
        )

        # Cross entropy loss
        loss = F.cross_entropy(sim, struct_target, reduction='none')

        # Apply mask and average
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss


class CountingLoss(nn.Layer):
    """
    Counting loss from CAN (Counting-Aware Network).
    Predicts the frequency of each symbol as an auxiliary task.
    Helps prevent repeated/missing symbols.
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, counting_pred, counting_target):
        """
        Args:
            counting_pred: [B, vocab_size] - predicted symbol counts
            counting_target: [B, vocab_size] - ground truth symbol counts
            
        Returns:
            loss: scalar
        """
        loss = F.l1_loss(counting_pred, counting_target, reduction=self.reduction)
        return loss


class HMELoss(nn.Layer):
    """
    Combined loss for HME recognition.
    
    L = L_symbol + λ_struct * L_struct + λ_count * L_counting
    
    Args:
        vocab_size: Vocabulary size for counting loss
        ignore_index: Padding index to ignore
        label_smoothing: Label smoothing factor for symbol loss
        struct_weight: Weight for tree structure loss (default: 1.0)
        counting_weight: Weight for counting loss (default: 0.0 = disabled)
        use_struct_loss: Whether to use tree structure loss
        use_counting_loss: Whether to use counting loss
    """

    def __init__(
        self,
        vocab_size=113,
        ignore_index=0,
        label_smoothing=0.0,
        struct_weight=1.0,
        counting_weight=0.1,
        use_struct_loss=True,
        use_counting_loss=False,
        **kwargs,
    ):
        super().__init__()

        self.symbol_loss = SymbolCrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

        self.use_struct_loss = use_struct_loss
        self.struct_weight = struct_weight
        if use_struct_loss:
            self.struct_loss = TreeStructureLoss(ignore_index=-1)

        self.use_counting_loss = use_counting_loss
        self.counting_weight = counting_weight
        if use_counting_loss:
            self.counting_loss = CountingLoss()

        self.vocab_size = vocab_size

    def _compute_counting_target(self, labels, ignore_index=0):
        """
        Compute ground truth symbol counts from target labels.
        
        Args:
            labels: [B, L] - target sequences (only use L2R direction)
            ignore_index: padding index
            
        Returns:
            counts: [B, vocab_size]
        """
        B, L = labels.shape
        counts = paddle.zeros([B, self.vocab_size], dtype='float32')

        for i in range(B):
            for j in range(L):
                idx = labels[i, j].item()
                if idx != ignore_index:
                    counts[i, idx] += 1

        return counts

    def forward(self, predicts, batch):
        """
        Args:
            predicts: Model outputs
                - If tuple: (logits, sim) where sim is structure similarity
                - If tensor: logits only
            batch: Tuple of (images, image_masks, labels, label_masks) from DyMaskCollator
                - images: [B, C, H, W] (not used in loss)
                - image_masks: [B, 1, H, W] (not used in loss)
                - labels: [B, L] target sequences
                - label_masks: [B, L] mask for valid label positions
                
        Returns:
            dict: Loss dictionary with 'loss' and component losses
        """
        # Unpack predictions
        if isinstance(predicts, tuple):
            logits, sim = predicts
        else:
            logits = predicts
            sim = None

        # Get labels from batch tuple (DyMaskCollator format)
        # batch = (images, image_masks, labels, label_masks)
        labels = batch[2]  # [B, L]
        label_masks = batch[3]  # [B, L]

        # Symbol cross-entropy loss (pass vocab_size for clamping)
        loss_symbol = self.symbol_loss(logits, labels, vocab_size=self.vocab_size)
        total_loss = loss_symbol

        loss_dict = {
            'loss_symbol': loss_symbol,
        }

        # Tree structure loss
        if self.use_struct_loss and sim is not None:
            # If no struct_label provided, compute from labels
            # Default: each token's parent is the previous token
            struct_target = paddle.arange(labels.shape[1], dtype='int64')
            struct_target = struct_target.unsqueeze(0).expand([labels.shape[0], -1])
            struct_target = struct_target - 1  # Parent is previous token
            struct_target = paddle.clip(struct_target, min=0)
            # Mask padding positions with -1
            struct_target = paddle.where(
                labels == 0,
                paddle.full_like(struct_target, -1),
                struct_target,
            )

            loss_struct = self.struct_loss(sim, struct_target)
            total_loss = total_loss + self.struct_weight * loss_struct
            loss_dict['loss_struct'] = loss_struct

        # Counting loss (optional) - disabled by default
        if self.use_counting_loss:
            # Compute counting target from labels
            counting_target = self._compute_counting_target(labels)
            # Note: counting_pred should come from model output, not batch
            # For now, this is disabled unless model provides counting predictions
            pass

        loss_dict['loss'] = total_loss
        return loss_dict


class HMELossV2(nn.Layer):
    """
    Loss for HMEHeadV2 with proper shifted labels.

    Expects:
        - predicts: dict with 'logits' [B, L, vocab_size] and 'aux_loss'
        - batch: (images, image_masks, decoder_inputs, decoder_targets, label_masks)

    Uses -100 as ignore_index for padding (standard PyTorch/Paddle convention).
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        vocab_size=113,
        label_smoothing=0.0,
        aux_loss_weight=0.01,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.aux_loss_weight = aux_loss_weight

    def forward(self, predicts, batch):
        """
        Args:
            predicts: dict with:
                - 'logits': [B, L, vocab_size]
                - 'aux_loss': scalar (MoE auxiliary loss)
            batch: tuple from DyMaskCollatorV2:
                - images: [B, C, H, W]
                - image_masks: [B, 1, H, W]
                - decoder_inputs: [B, L]
                - decoder_targets: [B, L]
                - label_masks: [B, L]

        Returns:
            dict with 'loss', 'loss_ce', 'loss_aux'
        """
        # Handle dict output from HMEHeadV2
        if isinstance(predicts, dict):
            logits = predicts['logits']
            aux_loss = predicts.get('aux_loss', 0.0)
        else:
            # Fallback for tuple output
            logits = predicts[0] if isinstance(predicts, tuple) else predicts
            aux_loss = 0.0

        # Get targets from batch (DyMaskCollatorV2 format)
        # batch = (images, image_masks, decoder_inputs, decoder_targets, label_masks)
        decoder_targets = batch[3]  # [B, L_target]
        label_masks = batch[4]  # [B, L_target]

        # Handle shape mismatch between model output and targets
        # Model outputs [B, max_len, V] but collator pads to [B, max_in_batch]
        B, L_logits, V = logits.shape
        L_target = decoder_targets.shape[1]

        if L_logits > L_target:
            # Truncate logits to match target length
            logits = logits[:, :L_target, :]
        elif L_target > L_logits:
            # Truncate targets to match logits length
            decoder_targets = decoder_targets[:, :L_logits]
            label_masks = label_masks[:, :L_logits]

        # Flatten for cross entropy
        B, L, V = logits.shape
        logits_flat = logits.reshape([-1, V])  # [B*L, V]
        targets_flat = decoder_targets.reshape([-1])  # [B*L]

        # Create mask: valid positions where target != IGNORE_INDEX
        mask = (targets_flat != self.IGNORE_INDEX).astype('float32')

        # Replace IGNORE_INDEX with 0 for cross_entropy computation
        # (will be masked out anyway)
        targets_safe = paddle.where(
            targets_flat == self.IGNORE_INDEX,
            paddle.zeros_like(targets_flat),
            targets_flat
        )

        # Clamp to valid vocab range
        targets_safe = paddle.clip(targets_safe, min=0, max=self.vocab_size - 1)

        # Compute cross entropy loss
        if self.label_smoothing > 0:
            log_probs = F.log_softmax(logits_flat, axis=-1)
            smooth_target = paddle.full_like(log_probs, self.label_smoothing / (V - 1))
            target_one_hot = F.one_hot(targets_safe, V).astype('float32')
            smooth_target = smooth_target * (1 - target_one_hot) + target_one_hot * (1 - self.label_smoothing)
            loss_ce = -paddle.sum(smooth_target * log_probs, axis=-1)
        else:
            loss_ce = F.cross_entropy(logits_flat, targets_safe, reduction='none')

        # Apply mask and average
        loss_ce = (loss_ce * mask).sum() / (mask.sum() + 1e-8)

        # Total loss with auxiliary MoE loss
        # Ensure aux_loss is a tensor for consistent return type
        if isinstance(aux_loss, (int, float)):
            aux_loss = paddle.to_tensor(aux_loss, dtype='float32')

        # NaN protection: if aux_loss is NaN, replace with 0
        if paddle.isnan(aux_loss).any():
            aux_loss = paddle.to_tensor(0.0, dtype='float32')

        total_loss = loss_ce + self.aux_loss_weight * aux_loss

        return {
            'loss': total_loss,
            'loss_ce': loss_ce,
            'loss_aux': aux_loss,
        }

