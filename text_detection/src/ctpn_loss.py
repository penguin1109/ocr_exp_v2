import os
import sys

sys.path.append(os.getcwd())

import torch

from torch import Tensor
from typing import Tuple, Optional
class BalancedL1Loss(torch.nn.Module):

    def __init__(self,
                 alpha: Optional[float] = 0.5,
                 gamma: Optional[float] = 1.5,
                 beta: Optional[float] = 1.0,
                 reduction: Optional[str] = "none"):
        r"""
        An implementation of the Balanced L1 Loss as described in:
        `Libra R-CNN: Towards Balanced Learning for Object Detection <https://arxiv.org/abs/1904.02701>`__.
        It is computed as follows:
        .. math::
            L_{b(x)} = \begin{cases}
                        \frac{a}{b}(b|x| + 1)ln(b|x| + 1) - \alpha|x|, & \text{if } |x| < 1 \\
                        \gamma|x| + C, & \text{otherwise }
                        \end{cases}
            \text{in which the parameters \gamma, \alpha, and b are constrained by}
            \alpha ln(b + 1) = \gamma
        Shape:
            - Input: :math:`(N, *)`.
            - Target: :math:`(N, *)`.
            where '*' means any number of dimensions.
        Examples::
            >>> inputs = torch.randn(1, 1, 5, 5, requires_grad=True)
            >>> targets = torch.rand(1, 1, 5, 5, dtype=torch.float32)
            >>> l1Loss = BalancedL1Loss(reduction="mean")
            >>> loss = l1Loss(inputs, targets)
            >>> loss.backward()
        Args:
            alpha (string, optional): A float number that either increases more gradients for inliers (accurate samples)
                if it is small or increases for outliers (negative samples). By default, it is set to 0.5.
            gamma (float, optional): A float number that is used to tune the upper bound of regression errors.
            beta (float, optional): A float number that represents a threshold at which to change the loss behavior .
            reduction (string, optional): Specifies the reduction to apply to the
                 output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                 'mean': the sum of the output will be divided by the number of elements
                 in the output, 'sum': the output will be summed.
                 Default: 'none'.
            
        """

        super(BalancedL1Loss, self).__init__()

        reductions = ("none", "mean", "sum")
        if reduction not in reductions:
            raise NotImplemented("This reduction operation '{0}' is not currently supported! "
                                 "Try one of these operations: {1}".format(reduction, reductions))

        if beta <= 0:
            raise ValueError("This value of beta '{0}' must be strictly positive".format(beta))

        self.beta: float = beta

        self.alpha: float = alpha

        self.gamma: float = gamma

        self.reduction: str = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:

        if sorted(list(inputs.size())) != sorted(list(targets.size())):
            raise ValueError("Input and target dimensions does not match!")

        if inputs.dtype != targets.dtype:
            raise ValueError("The inputs and targets must have the same data type!")

        valid_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float32, torch.float16)
        if inputs.dtype not in valid_types:
            raise ValueError("The input data type must be one of these following types: {0}.".format(valid_types))

        if targets.dtype not in valid_types:
            raise ValueError("The target data type must be one of these following types: {0}.".format(valid_types))

        if not inputs.device == targets.device:
            raise ValueError("The input and target must be in the same device. "
                             "Got: input device = {0} and target device = {1}.".format(inputs.device, targets.device))

        if targets.numel() == 0:
            return inputs.sum() * 0

        x = torch.abs(inputs - targets)

        # parameters γ, α, and b are constrained by: αln(b + 1) = γ.
        b = torch.exp(torch.as_tensor(self.gamma / self.alpha)) - 1.0

        loss = torch.where(
            x < self.beta,
            self.alpha / b * (b * x + 1) * torch.log(b * x / self.beta + 1) - self.alpha * x,
            self.gamma * x + self.gamma / b - self.alpha * self.beta
        )

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass

        return loss

class MultiBoxLoss(torch.nn.Module):

    def __init__(self, cfg):
        r"""
        An implementation of the multi box loss derived from:

        `Scalable Object Detection using Deep Neural Networks <https://arxiv.org/abs/1312.2249>`__.

        Args:
            configs (dict): The configuration file.

        """
        super(MultiBoxLoss, self).__init__()
        self.cfg = cfg
        self.lambda_reg: float = cfg.LOSS_LAMBDA_REG

        self.lambda_cls: float = cfg.LOSS_LAMBDA_CLS

        self.neg_pos_ratio: float = cfg.LOSS_NEG_POS_RATIO

        self.ignore_index: int = cfg.ANCHOR_IGNORE_LABEL

        self.positive_anchor_label: int = cfg.ANCHOR_POSITIVE_LABEL

        self.negative_anchor_label: int = cfg.ANCHOR_NEGATIVE_LABEL

        self.balancedL1Loss: torch.nn.Module = BalancedL1Loss(reduction="none")

        self.ceLoss: torch.nn.Module = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="none")

    def forward(self,pred_cls, pred_regr, targets) -> Tensor:
        # The predicted boxes and classifications whose shapes are respectively:
        # (batch_size, #anchors, 2) and (batch_size, #anchors, #classes)
        predicted_bboxes, predicted_classes = pred_regr, pred_cls
        
        # The encoded ground truth boxes and matching indicators whose shapes are respectively:
        # (batch_size, #anchors, 2) and (batch_size, #anchors).
        gt_bboxes, matching_indicators = targets

        # Identify anchors that are positives.
        positive_anchor_mask = matching_indicators == self.positive_anchor_label

        # Identify anchors that are negatives.
        negative_anchor_mask = matching_indicators == self.negative_anchor_label

        # ==================================================================================================
        # Localization loss = BalancedL1Loss(predicted_bboxes, gt_bboxes) is computed over positive anchors.
        # ==================================================================================================
        predicted_bboxes = predicted_bboxes.to(torch.float32)
        gt_bboxes = gt_bboxes.to(torch.float32)
        # Shape: (#matched_anchors, 2)
        balanced_l1_loss = self.balancedL1Loss(inputs=predicted_bboxes[positive_anchor_mask],
                                               targets=gt_bboxes[positive_anchor_mask])

        # As in the paper, 'Nv' is the total number of anchors used by the localization loss.
        Nv = balanced_l1_loss.size(0)

        localization_loss = (self.lambda_reg / Nv) * balanced_l1_loss.sum()

        # ===========================================================================================
        # Confidence loss = CrossEntropyLoss(predicted_classes, gt_classes) is computed over positive
        # and (hard) negative anchors.
        # ===========================================================================================

        # Useful variable.
        n_classes = predicted_classes.size(2)

        # The confidence loss over positive anchors.
        positive_confidence_loss = self.ceLoss(
            input=predicted_classes[positive_anchor_mask].contiguous().view(-1, n_classes),
            target=matching_indicators[positive_anchor_mask].contiguous().view(-1)
        )

        # The confidence loss over negative anchors.
        negative_confidence_loss = self.ceLoss(
            input=predicted_classes[negative_anchor_mask].contiguous().view(-1, n_classes),
            target=matching_indicators[negative_anchor_mask].contiguous().view(-1)
        )

        # ==========================================================================================================
        # Now, instead of using all the negative anchors, they are sorted using the highest (negative based anchors)
        # confidence loss and pick the top ones so that the ratio between the negative and positive ones
        # is at most 'neg_pos_ratio:1'.
        # ==========================================================================================================

        # The number of positive anchors.
        k_positive_anchors = positive_anchor_mask.long().sum()

        # The number of all negative anchors.
        all_negatives = negative_anchor_mask.long().sum()

        # The real number of negative anchors, i.e., the ones that respect the ratio mentioned above.
        k_negatives = k_positive_anchors * self.neg_pos_ratio

        # The real number of negative anchors, i.e., the ones that respect the ratio mentioned above.
        K = min(k_negatives, all_negatives)

        # Now we sort the negative anchors by using the highest confidence loss and pick the K-top ones.
        hard_negative_confidence_loss, _ = torch.topk(input=negative_confidence_loss,
                                                      k=int(K),
                                                      largest=True,
                                                      sorted=True)

        # As in the paper, 'Ns' is the total number of anchors used by the confidence loss.
        # That is to say, the number of positive and negative anchors.
        Ns = torch.count_nonzero(positive_confidence_loss) + torch.count_nonzero(hard_negative_confidence_loss)

        # The sum over positive anchors.
        # Shape: (batch_size,)
        cls_pos = positive_confidence_loss.sum()

        # The sum over negative anchors.
        # Shape: (batch_size,)
        cls_neg = hard_negative_confidence_loss.sum()

        # The confidence loss is the sum over positive and hard negatives anchors.
        confidence_loss = (self.lambda_cls / Ns) * (cls_pos + cls_neg)

        return localization_loss, confidence_loss
