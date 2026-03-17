"""
metrics.py
----------
Mean Intersection-over-Union (mIoU) for multi-class semantic segmentation.

mIoU is the de-facto evaluation metric for segmentation tasks — it measures
the average overlap between predicted and ground-truth masks per class.
"""

import torch
import torch.nn.functional as F


def mean_iou(
    predictions: torch.Tensor,
    ground_truths: torch.Tensor,
    num_classes: int = 2,
    dims: tuple = (1, 2),
) -> torch.Tensor:
    """
    Compute mean IoU over a batch.

    Args:
        predictions:   Predicted class indices of shape (B, H, W).
        ground_truths: Ground-truth class indices of shape (B, H, W).
        num_classes:   Total number of classes.
        dims:          Spatial dims (H, W) in the one-hot encoded tensor.
    Returns:
        Scalar mean IoU across the batch.
    """
    # One-hot encode → (B, H, W, num_classes)
    gt_oh   = F.one_hot(ground_truths, num_classes=num_classes)
    pred_oh = F.one_hot(predictions,   num_classes=num_classes)

    # |G ∩ P| and |G ∪ P| → (B, num_classes)
    intersection = (pred_oh * gt_oh).sum(dim=dims)
    union        = (pred_oh.sum(dim=dims) + gt_oh.sum(dim=dims)) - intersection

    # IoU per class; NaN (empty class) → 0
    iou = torch.nan_to_num(intersection / union, nan=0.0)

    # Average only over classes present in this sample
    num_present = torch.count_nonzero(pred_oh.sum(dim=dims) + gt_oh.sum(dim=dims), dim=1)
    iou_per_img = iou.sum(dim=1) / num_present

    return iou_per_img.mean()
