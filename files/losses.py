"""
losses.py
---------
Combo loss: Dice coefficient loss + Cross-Entropy.

Using cross-entropy alone under-performs on the vessel segmentation task because
vessels occupy only ~10% of retinal pixels (severe class imbalance). The Dice
term directly optimises for spatial overlap, complementing CE's per-pixel focus.

    L_total = (1 - Dice_mean) + CE
"""

import torch
import torch.nn.functional as F


def dice_coef_loss(
    predictions: torch.Tensor,
    ground_truths: torch.Tensor,
    num_classes: int = 2,
    dims: tuple = (1, 2),
    smooth: float = 1e-8,
) -> torch.Tensor:
    """
    Dice + Cross-Entropy combo loss for semantic segmentation.

    Args:
        predictions:   Raw logits of shape (B, num_classes, H, W).
        ground_truths: Integer class masks of shape (B, H, W).
        num_classes:   Number of segmentation classes.
        dims:          Spatial dimensions to reduce over (H, W axes in B×H×W×C).
        smooth:        Numerical stability term added to numerator and denominator.
    Returns:
        Scalar combo loss.
    """
    # One-hot encode ground truths → (B, H, W, num_classes)
    gt_oh = F.one_hot(ground_truths, num_classes=num_classes)

    # Softmax + transpose predictions → (B, H, W, num_classes)
    pred_soft = F.softmax(predictions, dim=1).permute(0, 2, 3, 1)

    # Intersection |G ∩ P| and sum |G| + |P|, both shape (B, num_classes)
    intersection = (pred_soft * gt_oh).sum(dim=dims)
    summation    = pred_soft.sum(dim=dims) + gt_oh.sum(dim=dims)

    # Dice per class, averaged over batch and classes
    dice = (2.0 * intersection + smooth) / (summation + smooth)
    dice_mean = dice.mean()

    # Cross-entropy over logits
    ce = F.cross_entropy(predictions, ground_truths)

    return (1.0 - dice_mean) + ce
