"""
utils.py
--------
Utility functions: seeding, denormalisation, colour conversion, and visualisation helpers.
"""

import random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from config import ID2COLOR_VIZ, DatasetConfig


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device Helpers
# ---------------------------------------------------------------------------

def get_default_device():
    """Return (torch.device, gpu_available: bool)."""
    gpu_available = torch.cuda.is_available()
    return torch.device("cuda" if gpu_available else "cpu"), gpu_available


# ---------------------------------------------------------------------------
# Image Helpers
# ---------------------------------------------------------------------------

def denormalize(tensors, mean=DatasetConfig.MEAN, std=DatasetConfig.STD):
    """
    Reverse ImageNet normalisation in-place on a batch of CHW tensors.

    Args:
        tensors: FloatTensor of shape (B, C, H, W)
    Returns:
        Clamped FloatTensor in [0, 1]
    """
    for c in range(3):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0.0, 1.0)


def rgb_to_grayscale(rgb_arr: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    Convert an RGB segmentation mask to a single-channel binary mask.

    Pixels with mean intensity >= threshold are treated as foreground (class 1).

    Args:
        rgb_arr:   H×W×C or H×W numpy array.
        threshold: Intensity cutoff separating background from foreground.
    Returns:
        H×W uint8 array with values in {0, 1}.
    """
    gray = np.mean(rgb_arr, axis=2) if rgb_arr.ndim == 3 else rgb_arr
    return (gray >= threshold).astype(np.uint8)


def num_to_rgb(num_arr: np.ndarray, color_map: dict = ID2COLOR_VIZ) -> np.ndarray:
    """
    Convert a single-channel class-ID mask to an RGB image.

    Args:
        num_arr:   H×W array of integer class IDs.
        color_map: Dict mapping class_id → (R, G, B) tuple.
    Returns:
        H×W×3 float32 array in [0.0, 1.0].
    """
    single_layer = np.squeeze(num_arr)
    output = np.zeros(num_arr.shape[:2] + (3,))
    for k, color in color_map.items():
        output[single_layer == k] = color
    return np.float32(output) / 255.0


def image_overlay(image: np.ndarray, segmented_image: np.ndarray) -> np.ndarray:
    """
    Blend original fundus image with its predicted segmentation mask.

    Args:
        image:           H×W×3 float32 array in [0, 1].
        segmented_image: H×W×3 float32 RGB segmentation mask.
    Returns:
        Blended H×W×3 float32 array clipped to [0, 1].
    """
    alpha, beta, gamma = 1.0, 0.7, 0.0
    seg_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(img_bgr, alpha, seg_bgr, beta, gamma)
    return np.clip(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def display_predictions(images, gt_masks, pred_masks, color_map=ID2COLOR_VIZ):
    """
    Render a 4-panel comparison for each sample in a batch:
    Original | Ground Truth | Predicted Mask | Overlay

    Args:
        images:     (B, H, W, 3) float32 denormalised numpy array.
        gt_masks:   (B, H, W) integer ground-truth masks.
        pred_masks: (B, H, W) integer predicted masks.
        color_map:  Class-to-RGB colour mapping for visualisation.
    """
    for i in range(len(images)):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = ["Actual Frame", "Ground Truth", "Predicted", "Overlay"]

        pred_rgb = num_to_rgb(pred_masks[i], color_map=color_map)
        gt_rgb   = num_to_rgb(gt_masks[i],   color_map=color_map)
        overlay  = image_overlay(images[i], pred_rgb)

        for ax, img, title in zip(
            axes,
            [images[i], gt_rgb, pred_rgb, overlay],
            titles,
        ):
            ax.imshow(img if img.ndim == 3 else img, cmap="gray")
            ax.set_title(title, fontsize=12)
            ax.axis("off")

        plt.tight_layout()
        plt.show()
