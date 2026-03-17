"""
inference.py
------------
Load best checkpoint and run visual inference on the test/validation set.

Usage:
    python src/inference.py --checkpoint checkpoints/best_model.tar
"""

import argparse

import torch
import torch.nn as nn
import wandb

from config import DatasetConfig, InferenceConfig, ModelConfig, TrainingConfig
from dataset import get_dataloader
from model import get_model
from utils import denormalize, display_predictions, get_default_device, num_to_rgb, ID2COLOR_VIZ


def parse_args():
    parser = argparse.ArgumentParser(description="Retinal vessel segmentation inference")
    parser.add_argument("--checkpoint", type=str, default=InferenceConfig.CHECKPOINT_PATH,
                        help="Path to the model checkpoint (.tar file)")
    parser.add_argument("--num_batches", type=int, default=InferenceConfig.NUM_BATCHES,
                        help="Number of batches to visualise")
    return parser.parse_args()


@torch.inference_mode()
def run_inference(model, loader, device, img_size, num_batches=2):
    """
    Generate and display predictions for *num_batches* validation batches.

    Args:
        model:       Trained SegFormer model.
        loader:      DataLoader yielding (images, masks) pairs.
        device:      Torch device.
        img_size:    Target output spatial size (H, W).
        num_batches: Number of batches to process before stopping.
    """
    for idx, (batch_img, batch_mask) in enumerate(loader):
        if idx >= num_batches:
            break

        # Forward pass + upsample
        outputs       = model(pixel_values=batch_img.to(device), return_dict=True)
        upsampled     = nn.functional.interpolate(
            outputs["logits"], size=img_size, mode="bilinear", align_corners=False
        )
        pred_masks    = upsampled.argmax(dim=1).cpu().numpy()

        # Denormalise images for display
        images_np     = denormalize(batch_img.cpu()).permute(0, 2, 3, 1).numpy()
        gt_masks_np   = batch_mask.numpy()

        display_predictions(images_np, gt_masks_np, pred_masks, color_map=ID2COLOR_VIZ)


def main():
    args  = parse_args()
    device, gpu_available = get_default_device()

    # Minimal config dict required by model factory and data loader
    cfg = {
        "MODEL_NAME":  ModelConfig.MODEL_NAME,
        "NUM_CLASSES": DatasetConfig.NUM_CLASSES,
        "BATCH_SIZE":  InferenceConfig.BATCH_SIZE,
        "IMG_SIZE":    (DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH),
    }

    # Load model
    model = get_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    print(f"✓ Loaded checkpoint from {args.checkpoint}")

    # Load data
    _, valid_loader = get_dataloader(
        configs=cfg,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=gpu_available,
        shuffle_validation=True,
    )

    run_inference(model, valid_loader, device, cfg["IMG_SIZE"], num_batches=args.num_batches)


if __name__ == "__main__":
    main()
