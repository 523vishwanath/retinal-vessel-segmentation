"""
evaluate.py
-----------
Evaluate a trained SegFormer checkpoint on the test set and report mIoU + accuracy.

Usage:
    python src/evaluate.py --checkpoint checkpoints/best_model.tar
"""

import argparse
import torch
from config import DatasetConfig, InferenceConfig, ModelConfig, TrainingConfig
from dataset import get_dataloader
from model import get_model
from utils import get_default_device
from train import evaluate  # Reuse the evaluate loop from train.py


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=InferenceConfig.CHECKPOINT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    device, gpu_available = get_default_device()

    cfg = {
        "MODEL_NAME":  ModelConfig.MODEL_NAME,
        "NUM_CLASSES": DatasetConfig.NUM_CLASSES,
        "BATCH_SIZE":  InferenceConfig.BATCH_SIZE,
        "IMG_SIZE":    (DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH),
    }

    model = get_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    _, valid_loader = get_dataloader(
        configs=cfg,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=gpu_available,
    )

    loss, iou, acc = evaluate(model, valid_loader, device, cfg["NUM_CLASSES"], 1, 1, split="Test")
    print(f"\n📊 Test Results → Loss: {loss:.4f}  |  mIoU: {iou * 100:.2f}%  |  Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
