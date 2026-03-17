"""
dataset.py
----------
PyTorch Dataset and DataLoader factory for the ICPR retinal vessel segmentation task.

Key design:
  - Albumentations handles all augmentations to guarantee joint image+mask transforms.
  - HuggingFace SegformerImageProcessor manages resizing & formatting.
  - rgb_to_grayscale collapses RGB masks (white vessel on black) into class-ID masks.
"""

import PIL
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor

from config import DatasetConfig, TrainingConfig
from utils import rgb_to_grayscale


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CustomSegDataset(Dataset):
    """
    Retinal vessel segmentation dataset compatible with SegFormer.

    Args:
        image_paths: List of paths to fundus images.
        mask_paths:  List of paths to binary segmentation masks (None for test-only).
        is_train:    If True, geometric augmentations are applied.
        num_classes: Number of segmentation classes (default 2).
        processor:   SegformerImageProcessor instance for resizing/formatting.
    """

    def __init__(
        self,
        *,
        image_paths: list,
        mask_paths: list = None,
        is_train: bool = False,
        num_classes: int = 2,
        processor: SegformerImageProcessor,
    ):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.num_classes = num_classes
        self.processor   = processor
        self.is_train    = is_train
        self.transforms  = self._build_transforms()

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def _build_transforms(self) -> A.Compose:
        """Construct the Albumentations pipeline for the current split."""
        pipeline = []

        if self.is_train:
            pipeline.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    scale_limit=0.2,
                    rotate_limit=0.2,
                    shift_limit=0.3,
                    p=0.5,
                ),
            ])

        # Normalisation (ImageNet stats) + CHW conversion — always applied.
        pipeline.extend([
            A.Normalize(
                mean=DatasetConfig.MEAN,
                std=DatasetConfig.STD,
                always_apply=True,
            ),
            ToTensorV2(),
        ])
        return A.Compose(pipeline)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: str) -> PIL.Image.Image:
        return PIL.Image.open(path).convert("RGB")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = self._load_image(self.image_paths[index])

        if self.mask_paths is not None:
            mask = self._load_image(self.mask_paths[index])

            # Preprocess with HF processor (resize, format) — returns numpy.
            encoded = self.processor.preprocess(
                images=image,
                segmentation_maps=mask,
                resample=PIL.Image.Resampling.NEAREST,
                return_tensors="np",
                data_format="channels_last",
                input_data_format="channels_last",
            )
            image_np = encoded["pixel_values"].squeeze(axis=0)  # H×W×C
            mask_np  = encoded["labels"].squeeze(axis=0)        # H×W×C

            # Collapse RGB mask → single-channel class-ID mask.
            mask_np = rgb_to_grayscale(mask_np)

            # Apply augmentations (jointly on image + mask).
            transformed = self.transforms(image=image_np, mask=mask_np)
            image_t = transformed["image"]                          # C×H×W
            mask_t  = transformed["mask"].to(torch.long)            # H×W

            return image_t, mask_t

        else:
            # Inference-only path (no mask available).
            encoded = self.processor.preprocess(
                images=image,
                return_tensors="pt",
                do_rescale=True,
                rescale_factor=1.0 / 255,
                do_normalize=True,
                image_mean=DatasetConfig.MEAN,
                image_std=DatasetConfig.STD,
                resample=PIL.Image.Resampling.NEAREST,
                input_data_format="channels_last",
            )
            return encoded["pixel_values"].squeeze_()  # C×H×W


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    configs: dict,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_validation: bool = False,
    custom_batch_size: int = None,
):
    """
    Build train and validation DataLoaders.

    Args:
        configs:            WandB / dict-like config object with IMG_SIZE, BATCH_SIZE, NUM_CLASSES.
        num_workers:        DataLoader worker count.
        pin_memory:         Pin tensors to CUDA-pinned memory (faster GPU transfer).
        shuffle_validation: Shuffle the validation loader (useful for visual inspection).
        custom_batch_size:  Override configs.BATCH_SIZE for inference.
    Returns:
        (train_loader, valid_loader)
    """
    batch_size    = custom_batch_size or configs["BATCH_SIZE"]
    height, width = configs["IMG_SIZE"]

    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": height, "width": width},
        do_rescale=False,
        do_normalize=False,
    )

    cfg = DatasetConfig()

    train_dataset = CustomSegDataset(
        image_paths=cfg.train_image_paths,
        mask_paths=cfg.train_label_paths,
        is_train=True,
        num_classes=configs["NUM_CLASSES"],
        processor=processor,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_dataset = CustomSegDataset(
        image_paths=cfg.valid_image_paths,
        mask_paths=cfg.valid_label_paths,
        is_train=False,
        num_classes=configs["NUM_CLASSES"],
        processor=processor,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=shuffle_validation,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader
