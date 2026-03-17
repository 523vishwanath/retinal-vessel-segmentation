"""
config.py
---------
Centralised configuration dataclasses for the retinal vessel segmentation project.
All hyperparameters live here — change once, applied everywhere.
"""

import os
from dataclasses import dataclass, field
from glob import glob


# ---------------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    NUM_CLASSES: int = 2
    IMG_WIDTH:  int = 576
    IMG_HEIGHT: int = 576

    DATASET_ROOT: str = "icpr_prepared"

    TRAIN_IMAGES_DIR: str = field(init=False)
    TRAIN_LABELS_DIR: str = field(init=False)
    VALID_IMAGES_DIR: str = field(init=False)
    VALID_LABELS_DIR: str = field(init=False)

    IMAGE_EXTENSIONS: tuple = (".tif", ".png", ".jpg", ".jpeg")

    # ImageNet statistics used for normalisation (matches SegFormer pre-training)
    MEAN: tuple = (0.485, 0.456, 0.406)
    STD:  tuple = (0.229, 0.224, 0.225)

    BACKGROUND_CLS_ID: int = 0

    def __post_init__(self):
        self.TRAIN_IMAGES_DIR = os.path.join(self.DATASET_ROOT, "train_images")
        self.TRAIN_LABELS_DIR = os.path.join(self.DATASET_ROOT, "train_labels")
        self.VALID_IMAGES_DIR = os.path.join(self.DATASET_ROOT, "test_images")
        self.VALID_LABELS_DIR = os.path.join(self.DATASET_ROOT, "test_labels")

    def get_file_paths(self, directory: str) -> list:
        """Return sorted list of all image files in *directory*."""
        paths = []
        for ext in self.IMAGE_EXTENSIONS:
            paths.extend(glob(os.path.join(directory, f"*{ext}")))
        return sorted(paths)

    @property
    def train_image_paths(self):
        return self.get_file_paths(self.TRAIN_IMAGES_DIR)

    @property
    def train_label_paths(self):
        return self.get_file_paths(self.TRAIN_LABELS_DIR)

    @property
    def valid_image_paths(self):
        return self.get_file_paths(self.VALID_IMAGES_DIR)

    @property
    def valid_label_paths(self):
        return self.get_file_paths(self.VALID_LABELS_DIR)


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    BATCH_SIZE:    int   = 10
    NUM_EPOCHS:    int   = 60
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY:  float = 1e-4
    NUM_WORKERS:   int   = field(default_factory=lambda: os.cpu_count())
    OPTIMIZER:     str   = "AdamW"
    LR_SCHEDULER:  str   = "MultiStepLR"


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    MODEL_NAME: str = "nvidia/segformer-b3-finetuned-ade-512-512"


# ---------------------------------------------------------------------------
# Inference Configuration
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    BATCH_SIZE:       int = 10
    NUM_BATCHES:      int = 2
    CHECKPOINT_PATH:  str = "checkpoints/best_model.tar"


# ---------------------------------------------------------------------------
# Class-to-colour mappings
# ---------------------------------------------------------------------------

# Ground-truth colour map: 0 → black (background), 1 → white (vessel)
ID2COLOR = {
    0: (0,   0,   0),
    1: (255, 255, 255),
}

# Visualisation colour map: green vessels for better contrast
ID2COLOR_VIZ = {
    0: (0,   0,   0),
    1: (20,  220, 20),
}

# Reverse mapping (RGB tuple → class id) used during mask loading
REV_ID2COLOR = {v: k for k, v in ID2COLOR.items()}
