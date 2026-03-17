"""
model.py
--------
SegFormer model factory.

Uses NVIDIA's pretrained SegFormer-B3 backbone (pretrained on ADE20K at 512×512)
and re-initialises the decode head for binary vessel segmentation.
"""

from transformers import SegformerForSemanticSegmentation


def get_model(configs: dict) -> SegformerForSemanticSegmentation:
    """
    Load a SegFormer model with a freshly initialised segmentation head.

    Args:
        configs: Dict-like object with keys MODEL_NAME and NUM_CLASSES.
    Returns:
        SegformerForSemanticSegmentation instance ready for training.
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        configs["MODEL_NAME"],
        num_labels=configs["NUM_CLASSES"],
        ignore_mismatched_sizes=True,  # Re-init the decode head for 2 classes
    )
    return model
