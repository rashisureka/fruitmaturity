"""
Central configuration for the mango-only ripeness classification pipeline.

This fork of deephs_fruit is restricted to a single task:
    fruit = mango, classification_type = ripeness

All other fruit (avocado, kiwi, papaya, persimmon/kaki) and all other
classification targets (firmness, sugar) are ignored. The measurement
definitions for the other fruit are left untouched in `core/measurements`
(they share files and the hand-picked val/test split lists with mango),
but they are filtered out before any data is loaded.
"""
from core.name_convention import CameraType, ClassificationType, Fruit

# The only fruit supported by this pipeline
TARGET_FRUIT = Fruit.MANGO

# The only classification target supported by this pipeline
TARGET_CLASSIFICATION_TYPE = ClassificationType.RIPENESS

# Mango was only recorded in measurement series 3 with these two cameras
# (no INNO-SPEC Redeye / NIR recordings exist for mango).
SUPPORTED_CAMERA_TYPES = (CameraType.VIS, CameraType.VIS_COR)
DEFAULT_CAMERA_TYPE = CameraType.VIS

# Class names in label-index order (see name_convention.ripeness2int after
# merge_ripeness_levels: UNRIPE -> 0, PERFECT -> 1, OVERRIPE -> 2)
RIPENESS_CLASS_NAMES = ["Unripe", "Ripe", "Overripe"]
NUM_CLASSES = len(RIPENESS_CLASS_NAMES)


def validate_task(fruit, classification_type, camera_type=None):
    """Raise a ValueError if the requested task is not mango ripeness classification."""
    if fruit != TARGET_FRUIT:
        raise ValueError(
            f"This pipeline only supports fruit='{TARGET_FRUIT.value.lower()}', got '{getattr(fruit, 'value', fruit)}'. "
            f"Other fruit classes (avocado, kiwi, papaya, persimmon) have been removed.")
    if classification_type != TARGET_CLASSIFICATION_TYPE:
        raise ValueError(
            f"This pipeline only supports classification_type='{TARGET_CLASSIFICATION_TYPE.value}', "
            f"got '{getattr(classification_type, 'value', classification_type)}'.")
    if camera_type is not None and camera_type not in SUPPORTED_CAMERA_TYPES:
        raise ValueError(
            f"No mango recordings exist for camera_type='{getattr(camera_type, 'value', camera_type)}'. "
            f"Supported: {[c.value for c in SUPPORTED_CAMERA_TYPES]}.")
