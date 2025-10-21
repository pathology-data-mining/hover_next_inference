"""
Constants and configuration parameters for HoVer-NeXt inference.

This module defines all constants used throughout the inference pipeline including:
- Size thresholds for nuclei filtering
- Color mappings for visualization
- Class labels for different datasets
- Resolution and magnification lookup tables
- Test-time augmentation parameters
- Valid pretrained model identifiers

These constants are tuned for optimal performance on the Lizard and PanNuke datasets
but can be modified for custom use cases.
"""

### Size thresholds for nuclei (in pixels), pannuke is less conservative
# These have been optimized for the conic challenge, but can be changed
# to get more small nuclei (e.g. by setting all min_threshs to 0)
# Thresholds format: [neutrophil, epithelial, lymphocyte, plasma, eosinophil, connective, mitosis]
MIN_THRESHS_LIZARD = [30, 30, 20, 20, 30, 30, 15]
MAX_THRESHS_LIZARD = [5000, 5000, 5000, 5000, 5000, 5000, 5000]

# PanNuke uses more permissive thresholds
# Thresholds format: [neoplastic, inflammatory, connective, dead, epithelial]
MIN_THRESHS_PANNUKE = [10, 10, 10, 10, 10]
MAX_THRESHS_PANNUKE = [20000, 20000, 20000, 3000, 10000]

# Maximal size of holes to remove from a nucleus (in pixels)
MAX_HOLE_SIZE = 128

# Colors for geojson output (RGB format)
# Lizard dataset: 7 cell types
COLORS_LIZARD = [
    [0, 255, 0],  # neu (neutrophil) - green
    [255, 0, 0],  # epi (epithelial) - red
    [0, 0, 255],  # lym (lymphocyte) - blue
    [0, 128, 0],  # pla (plasma) - dark green
    [0, 255, 255],  # eos (eosinophil) - cyan
    [255, 179, 102],  # con (connective) - peach
    [255, 0, 255],  # mitosis - magenta
]

# PanNuke dataset: 5 tissue types
COLORS_PANNUKE = [
    [255, 0, 0],  # neo (neoplastic) - red
    [0, 127, 255],  # inf (inflammatory) - light blue
    [255, 179, 102],  # con (connective) - peach
    [0, 0, 0],  # dead - black
    [0, 255, 0],  # epi (epithelial) - green
]

# Text labels for Lizard dataset (7 classes)
CLASS_LABELS_LIZARD = {
    "neutrophil": 1,
    "epithelial-cell": 2,
    "lymphocyte": 3,
    "plasma-cell": 4,
    "eosinophil": 5,
    "connective-tissue-cell": 6,
    "mitosis": 7,
}

# Text labels for PanNuke dataset (5 tissue types)
CLASS_LABELS_PANNUKE = {
    "neoplastic": 1,
    "inflammatory": 2,
    "connective": 3,
    "dead": 4,
    "epithelial": 5,
}

# Magnification and resolution lookup tables for WSI dataloader
# Maps magnification levels to their corresponding resolutions
LUT_MAGNIFICATION_X = [10, 20, 40, 80]  # Magnification levels
LUT_MAGNIFICATION_MPP = [0.97, 0.485, 0.2425, 0.124]  # Microns per pixel

# Target resolutions for different datasets
CONIC_MPP = 0.5  # Lizard/CoNIC dataset resolution
PANNUKE_MPP = 0.25  # PanNuke dataset resolution

# Parameters for test time augmentations
# These control the probability and range of each augmentation type
# DO NOT CHANGE - these are carefully tuned for the HoVer-NeXt model
TTA_AUG_PARAMS = {
    "mirror": {"prob_x": 0.5, "prob_y": 0.5, "prob": 0.75},  # Horizontal/vertical flipping
    "translate": {"max_percent": 0.03, "prob": 0.0},  # Translation (disabled)
    "scale": {"min": 0.8, "max": 1.2, "prob": 0.0},  # Scaling (disabled)
    "zoom": {"min": 0.8, "max": 1.2, "prob": 0.0},  # Zoom (disabled)
    "rotate": {"rot90": True, "prob": 0.75},  # 90-degree rotations
    "shear": {"max_percent": 0.1, "prob": 0.0},  # Shear (disabled)
    "elastic": {"alpha": [120, 120], "sigma": 8, "prob": 0.0},  # Elastic deformation (disabled)
}

# Current valid pre-trained weights to be automatically downloaded and used in HoVer-NeXt
# These weights are hosted on Zenodo and will be downloaded on first use
VALID_WEIGHTS = [
    "lizard_convnextv2_large",  # Best performance on Lizard dataset
    "lizard_convnextv2_base",
    "lizard_convnextv2_tiny",  # Fastest inference
    "pannuke_convnextv2_tiny_1",  # PanNuke 3-fold cross-validation models
    "pannuke_convnextv2_tiny_2",
    "pannuke_convnextv2_tiny_3",
]