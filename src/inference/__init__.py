"""
HoVer-NeXt Inference Package.

This package provides fast and efficient nuclei segmentation and classification
for histopathology images using deep learning models based on ConvNeXt-v2 architecture.

Main Components
---------------
- inference: Core inference pipeline for WSI, NPY, and image inputs
- post_process: Post-processing and instance segmentation refinement
- data_utils: Data loading and preprocessing utilities
- augmentations: Color augmentation modules
- spatial_augmenter: Geometric transformation for test-time augmentation
- multi_head_unet: Multi-head U-Net model architecture
- viz_utils: Visualization and export utilities

Example Usage
-------------
>>> from inference.__main__ import main
>>> # Run inference from command line
>>> # python -m inference --input slide.svs --output_dir results/ --cp lizard_convnextv2_large

For more information, see:
https://github.com/pathology-data-mining/hover_next_inference
"""
