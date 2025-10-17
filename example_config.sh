#!/bin/bash
# Example configuration script for HoVer-NeXt inference
# This file demonstrates common usage patterns and parameter configurations

# ==============================================================================
# BASIC CONFIGURATION
# ==============================================================================

# Example 1: Single WSI file with default parameters
python3 main.py \
    --input "path/to/slide.svs" \
    --output_dir "results/slide_output" \
    --cp "lizard_convnextv2_large" \
    --tta 4

# ==============================================================================
# PERFORMANCE TUNING
# ==============================================================================

# Example 2: High-performance configuration for a machine with 32 CPU cores
python3 main.py \
    --input "path/to/slide.svs" \
    --output_dir "results/slide_output" \
    --cp "lizard_convnextv2_large" \
    --tta 4 \
    --batch_size 64 \
    --inf_workers 32 \
    --pp_workers 31 \
    --pp_tiling 8

# Example 3: Memory-constrained configuration (if running out of memory)
python3 main.py \
    --input "path/to/slide.svs" \
    --output_dir "results/slide_output" \
    --cp "lizard_convnextv2_large" \
    --tta 4 \
    --batch_size 32 \
    --inf_workers 8 \
    --pp_workers 8 \
    --pp_tiling 16  # Increase this to reduce memory usage

# ==============================================================================
# BATCH PROCESSING
# ==============================================================================

# Example 4: Process multiple slides using glob pattern
python3 main.py \
    --input "/path/to/slides/*.svs" \
    --output_dir "results/batch_output" \
    --cp "lizard_convnextv2_large" \
    --tta 4 \
    --inf_workers 16 \
    --pp_workers 16

# Example 5: Process slides from a text file list
python3 main.py \
    --input "slide_list.txt" \
    --output_dir "results/batch_output" \
    --cp "lizard_convnextv2_large" \
    --tta 4

# ==============================================================================
# MODEL SELECTION
# ==============================================================================

# Example 6: Using PanNuke model
python3 main.py \
    --input "path/to/slide.svs" \
    --output_dir "results/pannuke_output" \
    --cp "pannuke_convnextv2_tiny_1" \
    --tta 4 \
    --metric "pannuke"

# Example 7: Ensemble prediction with multiple models
python3 main.py \
    --input "path/to/slide.svs" \
    --output_dir "results/ensemble_output" \
    --cp "lizard_convnextv2_large,lizard_convnextv2_base" \
    --tta 4

# ==============================================================================
# OUTPUT OPTIONS
# ==============================================================================

# Example 8: Generate QuPath-compatible polygon output
python3 main.py \
    --input "path/to/slide.svs" \
    --output_dir "results/qupath_output" \
    --cp "lizard_convnextv2_large" \
    --tta 4 \
    --save_polygon

# Example 9: Keep raw prediction files
python3 main.py \
    --input "path/to/slide.svs" \
    --output_dir "results/with_raw" \
    --cp "lizard_convnextv2_large" \
    --tta 4 \
    --keep_raw

# ==============================================================================
# CLUSTER/HPC USAGE
# ==============================================================================

# Example 10: Separate inference and post-processing (for GPU/CPU split on clusters)
# Step 1: Run inference on GPU node
python3 main.py \
    --input "path/to/slide.svs" \
    --output_dir "results/cluster_output" \
    --cp "lizard_convnextv2_large" \
    --tta 4 \
    --only_inference

# Step 2: Run post-processing on CPU node (remove --only_inference)
python3 main.py \
    --input "path/to/slide.svs" \
    --output_dir "results/cluster_output" \
    --cp "lizard_convnextv2_large" \
    --tta 4

# ==============================================================================
# DIFFERENT INPUT TYPES
# ==============================================================================

# Example 11: Process numpy array
python3 main.py \
    --input "path/to/image.npy" \
    --output_dir "results/npy_output" \
    --cp "lizard_convnextv2_large" \
    --tta 4

# Example 12: Process standard image format
python3 main.py \
    --input "path/to/image.png" \
    --output_dir "results/img_output" \
    --cp "lizard_convnextv2_large" \
    --tta 4

# ==============================================================================
# RESOLUTION-SPECIFIC SETTINGS
# ==============================================================================

# Example 13: High-resolution slides (0.25 mpp)
python3 main.py \
    --input "path/to/high_res_slide.svs" \
    --output_dir "results/high_res_output" \
    --cp "lizard_convnextv2_large" \
    --tta 4 \
    --overlap 0.9375  # Better overlap for high-resolution slides

# Example 14: Standard resolution slides (0.5 mpp)
python3 main.py \
    --input "path/to/slide.svs" \
    --output_dir "results/std_res_output" \
    --cp "lizard_convnextv2_large" \
    --tta 4 \
    --overlap 0.96875  # Default overlap for standard resolution
