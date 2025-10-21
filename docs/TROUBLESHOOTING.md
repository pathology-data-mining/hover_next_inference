# Troubleshooting Guide

This guide helps you diagnose and fix common issues when using HoVer-NeXt inference.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Output Quality Issues](#output-quality-issues)
- [File Format Issues](#file-format-issues)

## Installation Issues

### CUDA/GPU Not Available

**Symptom:** Warning message "CUDA is not available. Inference will be very slow on CPU."

**Solutions:**
1. Check if GPU is detected:
   ```bash
   nvidia-smi
   ```
2. Verify PyTorch CUDA installation:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.version.cuda)
   ```
3. Reinstall PyTorch with CUDA support:
   ```bash
   pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
   ```

### OpenSlide Installation Failed

**Symptom:** `ImportError: No module named 'openslide'`

**Solutions:**

**Ubuntu/Debian:**
```bash
sudo apt-get install openslide-tools python3-openslide
pip install openslide-python
```

**macOS:**
```bash
brew install openslide
pip install openslide-python
```

**Windows:**
Download OpenSlide binaries from https://openslide.org/download/ and add to PATH.

### Zarr Import Error

**Symptom:** `ImportError: No module named 'zarr'`

**Solution:**
```bash
pip install zarr numcodecs
```

## Runtime Errors

### Out of Memory (GPU)

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**

1. **Reduce batch size:**
   ```bash
   python3 main.py --input slide.svs --output_dir results/ --cp lizard_convnextv2_large --batch_size 32
   ```

2. **Use smaller model:**
   ```bash
   python3 main.py --input slide.svs --output_dir results/ --cp lizard_convnextv2_tiny
   ```

3. **Clear GPU memory:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Out of Memory (CPU/RAM) During Post-Processing

**Symptom:** Process killed or "MemoryError" during post-processing

**Solutions:**

1. **Increase post-processing tiling:**
   ```bash
   python3 main.py --input slide.svs --output_dir results/ --cp lizard_convnextv2_large --pp_tiling 16
   ```
   Higher `pp_tiling` = less memory, but slower

2. **Reduce number of workers:**
   ```bash
   python3 main.py --input slide.svs --output_dir results/ --cp lizard_convnextv2_large --pp_workers 8
   ```

### File Not Found Error

**Symptom:** `FileNotFoundError: [Errno 2] No such file or directory`

**Solutions:**

1. **Check file path:**
   ```bash
   # Use absolute paths
   python3 main.py --input /full/path/to/slide.svs --output_dir /full/path/to/results/
   ```

2. **For glob patterns, use quotes:**
   ```bash
   python3 main.py --input "/path/to/slides/*.svs" --output_dir results/
   ```

3. **Check file exists:**
   ```bash
   ls -la /path/to/slide.svs
   ```

### Model Download Fails

**Symptom:** `HTTPError` or `ConnectionError` when downloading model weights

**Solutions:**

1. **Manual download:**
   - Visit https://zenodo.org/records/10635618
   - Download model weights (e.g., `lizard_convnextv2_large.zip`)
   - Extract in repository root directory
   - Verify folder structure:
     ```
     lizard_convnextv2_large/
     ├── config.toml
     └── checkpoint.pth
     ```

2. **Check internet connection and proxy settings**

3. **Use local checkpoint:**
   ```bash
   python3 main.py --input slide.svs --output_dir results/ --cp /path/to/local/checkpoint/
   ```

### Permission Denied Error

**Symptom:** `PermissionError: [Errno 13] Permission denied`

**Solutions:**

1. **Check write permissions:**
   ```bash
   ls -la /path/to/output_dir/
   chmod 755 /path/to/output_dir/
   ```

2. **Use different output directory:**
   ```bash
   python3 main.py --input slide.svs --output_dir ~/results/
   ```

## Performance Issues

### Inference Too Slow

**Symptom:** Processing taking hours on single slide

**Diagnosis & Solutions:**

1. **Check GPU is being used:**
   ```bash
   nvidia-smi
   # Should show python process using GPU
   ```

2. **Increase dataloader workers:**
   ```bash
   python3 main.py --input slide.svs --output_dir results/ --cp lizard_convnextv2_large --inf_workers 16
   ```
   Set to number of CPU cores

3. **Use faster model:**
   ```bash
   # tiny is 3x faster than large
   python3 main.py --input slide.svs --output_dir results/ --cp lizard_convnextv2_tiny
   ```

4. **Reduce TTA views:**
   ```bash
   # tta=1 is fastest but less robust
   python3 main.py --input slide.svs --output_dir results/ --cp lizard_convnextv2_large --tta 1
   ```

5. **Check storage speed:**
   - Use SSD instead of HDD
   - Copy WSI to local storage if on network drive
   - Use `--cache` option:
     ```bash
     python3 main.py --input /network/slide.svs --output_dir results/ --cp lizard_convnextv2_large --cache /tmp/
     ```

### Post-Processing Too Slow

**Symptom:** Inference completes quickly but post-processing takes very long

**Solutions:**

1. **Increase post-processing workers:**
   ```bash
   python3 main.py --input slide.svs --output_dir results/ --cp lizard_convnextv2_large --pp_workers 16
   ```

2. **Reduce tiling if you have enough RAM:**
   ```bash
   python3 main.py --input slide.svs --output_dir results/ --cp lizard_convnextv2_large --pp_tiling 8
   ```

3. **Skip polygon generation if not needed:**
   ```bash
   # Don't use --save_polygon flag
   python3 main.py --input slide.svs --output_dir results/ --cp lizard_convnextv2_large
   ```

## Output Quality Issues

### Too Few Nuclei Detected

**Symptom:** Many nuclei are missing from the output

**Possible Causes & Solutions:**

1. **Thresholds too high:**
   - Modify `MIN_THRESHS` in `src/inference/constants.py`
   - Lower values = more detections

2. **Wrong model:**
   - Use appropriate model for your data
   - Lizard models for H&E at 0.5 MPP
   - PanNuke models for multi-organ tissue at 0.25 MPP

3. **Resolution mismatch:**
   - Check slide resolution matches model training resolution
   - Lizard: 20x magnification (0.5 MPP)
   - PanNuke: 40x magnification (0.25 MPP)

### Too Many False Positives

**Symptom:** Background or artifacts detected as nuclei

**Solutions:**

1. **Increase minimum size threshold:**
   - Modify `MIN_THRESHS` in `src/inference/constants.py`

2. **Use different metric:**
   ```bash
   # 'mpq' is more conservative than 'f1'
   python3 main.py --input slide.svs --output_dir results/ --cp lizard_convnextv2_large --metric mpq
   ```

### Nuclei Split or Merged Incorrectly

**Symptom:** Single nuclei split into multiple instances or multiple nuclei merged

**Solutions:**

1. **Adjust overlap for stitching:**
   ```bash
   python3 main.py --input slide.svs --output_dir results/ --cp lizard_convnextv2_large --overlap 0.9375
   ```
   Higher overlap (e.g., 0.96875) = better stitching but slower

2. **Use more TTA views:**
   ```bash
   python3 main.py --input slide.svs --output_dir results/ --cp lizard_convnextv2_large --tta 8
   ```

### Wrong Classifications

**Symptom:** Cell types are incorrectly classified

**Solutions:**

1. **Use ensemble of models:**
   ```bash
   python3 main.py --input slide.svs --output_dir results/ --cp lizard_convnextv2_large,lizard_convnextv2_base
   ```

2. **Check if using correct model:**
   - Lizard models: trained on colon tissue
   - PanNuke models: trained on multi-organ data
   - Consider retraining on your specific tissue type

## File Format Issues

### WSI Format Not Supported

**Symptom:** `OpenSlideError: Unsupported or missing image file`

**Solutions:**

1. **Check format support:**
   ```python
   import openslide
   print(openslide.OpenSlide.detect_format('/path/to/slide'))
   ```

2. **Supported formats:**
   - Aperio (.svs, .tif)
   - Hamamatsu (.ndpi, .vms, .vmu)
   - Leica (.scn)
   - MIRAX (.mrxs)
   - Philips (.tiff)
   - Sakura (.svslide)
   - Trestle (.tif)
   - Ventana (.bif, .tif)
   - Generic tiled TIFF
   - CZI (via pylibCZIrw)

3. **Convert unsupported formats:**
   - Use bioformats or QuPath to convert to TIFF

### Cannot Open Output Files

**Symptom:** Cannot open .zip files in output directory

**Solutions:**

1. **These are zarr compressed arrays, not regular zip files:**
   ```python
   import zarr
   import os
   
   # Load instance segmentation
   pinst = zarr.open(os.path.join(output_dir, "pinst_pp.zip"))
   
   # Convert to numpy array
   import numpy as np
   inst_array = np.array(pinst)
   
   # Save as TIFF
   import tifffile
   tifffile.imwrite("instance_map.tif", inst_array)
   ```

2. **For visualization, use TSV files:**
   - Import `pred_*.tsv` files into QuPath
   - Or convert to other formats using provided scripts

### QuPath Import Issues

**Symptom:** Cannot import TSV or GeoJSON files into QuPath

**Solutions:**

1. **TSV import in QuPath:**
   - Automate → Show script editor
   - Use detection import script
   - Point to TSV files in output directory

2. **GeoJSON import:**
   - File → Object data → Import object data
   - Select `poly.geojson` from output directory
   - Requires `--save_polygon` flag during inference

3. **Check coordinate system:**
   - Coordinates should match image dimensions
   - Check `ds_factor` in output

## Getting More Help

If your issue is not covered here:

1. **Check existing issues:** https://github.com/pathology-data-mining/hover_next_inference/issues
2. **Create new issue with:**
   - Complete error message
   - Python version (`python --version`)
   - PyTorch version (`python -c "import torch; print(torch.__version__)"`)
   - CUDA version (`nvidia-smi`)
   - Input file type and size
   - Complete command line used
   - Operating system

3. **Read the paper:** https://openreview.net/pdf?id=3vmB43oqIO

## Common Command Line Patterns

### Basic WSI Processing
```bash
python3 main.py \
    --input /path/to/slide.svs \
    --output_dir results/ \
    --cp lizard_convnextv2_large \
    --tta 4 \
    --inf_workers 16 \
    --pp_workers 16
```

### Memory-Constrained System
```bash
python3 main.py \
    --input /path/to/slide.svs \
    --output_dir results/ \
    --cp lizard_convnextv2_tiny \
    --batch_size 32 \
    --tta 1 \
    --pp_tiling 16 \
    --pp_workers 8
```

### High-Performance System
```bash
python3 main.py \
    --input /path/to/slide.svs \
    --output_dir results/ \
    --cp lizard_convnextv2_large \
    --batch_size 128 \
    --tta 8 \
    --inf_workers 32 \
    --pp_workers 31 \
    --pp_tiling 4
```

### Batch Processing
```bash
python3 main.py \
    --input "/path/to/slides/*.svs" \
    --output_dir batch_results/ \
    --cp lizard_convnextv2_large \
    --tta 4 \
    --inf_workers 16
```
