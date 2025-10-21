# HoVer-NeXt Inference API Documentation

This document provides comprehensive documentation for developers who want to use or extend the HoVer-NeXt inference pipeline programmatically.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Modules](#core-modules)
- [API Reference](#api-reference)
- [Advanced Usage](#advanced-usage)
- [Custom Extensions](#custom-extensions)

## Installation

### As a Python Package

```bash
pip install -e .
```

### Dependencies

See `requirements.txt` for the complete list. Key dependencies:
- PyTorch >= 2.1.1
- OpenSlide for WSI support
- Zarr for efficient array storage
- segmentation-models-pytorch
- timm (PyTorch Image Models)

## Quick Start

### Using as a Library

```python
from inference.inference import inference_main, get_inference_setup
from inference.post_process import post_process_main

# Setup parameters
params = {
    "cp": "lizard_convnextv2_large",
    "input": "/path/to/slide.svs",
    "output_dir": "results/",
    "tta": 4,
    "batch_size": 64,
    "tile_size": 256,
    "overlap": 0.96875,
    # ... other parameters
}

# Get model and augmenter
params, models, augmenter, color_aug_fn = get_inference_setup(params)

# Run inference
params, z = inference_main(params, models, augmenter, color_aug_fn)

# Post-process results
z_pp = post_process_main(params, z)
```

## Core Modules

### 1. inference.py

**Purpose:** Main inference pipeline for tile-based prediction on whole slide images, arrays, and standard images.

**Key Functions:**

- `inference_main(params, models, augmenter, color_aug_fn)`: Run inference on a single input
- `get_inference_setup(params)`: Initialize models and augmentation modules

**Supported Input Types:**
- WSI formats: All OpenSlide-supported formats (.svs, .ndpi, .mrxs, etc.)
- CZI format via pylibCZIrw
- Numpy arrays (.npy)
- Standard images (.jpg, .png, .jpeg, .bmp)

### 2. post_process.py

**Purpose:** Convert raw model predictions into final instance segmentation maps.

**Key Functions:**

- `post_process_main(params, z)`: Main post-processing pipeline
- Performs tile stitching, overlap resolution, and watershed segmentation

**Output Formats:**
- Instance segmentation map (zarr compressed)
- Class lookup dictionary (JSON)
- QuPath-compatible TSV files
- Optional GeoJSON polygons

### 3. data_utils.py

**Purpose:** Data loading and preprocessing utilities.

**Key Classes:**

- `WholeSlideDataset`: Dataset for WSI tile extraction
- `NpyDataset`: Dataset for numpy array inputs
- `ImageDataset`: Dataset for standard image formats
- `czi_wrapper`: Wrapper for CZI file support

**Key Functions:**

- `normalize_min_max(x, mi, ma)`: Min-max normalization
- `center_crop(t, croph, cropw)`: Center cropping utility

### 4. augmentations.py

**Purpose:** Color augmentation for histopathology images.

**Key Classes:**

- `HedNormalizeTorch`: Stain augmentation in HED color space
- `Rgb2Hed`: RGB to HED conversion module
- `Hed2Rgb`: HED to RGB conversion module
- `GaussianNoise`: Gaussian noise augmentation

**Key Functions:**

- `color_augmentations(train, sigma, bias, s, rank)`: Create augmentation pipeline

### 5. spatial_augmenter.py

**Purpose:** Geometric transformations for test-time augmentation.

**Key Class:**

- `SpatialAugmenter`: Reversible geometric augmentation module

**Supported Transformations:**
- Mirror (horizontal/vertical flipping)
- Translation
- Scaling
- Rotation (90° and arbitrary angles)
- Shearing
- Elastic deformation

### 6. multi_head_unet.py

**Purpose:** Model architecture implementation.

**Key Classes:**

- `MultiHeadModel`: Complete multi-head U-Net model
- `TimmEncoderFixed`: ConvNeXt encoder wrapper
- `UnetDecoder`: U-Net decoder with skip connections

**Key Functions:**

- `get_model(enc, out_channels_cls, out_channels_inst, pretrained)`: Create model
- `load_checkpoint(model, cp_path, device)`: Load model weights

### 7. viz_utils.py

**Purpose:** Visualization and export utilities.

**Key Functions:**

- `create_geojson(polygons, classids, lookup, params)`: Export to GeoJSON
- `create_tsvs(pcls_out, params)`: Export centroid TSVs for QuPath
- `cont(x, offset)`: Extract contour from binary mask

## API Reference

### Main Entry Point

#### `inference_main(params, models, augmenter, color_aug_fn)`

Run inference on a single input file.

**Parameters:**
- `params` (dict): Configuration parameters
- `models` (list): List of model instances
- `augmenter` (SpatialAugmenter): Geometric augmentation module
- `color_aug_fn` (torch.nn.Sequential): Color augmentation pipeline

**Returns:**
- `params` (dict): Updated parameters
- `z` (tuple): (instance_predictions, class_predictions) as zarr stores

**Example:**
```python
params, z = inference_main(params, models, augmenter, color_aug_fn)
```

### Post-Processing

#### `post_process_main(params, z)`

Post-process raw predictions into instance segmentation.

**Parameters:**
- `params` (dict): Configuration parameters
- `z` (tuple or None): Zarr stores from inference

**Returns:**
- `z_pp` (zarr.Array): Final instance segmentation map

**Example:**
```python
z_pp = post_process_main(params, z)
```

### Model Setup

#### `get_inference_setup(params)`

Initialize models and augmentation modules.

**Parameters:**
- `params` (dict): Configuration with 'cp', 'tta', etc.

**Returns:**
- `params` (dict): Updated parameters
- `models` (list): List of loaded models
- `augmenter` (SpatialAugmenter): Geometric augmenter
- `color_aug_fn` (torch.nn.Sequential): Color augmenter

**Example:**
```python
params, models, augmenter, color_aug = get_inference_setup(params)
```

### Dataset Classes

#### `WholeSlideDataset(slide_path, ...)`

Dataset for extracting tiles from whole slide images.

**Parameters:**
- `slide_path` (str): Path to WSI file
- `tile_size` (int): Size of tiles to extract (default: 256)
- `padding_factor` (float): Overlap between tiles (default: 0.96875)
- `ratio_object_thresh` (float): Minimum tissue ratio (default: 0.3)
- `min_tiss` (float): Minimum tissue percentage (default: 0.1)

**Methods:**
- `__len__()`: Number of tiles
- `__getitem__(idx)`: Get tile and metadata

**Example:**
```python
dataset = WholeSlideDataset(
    "slide.svs",
    tile_size=256,
    padding_factor=0.96875
)
dataloader = DataLoader(dataset, batch_size=64, num_workers=16)
```

## Advanced Usage

### Custom Input Handling

To add support for a new image format:

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, file_path, tile_size, **kwargs):
        # Initialize your reader
        self.reader = CustomReader(file_path)
        self.tile_size = tile_size
        # Calculate tiles
        
    def __len__(self):
        return self.num_tiles
    
    def __getitem__(self, idx):
        # Extract tile
        tile = self.reader.read_tile(idx)
        # Return (tile, tile_coords)
        return tile, coords
```

### Custom Post-Processing Parameters

Optimize post-processing for your specific use case:

```python
from inference.post_process_utils import get_pp_params

# Load default parameters
params = get_pp_params(params, optimize=True)

# Customize thresholds
params['best_fg_thresh_cl'] = [0.45] * num_classes  # Foreground threshold
params['best_seediness_thresh_cl'] = [0.55] * num_classes  # Seed threshold

# Run with custom parameters
z_pp = post_process_main(params, z)
```

### Batch Processing Multiple Files

```python
import glob
from pathlib import Path

# Get list of files
files = glob.glob("/path/to/slides/*.svs")

for file_path in files:
    print(f"Processing {file_path}")
    
    # Update parameters for this file
    params['p'] = file_path
    params['output_dir'] = f"results/{Path(file_path).stem}/"
    
    # Run pipeline
    params, z = inference_main(params, models, augmenter, color_aug_fn)
    z_pp = post_process_main(params, z)
    
    # Clean up
    if not params['keep_raw']:
        # Remove intermediate files
        pass
```

### Test-Time Augmentation (TTA)

```python
# Configure TTA views
params['tta'] = 8  # More views = more robust but slower

# Custom augmentation parameters
from inference.constants import TTA_AUG_PARAMS

custom_aug_params = TTA_AUG_PARAMS.copy()
custom_aug_params['rotate']['prob'] = 1.0  # Always rotate
custom_aug_params['mirror']['prob'] = 1.0  # Always mirror

augmenter = SpatialAugmenter(custom_aug_params)
```

## Custom Extensions

### Adding a New Model Checkpoint

1. Train your model using the [training repository](https://github.com/digitalpathologybern/hover_next_train)
2. Save checkpoint with this structure:
```python
{
    'model_state_dict': model.state_dict(),
    # Optional: other training info
}
```
3. Create a config.toml file:
```toml
[model]
encoder = "convnextv2_large.fcmae_ft_in22k_in1k"
out_channels_cls = 8
out_channels_inst = 5

[dataset]
pannuke = false
mpp = 0.5
```
4. Use in inference:
```python
params['cp'] = "/path/to/your/checkpoint/"
```

### Custom Metrics for Post-Processing

Add your own metric optimization:

```python
# In post_process_utils.py, extend get_pp_params()
def get_pp_params(params, optimize=True):
    # ... existing code ...
    
    if params['metric'] == 'custom':
        # Load your optimized parameters
        params['best_fg_thresh_cl'] = [0.50] * num_classes
        params['best_seediness_thresh_cl'] = [0.60] * num_classes
        # ... other parameters
    
    return params
```

### Integration with Other Tools

#### Export for CellProfiler

```python
import tifffile

# Load results
pinst = zarr.open(os.path.join(output_dir, "pinst_pp.zip"))

# Save as TIFF
tifffile.imwrite(
    "instance_map.tif",
    np.array(pinst),
    compression='lzw'
)
```

#### Export for ImageJ/Fiji

```python
from skimage.segmentation import find_boundaries

# Load instance map
pinst = zarr.open(os.path.join(output_dir, "pinst_pp.zip"))

# Extract boundaries
boundaries = find_boundaries(pinst[:], mode='inner')

# Save as binary mask
cv2.imwrite("boundaries.png", boundaries.astype(np.uint8) * 255)
```

## Performance Optimization

### GPU Memory Management

```python
# Reduce batch size if running out of memory
params['batch_size'] = 32

# Increase tiling for post-processing
params['pp_tiling'] = 16  # Higher = less memory but slower
```

### CPU Parallelization

```python
# Set workers based on CPU cores
import multiprocessing

n_cores = multiprocessing.cpu_count()
params['inf_workers'] = n_cores  # Inference dataloader
params['pp_workers'] = n_cores - 1  # Post-processing
```

### Caching for Network Storage

```python
# Cache WSI to local storage for faster access
params['cache'] = "/tmp/cache/"
```

## Error Handling

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size`
   - Increase `pp_tiling`
   - Use smaller model variant (tiny instead of large)

2. **Slow Inference**
   - Check GPU is being used
   - Increase `inf_workers`
   - Use faster storage (SSD vs HDD)

3. **Missing Tissue Detection**
   - Adjust `ratio_object_thresh` and `min_tiss`
   - Check WSI thumbnail quality

## Support

For issues and questions:
- GitHub Issues: https://github.com/pathology-data-mining/hover_next_inference/issues
- Documentation: https://github.com/pathology-data-mining/hover_next_inference#readme
- Paper: https://openreview.net/pdf?id=3vmB43oqIO

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{baumann2024hover,
  title={HoVer-NeXt: A Fast Nuclei Segmentation and Classification Pipeline for Next Generation Histopathology},
  author={Baumann, Elias and Dislich, Bastian and Rumberger, Josef Lorenz and Nagtegaal, Iris D and Martinez, Maria Rodriguez and Zlobec, Inti},
  booktitle={Medical Imaging with Deep Learning},
  year={2024}
}
```
