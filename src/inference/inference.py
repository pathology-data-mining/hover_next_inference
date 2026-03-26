import os
from pathlib import Path
import copy
import toml
import requests
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import List, Union, Tuple
import torch
import numpy as np
import zarr
import zipfile
from numcodecs import Blosc
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from scipy.special import softmax
from inference.multi_head_unet import get_model, load_checkpoint
from inference.data_utils import WholeSlideDataset, NpyDataset, ImageDataset
from inference.augmentations import color_augmentations
from inference.spatial_augmenter import SpatialAugmenter
from inference.constants import TTA_AUG_PARAMS, VALID_WEIGHTS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inference_main(
    params: dict,
    models,
    augmenter,
    color_aug_fn,
):
    """
    Inference function for a single input file.

    Parameters
    ----------
    params: dict
        Parameter store, defined in initial main
    models: List[torch.nn.Module]
        list of models to run inference with, e.g. multiple folds or a single model in a list
    augmenter: SpatialAugmenter
        Augmentation module for geometric transformations
    color_aug_fn: torch.nn.Sequential
        Color Augmentation module

    Returns
    ----------
    params: dict
        Parameter store, defined in initial main and modified by this function
    z: Union(Tuple[zarr.storage.ZipStore, zarr.storage.ZipStore], None)
        instance and class segmentation results as zarr stores, kept open for further processing. None if inference was skipped.
    """
    output_dir = Path(params.get("output_dir", '.'))
    output_dir.mkdir(parents=True, exist_ok=True)
    params["model_out_p"] = os.path.join(
        params["output_dir"], "raw_" + str(params["tile_size"])
    )
    prog_path = os.path.join(params["output_dir"], "progress.txt")

    if os.path.exists(os.path.join(params["output_dir"], "pinst_pp.zip")):
        print(
            "inference and postprocessing already completed, delete output or specify different output path to re-run"
        )
        return params, None

    if (
        os.path.exists(params["model_out_p"] + "_inst.zip")
        and os.path.exists(params["model_out_p"] + "_cls.zip")
        and not os.path.exists(prog_path)
    ):
        try:
            z_inst = zarr.open(zarr.storage.ZipStore(params["model_out_p"] + "_inst.zip", mode="r"))
            z_cls = zarr.open(zarr.storage.ZipStore(params["model_out_p"] + "_cls.zip", mode="r"))
            print("Inference already completed", z_inst.shape, z_cls.shape)
            return params, (z_inst, z_cls)
        except (KeyError, zipfile.BadZipFile):
            z_inst = None
            z_cls = None
            print(
                "something went wrong with previous output files, rerunning inference"
            )

    z_inst = None
    z_cls = None

    if not torch.cuda.is_available():
        print("trying to run inference on CPU, aborting...")
        print("if this is intended, remove this check")
        raise Exception("No GPU available")

    # create datasets from specified input

    if params["input_type"] == "npy":
        dataset = NpyDataset(
            params["p"],
            params["tile_size"],
            padding_factor=params["overlap"],
            ratio_object_thresh=0.3,
            min_tiss=0.1,
        )
    elif params["input_type"] == "img":
        dataset = ImageDataset(
            params["p"],
            params["tile_size"],
            padding_factor=params["overlap"],
            ratio_object_thresh=0.3,
            min_tiss=0.1,
        )
    else:
        level = 40 if params["pannuke"] else 20
        dataset = WholeSlideDataset(
            params["p"],
            crop_sizes_px=[params["tile_size"]],
            crop_magnifications=[level],
            padding_factor=params["overlap"],
            remove_background=True,
            ratio_object_thresh=0.0001,
        )

    # setup output files to write to, also create dummy file to resume inference if interrupted

    # Create zarr compressed arrays for storing raw predictions
    # Instance predictions: HoVer maps (horizontal/vertical gradients + foreground)
    z_inst = zarr.open(
        zarr.storage.ZipStore(params["model_out_p"] + "_inst.zip", mode="w"),
        shape=(len(dataset), 3, params["tile_size"], params["tile_size"]),
        chunks=(params["batch_size"], 3, params["tile_size"], params["tile_size"]),
        zarr_format=2,
        dtype="f4",
        compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.SHUFFLE),
    )
    # Class predictions: cell type probabilities
    z_cls = zarr.open(
        zarr.storage.ZipStore(params["model_out_p"] + "_cls.zip", mode="w"),
        shape=(
            len(dataset),
            params["out_channels_cls"],
            params["tile_size"],
            params["tile_size"],
        ),
        chunks=(
            params["batch_size"],
            params["out_channels_cls"],
            params["tile_size"],
            params["tile_size"],
        ),
        zarr_format=2,
        dtype="u1",
        compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.BITSHUFFLE),
    )
    
    # Create progress file to enable resuming interrupted inference runs
    with open(prog_path, "w") as f:
        f.write("0")
    inf_start = 0

    n_workers = params["inf_workers"]
    dataloader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=n_workers > 0,
        prefetch_factor=4 if n_workers > 0 else None,
    )

    # IO thread function: transfers GPU tensors to CPU and writes to zarr.
    # Running in a background thread overlaps GPU->CPU transfer and disk I/O
    # with the GPU compute of the next batch.
    def dump_results(ct_gpu, inst_gpu, zc_, z_cls, z_inst, prog_path):
        """
        Transfer predictions from GPU to CPU and write to zarr stores.

        Accepts GPU tensors so the transfer happens in the background thread,
        overlapping with the next batch's GPU computation.
        """
        cls_ = ct_gpu.cpu().numpy()
        inst_ = inst_gpu.cpu().numpy()
        cls_ = (softmax(cls_.astype(np.float32), axis=1) * 255).astype(np.uint8)
        z_cls[zc_ : zc_ + cls_.shape[0]] = cls_
        z_inst[zc_ : zc_ + inst_.shape[0]] = inst_.astype(np.float32)
        with open(prog_path, "w") as f:
            f.write(str(zc_))

    # Separate thread pool for I/O to prevent blocking GPU inference
    with ThreadPoolExecutor(max_workers=params["inf_writers"]) as executor:
        futures = []
        zc = inf_start
        for raw, _ in tqdm(dataloader):
            raw = raw.to(device, non_blocking=True).float()
            raw = raw.permute(0, 3, 1, 2)  # BHWC -> BCHW

            with torch.inference_mode():
                ct, inst = batch_pseudolabel_ensemb(
                    raw, models, params["tta"], augmenter, color_aug_fn
                )
                # Pass GPU tensors directly; the thread handles CPU transfer.
                # This lets the main thread return to GPU work immediately.
                futures.append(
                    executor.submit(dump_results, ct, inst, zc, z_cls, z_inst, prog_path)
                )
                zc += params["batch_size"]

        for _ in concurrent.futures.as_completed(futures):
            pass
    # clean up
    if os.path.exists(prog_path):
        os.remove(prog_path)
    return params, (z_inst, z_cls)


def batch_pseudolabel_ensemb(
    raw: torch.Tensor,
    models: List[torch.nn.Module],
    nviews: int,
    aug: SpatialAugmenter,
    color_aug_fn: torch.nn.Sequential,
):
    """
    Run inference step on batch of images with test time augmentations

    Parameters
    ----------

    raw: torch.Tensor
        batch of input images
    models: List[torch.nn.Module]
        list of models to run inference with, e.g. multiple folds or a single model in a list
    nviews: int
        Number of test-time augmentation views to aggregate
    aug: SpatialAugmenter
        Augmentation module for geometric transformations
    color_aug_fn: torch.nn.Sequential
        Color Augmentation module

    Returns
    ----------

    ct: torch.Tensor
        Per pixel class predictions as a tensor of shape (batch_size, n_classes+1, tilesize, tilesize)
    inst: torch.Tensor
        Per pixel 3 class prediction map with boundary, background and foreground classes, shape (batch_size, 3, tilesize, tilesize)
    """
    # Use running accumulation instead of building a list then stacking,
    # which avoids holding `nviews` extra tensors in GPU memory.
    ct_acc: torch.Tensor | None = None
    inst_acc: torch.Tensor | None = None

    n = max(1, nviews)
    use_tta = nviews > 0

    for _ in range(n):
        if use_tta:
            aug.interpolation = "bilinear"
            view = aug.forward_transform(raw)
            aug.interpolation = "nearest"
            view = torch.clamp(color_aug_fn(view), 0, 1)
        else:
            view = raw

        out_fast = []
        for mod in models:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = mod(view)
            if use_tta:
                pred = aug.inverse_transform(pred)
            out_fast.append(pred)

        out = torch.stack(out_fast).nanmean(0)
        view_inst = out[:, 2:5].softmax(1)
        view_ct = out[:, 5:].softmax(1)

        if ct_acc is None:
            ct_acc, inst_acc = view_ct, view_inst
        else:
            ct_acc = ct_acc + view_ct
            inst_acc = inst_acc + view_inst

    return ct_acc / n, inst_acc / n


def get_inference_setup(params):
    """
    Load model checkpoints, create augmentation functions, and configure inference parameters.

    Parameters
    ----------
    params : dict
        Parameter dictionary. Must contain 'data_dirs' (list of checkpoint paths).
        Modified in-place to add 'out_channels_cls', 'inst_channels', 'pannuke'.

    Returns
    -------
    params : dict
        Updated parameter dictionary
    models : list of torch.nn.Module
        Loaded and ready-to-use model instances
    augmenter : SpatialAugmenter
        Geometric TTA augmentation module on device
    color_aug_fn : torch.nn.Sequential
        Color augmentation module on device

    Raises
    ------
    ValueError
        If data_dirs is empty or models have conflicting dataset types
    """
    if not params["data_dirs"]:
        raise ValueError("No model checkpoints specified in 'data_dirs'.")

    models = []
    pannuke_flags = []
    for pth in params["data_dirs"]:
        if not os.path.exists(pth):
            pth = download_weights(os.path.split(pth)[-1])

        checkpoint_path = f"{pth}/train/best_model"
        mod_params = toml.load(f"{pth}/params.toml")
        params["out_channels_cls"] = mod_params["out_channels_cls"]
        params["inst_channels"] = mod_params["inst_channels"]
        pannuke_flags.append(mod_params["dataset"] == "pannuke")
        model = get_model(
            enc=mod_params["encoder"],
            out_channels_cls=params["out_channels_cls"],
            out_channels_inst=params["inst_channels"],
        ).to(device)
        model = load_checkpoint(model, checkpoint_path, device)
        model.eval()
        try:
            # torch.compile reduces kernel launch overhead via CUDA graphs.
            # mode="reduce-overhead" is optimal for repeated same-shape inference.
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass  # Graceful fallback for environments without compiler support
        models.append(copy.deepcopy(model))

    if len(set(pannuke_flags)) > 1:
        raise ValueError("All model checkpoints must be trained on the same dataset (cannot mix Lizard and PanNuke models).")
    params["pannuke"] = pannuke_flags[0]

    augmenter = SpatialAugmenter(TTA_AUG_PARAMS).to(device)
    color_aug_fn = color_augmentations(False, rank=device)

    print(
        "Processing input using",
        "PanNuke" if params["pannuke"] else "Lizard",
        f"trained model ({len(models)} checkpoint(s))",
    )

    return params, models, augmenter, color_aug_fn


def download_weights(model_code):
    if model_code in VALID_WEIGHTS:
        url = f"https://zenodo.org/records/10635618/files/{model_code}.zip"
        print("downloading",model_code,"weights to",os.getcwd())
        try:
            response = requests.get(url, stream=True, timeout=60.0)
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Timed out downloading weights for '{model_code}' from {url}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download weights for '{model_code}': {e}")
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        with tqdm(total=total_size, unit="iB", unit_scale=True) as t:
            with open("cache.zip", "wb") as f:
                for data in response.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
        with zipfile.ZipFile("cache.zip", "r") as zip:
            zip.extractall("")
        os.remove("cache.zip")
        return model_code
    else:
        raise ValueError(f"Unknown model ID '{model_code}'. Valid options: {VALID_WEIGHTS}")
