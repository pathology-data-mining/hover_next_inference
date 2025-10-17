import os
import argparse
import sys
from timeit import default_timer as timer
from datetime import timedelta
import torch
from glob import glob
from inference.inference import inference_main, get_inference_setup
from inference.post_process import post_process_main
from inference.data_utils import copy_img

torch.backends.cudnn.benchmark = True
print(torch.cuda.device_count(), " cuda devices")


def prepare_input(params):
    """
    Check if input is a text file, glob pattern, or a directory, and return a list of input files

    Parameters
    ----------
    params: dict
        input parameters from argparse

    Returns
    -------
    list
        List of input file paths

    Raises
    ------
    FileNotFoundError
        If the input file or pattern does not exist
    ValueError
        If no files match the pattern
    """
    print("Input specified:", params["input"])
    
    if params["input"].endswith(".txt"):
        if os.path.exists(params["input"]):
            with open(params["input"], "r") as f:
                input_list = [line.strip() for line in f if line.strip()]
            if not input_list:
                raise ValueError(f"Text file {params['input']} is empty or contains no valid paths")
        else:
            raise FileNotFoundError(f"Input text file not found: {params['input']}")
    else:
        input_list = sorted(glob(params["input"].rstrip()))
        if not input_list:
            raise ValueError(f"No files found matching pattern: {params['input']}")
    
    print(f"Found {len(input_list)} file(s) to process")
    return input_list


def get_input_type(params):
    """
    Check if input is an image, numpy array, or whole slide image, and return the input type
    If you are trying to process other images that are supported by opencv (e.g. tiff), you can add the extension to the list

    Parameters
    ----------
    params: dict
        input parameters from argparse
    """
    params["ext"] = os.path.splitext(params["p"])[-1]
    if params["ext"] == ".npy":
        params["input_type"] = "npy"
    elif params["ext"] in [".jpg", ".png", ".jpeg", ".bmp"]:
        params["input_type"] = "img"
    else:
        params["input_type"] = "wsi"
    return params


def infer(params: dict):
    """
    Start nuclei segmentation and classification pipeline using specified parameters from argparse

    Parameters
    ----------
    params: dict
        input parameters from argparse
    
    Raises
    ------
    ValueError
        If required parameters are invalid
    """
    
    # Validate checkpoint parameter
    if not params["cp"]:
        raise ValueError("Checkpoint parameter (--cp) is required. Please specify a model checkpoint.")
    
    # Validate metric
    if params["metric"] not in ["mpq", "f1", "pannuke"]:
        print(f"Warning: Invalid metric '{params['metric']}', falling back to 'f1'")
        params["metric"] = "f1"
    else:
        print(f"Optimizing postprocessing for: {params['metric']}")

    params["data_dirs"] = params["cp"].split(",")
    
    # Create output directory if it doesn't exist
    os.makedirs(params["output_dir"], exist_ok=True)
    print(f"Results will be saved to: {params['output_dir']}")
    print(f"Loading model from: {params['data_dirs']}")

    # Run per tile inference and store results
    params, models, augmenter, color_aug_fn = get_inference_setup(params)

    input_list = prepare_input(params)
    print("Running inference on", len(input_list), "file(s)")

    for inp in input_list:
        start_time = timer()
        params["p"] = inp.rstrip()
        params = get_input_type(params)
        print("Processing ", params["p"])
        if params["cache"] is not None:
            print("Caching input at:")
            params["p"] = copy_img(params["p"], params["cache"])
            print(params["p"])

        params, z = inference_main(params, models, augmenter, color_aug_fn)
        print(
            "::: finished or skipped inference after",
            timedelta(seconds=timer() - start_time),
        )
        process_timer = timer()
        if params["only_inference"]:
            try:
                z[0].store.close()
                z[1].store.close()
            except TypeError:
                # if z is None, z cannot be indexed -> throws a TypeError
                pass
            print("Exiting after inference")
            sys.exit(2)
        # Stitch tiles together and postprocess to get instance segmentation
        if not os.path.exists(os.path.join(params["output_dir"], "pinst_pp.zip")):
            print("running post-processing")

            z_pp = post_process_main(
                params,
                z,
            )
            if not params["keep_raw"]:
                try:
                    os.remove(params["model_out_p"] + "_inst.zip")
                    os.remove(params["model_out_p"] + "_cls.zip")
                except FileNotFoundError:
                    pass
        else:
            z_pp = None
        print(
            "::: postprocessing took",
            timedelta(seconds=timer() - process_timer),
            "total elapsed time",
            timedelta(seconds=timer() - start_time),
        )
        if z_pp is not None:
            z_pp.store.close()
    print("done")


def main():
    """
    Main entry point for HoVer-NeXt inference pipeline
    """
    print("=" * 80)
    print("HoVer-NeXt Nuclei Segmentation and Classification Pipeline")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Inference will be very slow on CPU.")
        print("Please ensure you have a GPU and CUDA installed for optimal performance.")

    parser = argparse.ArgumentParser(
        description="HoVer-NeXt: Fast Nuclei Segmentation and Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single WSI file
  python3 main.py --input sample.svs --output_dir results/ --cp lizard_convnextv2_large --tta 4

  # Process multiple files using a glob pattern
  python3 main.py --input "/path/to/slides/*.svs" --output_dir results/ --cp lizard_convnextv2_large

  # Process files listed in a text file
  python3 main.py --input file_list.txt --output_dir results/ --cp pannuke_convnextv2_tiny_1

For more information, visit: https://github.com/pathology-data-mining/hover_next_inference
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to WSI/image/npy file, glob pattern (e.g., '/path/*.svs'), or text file containing paths",
        required=True,
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Output directory where results will be saved", 
        required=True
    )
    parser.add_argument(
        "--cp",
        type=str,
        default=None,
        help="Model checkpoint ID (e.g., 'lizard_convnextv2_large') or comma-separated list for ensemble",
        required=True,
    )
    parser.add_argument(
        "--only_inference",
        action="store_true",
        help="Only run inference step (useful for splitting GPU/CPU work on clusters)",
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        default="f1", 
        help="Metric to optimize post-processing for: 'f1', 'mpq', or 'pannuke' (default: f1)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64, 
        help="Batch size for inference (default: 64)"
    )
    parser.add_argument(
        "--tta",
        type=int,
        default=4,
        help="Number of test-time augmentation views (default: 4, use 4 for robust results)",
    )
    parser.add_argument(
        "--save_polygon",
        action="store_true",
        help="Save output as polygon GeoJSON files for QuPath",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=256,
        help="Tile size in pixels (default: 256, models are trained on 256x256)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.96875,
        help="Overlap between tiles as a fraction (default: 0.96875 for 0.5mpp, use 0.9375 for 0.25mpp)",
    )
    parser.add_argument(
        "--inf_workers",
        type=int,
        default=4,
        help="Number of workers for inference dataloader (default: 4, set to number of CPU cores for best performance)",
    )
    parser.add_argument(
        "--inf_writers",
        type=int,
        default=2,
        help="Number of writers for inference dataloader (default: 2, tune based on core availability)",
    )
    parser.add_argument(
        "--pp_tiling",
        type=int,
        default=8,
        help="Tiling factor for post-processing (default: 8, increase if running out of memory)",
    )
    parser.add_argument(
        "--pp_overlap",
        type=int,
        default=256,
        help="Overlap for post-processing tiles in pixels (default: 256, set to around tile_size)",
    )
    parser.add_argument(
        "--pp_workers",
        type=int,
        default=16,
        help="Number of workers for post-processing (default: 16, set to number of CPU cores)",
    )
    parser.add_argument(
        "--keep_raw",
        action="store_true",
        help="Keep raw prediction files (warning: can be large files, especially for PanNuke)",
    )
    parser.add_argument(
        "--cache", 
        type=str, 
        default=None, 
        help="Cache path for temporary files"
    )
    params = vars(parser.parse_args())
    
    try:
        infer(params)
        print("\n" + "=" * 80)
        print("Pipeline completed successfully!")
        print("=" * 80)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please check that your input files exist and paths are correct.")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please check your input parameters and try again.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
        print("If this error persists, please report it as an issue on GitHub.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

