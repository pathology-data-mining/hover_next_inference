"""
Visualization and export utilities for inference results.

This module provides functions to convert instance segmentation results into various
formats for visualization and analysis in external tools like QuPath.

Key Features
------------
- GeoJSON export for QuPath polygon visualization
- TSV export for QuPath centroid detection import
- Polygon extraction from instance masks
- Automatic color assignment by class
- Coordinate transformation for different resolutions

Main Functions
--------------
create_geojson : Create GeoJSON file with nucleus polygons
create_tsvs : Create TSV files with nucleus centroids
cont : Extract polygon contours from binary masks
create_polygon_output : Generate polygon outputs for visualization

Supported Formats
-----------------
- GeoJSON: Full polygon geometries with classifications
- TSV: Centroid coordinates with class labels
- Compatible with QuPath v0.3+ for direct import

Examples
--------
>>> from inference.viz_utils import create_geojson
>>> create_geojson(polygons, class_ids, CLASS_LABELS_LIZARD, params)
>>> # Creates poly.geojson file in output directory
"""
import os
import numpy as np
import geojson
import openslide
import cv2
from skimage.measure import regionprops
from inference.constants import (
    CLASS_LABELS_LIZARD,
    CLASS_LABELS_PANNUKE,
    COLORS_LIZARD,
    COLORS_PANNUKE,
    CONIC_MPP,
    PANNUKE_MPP,
)


def create_geojson(polygons, classids, lookup, params):
    """
    Create a GeoJSON file from nucleus polygons and classifications.
    
    This function converts instance segmentation results into a GeoJSON format
    that can be imported into QuPath or other visualization tools.
    
    Parameters
    ----------
    polygons : list
        List of polygon coordinates for each nucleus
    classids : list
        List of class IDs corresponding to each polygon
    lookup : dict
        Dictionary mapping class IDs to class names
    params : dict
        Parameter dictionary containing:
        - 'pannuke': boolean indicating if using PanNuke classes
        - 'ds_factor': downsampling factor for coordinate conversion
        - 'output_dir': directory to save the GeoJSON file
    
    Returns
    -------
    None
        Writes GeoJSON file to params['output_dir']/poly.geojson
    
    Notes
    -----
    Invalid polygons are automatically skipped with a warning message.
    Colors are assigned based on the model type (PanNuke or Lizard).
    """
    features = []
    colors = COLORS_PANNUKE if params["pannuke"] else COLORS_LIZARD 
    if isinstance(classids[0], (list, tuple)):
        classids = [cid[0] for cid in classids]
    for i, (poly, cid) in enumerate(zip(polygons, classids)):
        poly = np.array(poly)
        poly = poly[:, [1, 0]] * params["ds_factor"]
        poly = poly.tolist()
            
        geom = geojson.Polygon([poly], precision=2)
        if not geom.is_valid:
            print(f"Polygon {i}:{[poly]} is not valid, skipping...")
            continue
        # poly.append(poly[0])
        measurements = {classifications: 0 for classifications in lookup.values()}
        measurements[lookup[cid]] = 1
        feature = geojson.Feature(
            geometry=geojson.Polygon([poly], precision=2),
            properties={
                "Name": f"Nuc {i}",
                "Type": "Polygon",
                "color": colors[cid - 1],
                "classification": lookup[cid],
                "measurements": measurements,
                "objectType": "tile"
            },
        )
        features.append(feature)
    feature_collection = geojson.FeatureCollection(features)
    with open(params["output_dir"] + "/poly.geojson", "w") as outfile:
        geojson.dump(feature_collection, outfile)


def create_tsvs(pcls_out, params):
    """
    Create TSV files for QuPath import with nucleus centroids and classifications.
    
    Generates one TSV file per class type with centroid coordinates in image space.
    These files can be directly imported into QuPath for visualization and analysis.
    
    Parameters
    ----------
    pcls_out : dict
        Dictionary mapping instance IDs to (class_id, centroid) tuples
    params : dict
        Parameter dictionary containing:
        - 'pannuke': boolean indicating if using PanNuke classes
        - 'ds_factor': downsampling factor for coordinate conversion
        - 'output_dir': directory to save TSV files
    
    Returns
    -------
    None
        Writes one TSV file per class to params['output_dir']/pred_<class_name>.tsv
    
    Notes
    -----
    TSV format: x, y, name, color (tab-separated)
    Coordinates are scaled by ds_factor to match original image resolution.
    """
    pred_keys = CLASS_LABELS_PANNUKE if params["pannuke"] else CLASS_LABELS_LIZARD

    coord_array = np.array([[i[0], *i[1]] for i in pcls_out.values()])
    classes = list(pred_keys.keys())
    colors = ["-256", "-65536"]
    i = 0
    for pt in classes:
        file = os.path.join(params["output_dir"], "pred_" + pt + ".tsv")
        textfile = open(file, "w")

        textfile.write("x" + "\t" + "y" + "\t" + "name" + "\t" + "color" + "\n")
        textfile.writelines(
            [
                str(element[2] * params["ds_factor"])
                + "\t"
                + str(element[1] * params["ds_factor"])
                + "\t"
                + pt
                + "\t"
                + colors[0]
                + "\n"
                for element in coord_array[coord_array[:, 0] == pred_keys[pt]]
            ]
        )

        textfile.close()
        i += 1


def cont(x, offset=None):
    """
    Extract contour polygon from a binary nucleus mask.
    
    Uses OpenCV contour detection with TC89_KCOS chain approximation for efficient
    polygon representation. Handles edge cases like single-pixel detections.
    
    Parameters
    ----------
    x : tuple
        Tuple containing (label_id, binary_mask, bounding_box)
    offset : list or tuple, optional
        [y_offset, x_offset] to add to contour coordinates for global positioning
    
    Returns
    -------
    np.ndarray
        Contour coordinates as (N, 2) array of [x, y] points
    
    Notes
    -----
    For single-pixel nuclei, the mask is upscaled 2x before contour detection
    to ensure valid contour extraction.
    """
    _, im, bb = x
    im = np.pad(im.astype(np.uint8), 1, mode="constant", constant_values=0)

    # initial contour finding
    cont = cv2.findContours(
        im,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_TC89_KCOS,
    )[0][0].reshape(-1, 2)[:, [1, 0]]
    # since opencv does not do "pixel" contours, we artificially do this for single pixel detections (if they exist)
    if cont.shape[0] <= 1:
        im = cv2.resize(im, None, fx=2.0, fy=2.0)
        cont = (
            cv2.findContours(
                im,
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_TC89_KCOS,
            )[0][0].reshape(-1, 2)[:, [1, 0]]
            / 2.0
        )
    if offset is not None:
        cont = (cont + offset + bb[0:2] - 1).tolist()
    else:
        cont = (cont + bb[0:2] - 1).tolist()
    # close polygon:
    if cont[0] != cont[-1]:
        cont.append(cont[0])
    return cont


def create_polygon_output(pinst, pcls_out, params):
    # polygon output is slow and unwieldy, TODO
    pred_keys = CLASS_LABELS_PANNUKE if params["pannuke"] else CLASS_LABELS_LIZARD
    # whole slide regionprops could be avoided to speed up this process...
    print("getting all detections...")
    props = [(p.label, p.image, p.bbox) for p in regionprops(np.asarray(pinst))]
    class_labels = [pcls_out[str(p[0])] for p in props]
    print("generating contours...")
    res_poly = [cont(i) for i in props]
    print("creating output...")
    create_geojson(
        res_poly,
        class_labels,
        dict((v, k) for k, v in pred_keys.items()),
        params,
    )
