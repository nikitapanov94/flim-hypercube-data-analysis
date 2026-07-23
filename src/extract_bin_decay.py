"""
extract_bin_decay.py

Purpose
-------
Extract one analysis-bin decay curve from a PTU lifetime histogram cube.

In Version 2, the analysis bin is the fundamental fitted spatial unit.
A detector pixel is simply the smallest possible analysis bin.

An analysis bin may contain:
    1 x 1 detector pixels
    2 x 2 detector pixels
    4 x 4 detector pixels
    ...
    the full rectangular ROI

The decay curve for an analysis bin is produced by summing all detector-pixel
decay histograms inside that bin.
"""

import numpy as np


def extract_bin_decay(
    ptu_data,
    roi,
    bin_size,
    bin_y,
    bin_x,
):
    """
    Extract the summed decay curve for one analysis bin.

    Parameters
    ----------
    ptu_data : dict
        Dictionary returned by read_ptu().

    roi : dict
        Rectangular region of interest in detector-pixel coordinates.

        Required keys:
            "y_start"
            "y_stop"
            "x_start"
            "x_stop"

        Stop coordinates are exclusive.

    bin_size : dict
        Size of one analysis bin in detector pixels.

        Required keys:
            "height"
            "width"

    bin_y : int
        Analysis-bin row index, zero-based.

    bin_x : int
        Analysis-bin column index, zero-based.

    Returns
    -------
    result : dict
        Dictionary containing the summed decay and spatial metadata.

        Keys:
            "decay"
            "detector_y_start"
            "detector_y_stop"
            "detector_x_start"
            "detector_x_stop"
            "n_detector_pixels"
    """

    cube = ptu_data["lifetime_histogram_cube"]

    image_height, image_width = cube.shape[:2]

    y_start = int(roi["y_start"])
    y_stop = int(roi["y_stop"])
    x_start = int(roi["x_start"])
    x_stop = int(roi["x_stop"])

    bin_height = int(bin_size["height"])
    bin_width = int(bin_size["width"])

    bin_y = int(bin_y)
    bin_x = int(bin_x)

    # -------------------------------------------------
    # Validate ROI
    # -------------------------------------------------

    if y_start < 0 or x_start < 0:
        raise ValueError("ROI start coordinates must be non-negative.")

    if y_stop > image_height:
        raise ValueError(
            f"ROI y_stop={y_stop} exceeds image height {image_height}."
        )

    if x_stop > image_width:
        raise ValueError(
            f"ROI x_stop={x_stop} exceeds image width {image_width}."
        )

    if y_stop <= y_start:
        raise ValueError("ROI y_stop must be greater than y_start.")

    if x_stop <= x_start:
        raise ValueError("ROI x_stop must be greater than x_start.")

    # -------------------------------------------------
    # Validate bin size
    # -------------------------------------------------

    if bin_height <= 0:
        raise ValueError("bin_size['height'] must be positive.")

    if bin_width <= 0:
        raise ValueError("bin_size['width'] must be positive.")

    roi_height = y_stop - y_start
    roi_width = x_stop - x_start

    if roi_height % bin_height != 0:
        raise ValueError(
            "ROI height must be exactly divisible by analysis-bin height."
        )

    if roi_width % bin_width != 0:
        raise ValueError(
            "ROI width must be exactly divisible by analysis-bin width."
        )

    output_height = roi_height // bin_height
    output_width = roi_width // bin_width

    if bin_y < 0 or bin_y >= output_height:
        raise ValueError(
            f"bin_y={bin_y} is outside the valid range 0 to {output_height - 1}."
        )

    if bin_x < 0 or bin_x >= output_width:
        raise ValueError(
            f"bin_x={bin_x} is outside the valid range 0 to {output_width - 1}."
        )

    # -------------------------------------------------
    # Convert analysis-bin coordinates to detector-pixel
    # coordinates in the original PTU image.
    # -------------------------------------------------

    detector_y_start = y_start + bin_y * bin_height
    detector_y_stop = detector_y_start + bin_height

    detector_x_start = x_start + bin_x * bin_width
    detector_x_stop = detector_x_start + bin_width

    # -------------------------------------------------
    # Sum all detector-pixel histograms inside this bin.
    # -------------------------------------------------

    decay = np.sum(
        cube[
            detector_y_start:detector_y_stop,
            detector_x_start:detector_x_stop,
            :,
        ],
        axis=(0, 1),
    )

    result = {
        "decay": decay,
        "detector_y_start": detector_y_start,
        "detector_y_stop": detector_y_stop,
        "detector_x_start": detector_x_start,
        "detector_x_stop": detector_x_stop,
        "n_detector_pixels": bin_height * bin_width,
    }

    return result