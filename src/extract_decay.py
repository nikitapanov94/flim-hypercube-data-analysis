"""
extract_decay.py

Purpose
-------
Extract fluorescence decay curves from ptu_data.

Input
-----
ptu_data returned by read_ptu.py, plus pixel coordinates.

Output
------
decay_curve
"""

import numpy as np


def extract_decay(ptu_data, x, y):
    """
    Extract the lifetime decay curve from one pixel.

    Parameters
    ----------
    ptu_data : dict
        Output from read_ptu().
    x : int
        Pixel column.
    y : int
        Pixel row.

    Returns
    -------
    decay_curve : np.ndarray
        One-dimensional fluorescence decay curve.
    """

    lifetime_histogram_cube = ptu_data["lifetime_histogram_cube"]

    height, width = lifetime_histogram_cube.shape[:2]

    if x < 0 or x >= width:
        raise ValueError(f"x={x} is outside the valid range 0 to {width - 1}")

    if y < 0 or y >= height:
        raise ValueError(f"y={y} is outside the valid range 0 to {height - 1}")

    decay_curve = lifetime_histogram_cube[y, x, :]

    return decay_curve