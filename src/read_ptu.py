"""
=========================================================
read_ptu.py
=========================================================

Purpose
-------
Read a PicoQuant PTU file and convert it into a Python
representation that can be used by the rest of the project.

Input
-----
Path to a PTU file.

Output
------
ptu_data

Current structure
-----------------
ptu_data = {
    "filename": ...,
    "metadata": ...,
    "lifetime_histogram_cube": ...,
    "time_axis": ...
}
"""

from pathlib import Path

import numpy as np
import ptufile


def read_ptu(file_path):
    """
    Read one PTU file.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    ptu_data : dict
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find file:\n{file_path}")

    print(f"\nReading: {file_path.name}")

    with ptufile.PtuFile(file_path) as ptu:

        # -------------------------------------------------
        # Lifetime histogram cube
        # -------------------------------------------------

        raw_array = ptu.decode_image()

        lifetime_histogram_cube = np.squeeze(raw_array)

        # -------------------------------------------------
        # Lifetime axis
        # -------------------------------------------------

        time_axis = np.asarray(ptu.coords["H"])

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        metadata = {
            # Acquisition

            "acquisition_time": ptu.acquisition_time,

            "tcspc_resolution": ptu.tcspc_resolution,

            # Image information
            "image_height": lifetime_histogram_cube.shape[0],

            "image_width": lifetime_histogram_cube.shape[1],

            "tcspc_bins": lifetime_histogram_cube.shape[2],

            "histogram_shape": lifetime_histogram_cube.shape,

            # Photon statistics
            "detected_photons": int(
                np.sum(lifetime_histogram_cube)
            ),
        }

    # -----------------------------------------------------
    # Python representation of the PTU file
    # -----------------------------------------------------

    ptu_data = {
        "filename": file_path.name,
        "metadata": metadata,
        "lifetime_histogram_cube": lifetime_histogram_cube,
        "time_axis": time_axis,
    }

    return ptu_data