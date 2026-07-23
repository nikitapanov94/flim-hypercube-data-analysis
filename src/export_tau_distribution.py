"""
export_tau_distribution.py

Purpose
-------
Export the fitted lifetime values from all successfully fitted analysis bins.

Output
------
One text file containing one fitted lifetime value per line.
Only successful fits are included.
"""

from pathlib import Path

import numpy as np


def export_tau_distribution(
    analysis_result,
    output_folder,
    measurement_name,
):
    """
    Export the distribution of fitted lifetimes.

    Parameters
    ----------
    analysis_result : dict
        Dictionary returned by generate_maps().

    output_folder : str or Path
        Folder where the output file will be written.

    measurement_name : str
        Measurement name used as the filename prefix.

    Returns
    -------
    output_path : Path
        Path to the generated text file.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    maps = analysis_result["maps"]

    tau_map = maps["tau_ns"]
    success_map = maps["success"]

    tau_distribution = tau_map[success_map == 1]

    tau_distribution = tau_distribution[np.isfinite(tau_distribution)]

    tau_distribution = np.sort(tau_distribution)

    output_path = (
        output_folder /
        f"{measurement_name}_tau_distribution.txt"
    )

    np.savetxt(
        output_path,
        tau_distribution,
        fmt="%.6f",
        header=(
            "Fitted lifetime distribution\n"
            "Quantity : tau\n"
            "Units    : ns\n"
            "Source   : successful analysis bins"
        ),
        comments="# ",
    )

    return output_path