"""
export_snr_vs_photons.py

Purpose
-------
Export the relationship between total photon counts and signal-to-noise ratio
(SNR) for every valid analysis bin.

The output is intended for downstream plotting and fitting in external
software such as Origin.

The exported table contains two columns:

    Total_photons
    SNR

Rows are sorted by increasing total photon count.
"""

from pathlib import Path

import numpy as np


def export_snr_vs_photons(
    analysis_result,
    output_folder,
    measurement_name,
):
    """
    Export total photon count versus signal-to-noise ratio.

    Parameters
    ----------
    analysis_result : dict
        Dictionary returned by generate_maps().

    output_folder : pathlib.Path or str
        Output directory.

    measurement_name : str
        Measurement name used as filename prefix.
    """

    maps = analysis_result["maps"]

    total_photons = maps["total_photons"].ravel()
    snr = maps["snr"].ravel()

    valid = (
        np.isfinite(total_photons)
        & np.isfinite(snr)
    )

    total_photons = total_photons[valid]
    snr = snr[valid]

    sort_index = np.argsort(total_photons)

    total_photons = total_photons[sort_index]
    snr = snr[sort_index]

    export_data = np.column_stack(
        (
            total_photons,
            snr,
        )
    )

    output_path = (
        Path(output_folder)
        / f"{measurement_name}_snr_vs_total_photons.txt"
    )

    np.savetxt(
        output_path,
        export_data,
        delimiter="\t",
        fmt="%.6f",
        header="Total_photons\tSNR",
        comments="",
    )