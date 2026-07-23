"""
save_maps.py

Purpose
-------
Save FLIM parameter maps to disk.

Outputs
-------
1. One tab-delimited TXT matrix per map, suitable for Origin.
2. One NPZ archive containing all maps and Version 2 analysis metadata.
3. One human-readable analysis report.
"""

from pathlib import Path

import numpy as np


def save_maps(
    analysis_result,
    settings,
    metadata,
    output_folder,
    measurement_name,
):
    """
    Save generated FLIM maps.

    Parameters
    ----------
    analysis_result : dict
        Dictionary returned by generate_maps().

    settings : dict
        Dictionary containing analysis settings assembled in run_analysis.py.

    metadata : dict
        Metadata dictionary from ptu_data["metadata"].

    output_folder : str or Path
        Folder where maps will be saved.

    measurement_name : str
        Base name used for output files.

    Returns
    -------
    saved_files : dict
        Dictionary mapping output names to saved file paths.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    maps = analysis_result["maps"]
    roi = analysis_result["roi"]
    bin_size = analysis_result["bin_size"]
    geometry = analysis_result["geometry"]
    analysis_settings = analysis_result["settings"]

    saved_files = {}

    # ---------------------------------------------------
    # Save individual Origin-friendly TXT maps
    # ---------------------------------------------------

    for map_name, array in maps.items():

        txt_path = output_folder / f"{measurement_name}_{map_name}_map.txt"

        np.savetxt(
            txt_path,
            array,
            delimiter="\t",
            fmt="%.8f",
        )

        saved_files[map_name] = txt_path

    # ---------------------------------------------------
    # Save complete Python archive
    # ---------------------------------------------------

    npz_path = output_folder / f"{measurement_name}_maps.npz"

    np.savez_compressed(
        npz_path,
        **maps,
        roi=roi,
        bin_size=bin_size,
        geometry=geometry,
        analysis_settings=analysis_settings,
        run_settings=settings,
    )

    saved_files["npz"] = npz_path

    # ---------------------------------------------------
    # Prepare analysis summary values
    # ---------------------------------------------------

    success_map = maps["success"]

    output_height, output_width = success_map.shape

    total_bins = output_height * output_width
    successful_fits = int(np.sum(success_map))
    failed_fits = total_bins - successful_fits

    # ---------------------------------------------------
    # Save analysis report
    # ---------------------------------------------------

    report_path = (
        output_folder /
        f"{measurement_name}_analysis_report.txt"
    )

    with open(report_path, "w") as f:

        f.write("========================================\n")
        f.write("FLIM ANALYSIS REPORT\n")
        f.write("========================================\n\n")

        # -----------------------------------------------
        # Measurement information
        # -----------------------------------------------

        f.write("Measurement Information\n")
        f.write("-----------------------\n")

        f.write(f"PTU file           : {settings['PTU_file']}\n")
        f.write(f"Measurement name   : {measurement_name}\n")

        f.write(
            f"Measurement type   : "
            f"{metadata.get('measurement_type', 'not available')}\n"
        )

        f.write(
            f"Detector image size: "
            f"{metadata['image_width']} x "
            f"{metadata['image_height']} pixels\n"
        )

        f.write(
            f"TCSPC bins         : "
            f"{metadata['tcspc_bins']}\n"
        )

        f.write(
            f"TCSPC resolution   : "
            f"{metadata['tcspc_resolution'] * 1e9:.2f} ns\n"
        )

        f.write(
            f"Acquisition time   : "
            f"{metadata['acquisition_time']:.2f} s\n"
        )

        f.write(
            f"Detected photons   : "
            f"{metadata['detected_photons']:,}\n"
        )

        f.write(
            f"Histogram shape    : "
            f"{metadata['histogram_shape']}\n"
        )

        f.write("\n")

        f.write(
            f"Analysis date      : "
            f"{settings['analysis_date']}\n"
        )

        f.write(
            f"Software version   : "
            f"{settings['software_version']}\n\n"
        )

        # -----------------------------------------------
        # Version 2 spatial analysis settings
        # -----------------------------------------------

        f.write("Spatial Analysis Settings\n")
        f.write("-------------------------\n")

        f.write("Coordinate convention: zero-based\n")
        f.write("ROI stop coordinates: exclusive\n\n")

        f.write(f"ROI y_start       : {roi['y_start']}\n")
        f.write(f"ROI y_stop        : {roi['y_stop']}\n")
        f.write(f"ROI x_start       : {roi['x_start']}\n")
        f.write(f"ROI x_stop        : {roi['x_stop']}\n")

        f.write(
            f"ROI size          : "
            f"{geometry['roi_width']} x {geometry['roi_height']} "
            f"detector pixels\n"
        )

        f.write(
            f"Analysis-bin size : "
            f"{bin_size['width']} x {bin_size['height']} "
            f"detector pixels\n"
        )

        f.write(
            f"Output map size   : "
            f"{geometry['output_width']} x {geometry['output_height']} "
            f"analysis bins\n\n"
        )

        # -----------------------------------------------
        # Fitting settings
        # -----------------------------------------------

        f.write("Fit Settings\n")
        f.write("------------\n")

        f.write(
            f"Minimum photons    : "
            f"{analysis_settings['min_total_photons']}\n"
        )

        f.write(
            f"Trim end bins      : "
            f"{analysis_settings['trim_end_bins']}\n"
        )

        f.write(
            f"Background fraction: "
            f"{analysis_settings['background_fraction']}\n"
        )

        f.write(
            f"Threshold mode     : "
            f"{analysis_settings['threshold_mode']}\n"
        )

        f.write(
            f"Fit amplitude      : "
            f"{analysis_settings['fit_amplitude']}\n"
        )

        f.write(
            f"Fit background     : "
            f"{analysis_settings['fit_background']}\n\n"
        )

        # -----------------------------------------------
        # Optional exports
        # -----------------------------------------------

        f.write("Optional Exports\n")
        f.write("----------------\n")

        f.write(
            f"Tau distribution export : "
            f"{settings['run_tau_distribution_export']}\n"
        )

        f.write(
            f"Selected decay export   : "
            f"{settings['run_selected_decay_export']}\n"
        )

        f.write(
            f"Selected bins           : "
            f"{analysis_settings['selected_bins']}\n\n"
        )

        # -----------------------------------------------
        # Analysis results
        # -----------------------------------------------

        f.write("Analysis Results\n")
        f.write("----------------\n")

        f.write(f"Analysis-bin width : {output_width}\n")
        f.write(f"Analysis-bin height: {output_height}\n")
        f.write(f"Bins analysed      : {total_bins}\n")
        f.write(f"Successful fits    : {successful_fits}\n")
        f.write(f"Failed/skipped fits: {failed_fits}\n")

    saved_files["analysis_report"] = report_path

    return saved_files