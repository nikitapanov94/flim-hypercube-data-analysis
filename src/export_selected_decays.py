"""
export_selected_decays.py

Purpose
-------
Export raw and fitted decay curves for selected analysis bins.

This replaces export_pixel_fits.py.

The exported five-column format is preserved:

    Raw_time_ns
    Raw_counts
    Fit_time_ns
    Fit_intensity
    Fitted_curve
"""

from pathlib import Path

import numpy as np


def _pad_to_length(array, target_length):
    """
    Pad a one-dimensional array with NaNs to a target length.
    """

    array = np.asarray(array, dtype=float)

    padded = np.full(target_length, np.nan, dtype=float)
    padded[: len(array)] = array

    return padded


def export_selected_decays(
    analysis_result,
    output_folder,
    measurement_name,
):
    """
    Export selected analysis-bin decay curves.

    Parameters
    ----------
    analysis_result : dict
        Dictionary returned by generate_maps().

    output_folder : str or Path
        Folder where output files will be saved.

    measurement_name : str
        Measurement name used as filename prefix.

    Returns
    -------
    saved_files : dict
        Dictionary mapping selected-bin labels to saved file paths.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    selected_decays = analysis_result["selected_decays"]

    saved_files = {}

    for record in selected_decays:

        bin_y = int(record["bin_y"])
        bin_x = int(record["bin_x"])

        bin_label = f"bin_y{bin_y}_x{bin_x}"

        raw_time_ns = np.asarray(record["raw_time"], dtype=float) * 1e9
        raw_counts = np.asarray(record["raw_counts"], dtype=float)

        fit_time_ns = np.asarray(record["fit_time"], dtype=float) * 1e9
        fit_intensity = np.asarray(record["fit_intensity"], dtype=float)
        fitted_curve = np.asarray(record["fitted_curve"], dtype=float)

        n_rows = max(
            len(raw_time_ns),
            len(fit_time_ns),
        )

        export_table = np.column_stack(
            [
                _pad_to_length(raw_time_ns, n_rows),
                _pad_to_length(raw_counts, n_rows),
                _pad_to_length(fit_time_ns, n_rows),
                _pad_to_length(fit_intensity, n_rows),
                _pad_to_length(fitted_curve, n_rows),
            ]
        )

        output_path = (
            output_folder /
            f"{measurement_name}_{bin_label}_decay_fit.txt"
        )

        header_lines = [
            "Raw_time_ns\tRaw_counts\tFit_time_ns\tFit_intensity\tFitted_curve",
            f"Analysis_bin_y: {bin_y}",
            f"Analysis_bin_x: {bin_x}",
            f"Detector_y_start: {record['detector_y_start']}",
            f"Detector_y_stop_exclusive: {record['detector_y_stop']}",
            f"Detector_x_start: {record['detector_x_start']}",
            f"Detector_x_stop_exclusive: {record['detector_x_stop']}",
            f"Detector_pixels_pooled: {record['n_detector_pixels']}",
        ]

        fit = record.get("fit", None)

        if fit is not None:
            header_lines.append(f"Fit_successful: {fit.get('ok', False)}")

            if fit.get("ok", False):
                header_lines.append(f"tau_ns: {fit['tau'] * 1e9:.8f}")
                header_lines.append(f"beta: {fit['beta']:.8f}")
                header_lines.append(f"A: {fit['A']:.8f}")
                header_lines.append(f"C: {fit['C']:.8f}")
                header_lines.append(f"r2: {fit['r2']:.8f}")
                header_lines.append(f"rmse: {fit['rmse']:.8f}")
                header_lines.append(f"cutoff: {fit['cutoff']:.8f}")
                header_lines.append(f"fit_end_index: {fit['fit_end_index']}")
                header_lines.append(
                    f"fit_end_time_ns: {fit['fit_end_time'] * 1e9:.8f}"
                )
            else:
                header_lines.append(
                    f"Failure_reason: {fit.get('reason', 'unknown')}"
                )

        np.savetxt(
            output_path,
            export_table,
            delimiter="\t",
            fmt="%.8f",
            header="\n".join(header_lines),
            comments="# ",
        )

        saved_files[bin_label] = output_path

    return saved_files