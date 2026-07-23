"""
generate_maps.py

Purpose
-------
Generate FLIM parameter maps by fitting one decay per analysis bin.

Version 2 architecture
----------------------
The analysis bin is the fundamental fitted spatial unit.

A detector pixel is simply the smallest possible analysis bin.

Examples:
    ROI 128 x 128, bin size 1 x 1       -> 128 x 128 maps
    ROI 128 x 128, bin size 2 x 2       -> 64 x 64 maps
    ROI 128 x 128, bin size 128 x 128   -> 1 x 1 maps

The fitting algorithm is unchanged. Each analysis bin is converted into one
summed decay curve, then passed through the same preprocessing and fitting
pipeline used in Version 1.
"""

import numpy as np

from analysis.extract_bin_decay import extract_bin_decay
from analysis.preprocess_decay import preprocess_decay
from analysis.fit_decay import fit_decay


def _validate_analysis_geometry(
    cube,
    roi,
    bin_size,
):
    """
    Validate ROI and analysis-bin size.
    """

    image_height, image_width = cube.shape[:2]

    y_start = int(roi["y_start"])
    y_stop = int(roi["y_stop"])
    x_start = int(roi["x_start"])
    x_stop = int(roi["x_stop"])

    bin_height = int(bin_size["height"])
    bin_width = int(bin_size["width"])

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

    geometry = {
        "image_height": image_height,
        "image_width": image_width,
        "roi_height": roi_height,
        "roi_width": roi_width,
        "bin_height": bin_height,
        "bin_width": bin_width,
        "output_height": output_height,
        "output_width": output_width,
        "total_bins": output_height * output_width,
    }

    return geometry


def _validate_selected_bins(
    selected_bins,
    output_height,
    output_width,
):
    """
    Validate selected analysis-bin coordinates.

    selected_bins coordinates are (bin_y, bin_x), zero-based.
    """

    if selected_bins is None:
        return []

    validated = []

    for bin_y, bin_x in selected_bins:

        bin_y = int(bin_y)
        bin_x = int(bin_x)

        if bin_y < 0 or bin_y >= output_height:
            raise ValueError(
                f"Selected bin_y={bin_y} is outside the valid range "
                f"0 to {output_height - 1}."
            )

        if bin_x < 0 or bin_x >= output_width:
            raise ValueError(
                f"Selected bin_x={bin_x} is outside the valid range "
                f"0 to {output_width - 1}."
            )

        validated.append((bin_y, bin_x))

    return validated


def _initialize_maps(
    output_height,
    output_width,
):
    """
    Initialize all output maps.

    Numerical maps are initialized with NaN so that failed or skipped bins do
    not appear as real zeros in plotting software.
    """

    shape = (output_height, output_width)

    maps = {
        "tau_ns": np.full(shape, np.nan, dtype=float),
        "beta": np.full(shape, np.nan, dtype=float),
        "A": np.full(shape, np.nan, dtype=float),
        "C": np.full(shape, np.nan, dtype=float),
        "r2": np.full(shape, np.nan, dtype=float),
        "rmse": np.full(shape, np.nan, dtype=float),
        "total_photons": np.full(shape, np.nan, dtype=float),
        "peak_counts": np.full(shape, np.nan, dtype=float),
        "snr": np.full(shape, np.nan, dtype=float),
        "fit_end_index": np.full(shape, np.nan, dtype=float),
        "fit_end_time_ns": np.full(shape, np.nan, dtype=float),
        "cutoff": np.full(shape, np.nan, dtype=float),

        # Final accepted-bin mask.
        # A bin is successful only if:
        #   1. photon threshold passes, if enabled
        #   2. fitting succeeds
        #   3. R2 threshold passes, if enabled
        "success": np.zeros(shape, dtype=int),

        # Diagnostic R2 acceptance map.
        # This is calculated for fitted bins regardless of whether
        # APPLY_R2_THRESHOLD is enabled in run_analysis.py.
        "r2_accepted": np.zeros(shape, dtype=int),
    }

    return maps


def _make_selected_decay_record(
    bin_y,
    bin_x,
    bin_decay,
    raw_time,
    preprocessed=None,
    fit=None,
):
    """
    Assemble one selected-bin decay record for optional export.
    """

    raw_counts = np.asarray(bin_decay["decay"], dtype=float)

    if preprocessed is not None and fit is not None and fit.get("ok", False):

        fit_end_index = int(fit["fit_end_index"])

        fit_time = np.asarray(fit["fit_time"], dtype=float)
        fit_intensity = np.asarray(
            preprocessed["intensity"][:fit_end_index],
            dtype=float,
        )
        fitted_curve = np.asarray(fit["fitted_curve"], dtype=float)

    else:

        fit_time = np.array([], dtype=float)
        fit_intensity = np.array([], dtype=float)
        fitted_curve = np.array([], dtype=float)

    record = {
        "bin_y": int(bin_y),
        "bin_x": int(bin_x),
        "detector_y_start": int(bin_decay["detector_y_start"]),
        "detector_y_stop": int(bin_decay["detector_y_stop"]),
        "detector_x_start": int(bin_decay["detector_x_start"]),
        "detector_x_stop": int(bin_decay["detector_x_stop"]),
        "n_detector_pixels": int(bin_decay["n_detector_pixels"]),
        "raw_time": np.asarray(raw_time, dtype=float),
        "raw_counts": raw_counts,
        "fit_time": fit_time,
        "fit_intensity": fit_intensity,
        "fitted_curve": fitted_curve,
        "fit": fit,
    }

    return record


def generate_maps(
    ptu_data,
    roi,
    bin_size,
    selected_bins=None,
    apply_photon_threshold=True,
    min_total_photons=100,
    apply_r2_threshold=False,
    min_r2=0.95,
    threshold_mode="noise",
    fit_amplitude=True,
    fit_background=True,
    background_fraction=0.05,
    trim_end_bins=5,
    verbose=True,
):
    """
    Generate fitted FLIM parameter maps for one PTU measurement.

    Parameters
    ----------
    ptu_data : dict
        Dictionary returned by read_ptu().

    roi : dict
        Rectangular ROI in detector-pixel coordinates.

    bin_size : dict
        Analysis-bin size in detector pixels.

    selected_bins : list of tuple, optional
        Analysis-bin coordinates to retain for decay export.

    apply_photon_threshold : bool
        If True, skip analysis bins whose summed photon count is below
        min_total_photons before fitting.

    min_total_photons : float
        Minimum total photon count required for an analysis bin to be fitted
        when apply_photon_threshold is True.

    apply_r2_threshold : bool
        If True, only fits with R2 >= min_r2 are marked successful.

    min_r2 : float
        Minimum accepted R2 value when apply_r2_threshold is True.

    threshold_mode : str
        Fit-window threshold mode passed to fit_decay().

    fit_amplitude : bool
        If True, fit A.

    fit_background : bool
        If True, fit C.

    background_fraction : float
        Fraction of the trimmed decay used to estimate background.

    trim_end_bins : int
        Number of final raw TCSPC bins removed before preprocessing.

    verbose : bool
        If True, print progress information.

    Returns
    -------
    analysis_result : dict
        Dictionary containing maps, selected decay records, and metadata.
    """

    cube = ptu_data["lifetime_histogram_cube"]
    time_axis = ptu_data["time_axis"]

    geometry = _validate_analysis_geometry(
        cube=cube,
        roi=roi,
        bin_size=bin_size,
    )

    selected_bins = _validate_selected_bins(
        selected_bins=selected_bins,
        output_height=geometry["output_height"],
        output_width=geometry["output_width"],
    )

    selected_bin_set = set(selected_bins)

    maps = _initialize_maps(
        output_height=geometry["output_height"],
        output_width=geometry["output_width"],
    )

    selected_decay_records = []

    # -------------------------------------------------
    # Loop over analysis bins
    # -------------------------------------------------

    bin_counter = 0

    for bin_y in range(geometry["output_height"]):
        for bin_x in range(geometry["output_width"]):

            bin_counter += 1

            if verbose and bin_counter % 100 == 0:
                print(
                    f"Processing analysis bin "
                    f"{bin_counter} / {geometry['total_bins']}"
                )

            # -----------------------------------------
            # Extract summed decay for this bin
            # -----------------------------------------

            bin_decay = extract_bin_decay(
                ptu_data=ptu_data,
                roi=roi,
                bin_size=bin_size,
                bin_y=bin_y,
                bin_x=bin_x,
            )

            decay = bin_decay["decay"]

            total_photons = float(np.sum(decay))
            peak_counts = float(np.max(decay))

            maps["total_photons"][bin_y, bin_x] = total_photons
            maps["peak_counts"][bin_y, bin_x] = peak_counts

            is_selected = (bin_y, bin_x) in selected_bin_set

            # -----------------------------------------
            # Optional pre-fit photon threshold
            # -----------------------------------------

            if apply_photon_threshold and total_photons < min_total_photons:

                fit = {
                    "ok": False,
                    "accepted": False,
                    "reason": "below_min_total_photons",
                }

                if is_selected:
                    selected_decay_records.append(
                        _make_selected_decay_record(
                            bin_y=bin_y,
                            bin_x=bin_x,
                            bin_decay=bin_decay,
                            raw_time=time_axis,
                            preprocessed=None,
                            fit=fit,
                        )
                    )

                continue

            # -----------------------------------------
            # Preprocess and fit
            # -----------------------------------------

            try:

                preprocessed = preprocess_decay(
                    decay,
                    time_axis=time_axis,
                    background_fraction=background_fraction,
                    trim_end_bins=trim_end_bins,
                )

                # -----------------------------------------
                # Calculate signal-to-noise ratio (SNR)
                #
                # Signal-to-noise ratio (SNR) is defined as
                #
                #     SNR = peak_counts / baseline_std
                #
                # where baseline_std is the standard deviation of the estimated
                # baseline before normalization.
                #
                # Numerically, this is equivalent to
                #
                #     SNR = 1 / normalized_baseline_std
                #
                # because preprocessing normalizes the decay to a peak intensity of 1.
                # -----------------------------------------

                background_std = preprocessed["stats"]["background_std"]
                peak = preprocessed["stats"]["peak"]

                if (
                    np.isfinite(background_std)
                    and background_std > 0
                    and np.isfinite(peak)
                ):
                    maps["snr"][bin_y, bin_x] = peak / background_std

                fit = fit_decay(
                    preprocessed["time"],
                    preprocessed["intensity"],
                    stats=preprocessed["stats"],
                    threshold_mode=threshold_mode,
                    fit_amplitude=fit_amplitude,
                    fit_background=fit_background,
                )

            except Exception as error:

                fit = {
                    "ok": False,
                    "accepted": False,
                    "reason": str(error),
                }
                preprocessed = None

            if not fit.get("ok", False):

                if is_selected:
                    selected_decay_records.append(
                        _make_selected_decay_record(
                            bin_y=bin_y,
                            bin_x=bin_x,
                            bin_decay=bin_decay,
                            raw_time=time_axis,
                            preprocessed=preprocessed,
                            fit=fit,
                        )
                    )

                continue

            # -----------------------------------------
            # Store raw fit results in maps
            # -----------------------------------------
            #
            # R2 is stored even if the bin later fails the R2 acceptance
            # threshold. This makes r2_map and r2_accepted_map useful
            # diagnostic outputs.
            # -----------------------------------------

            r2_value = float(fit["r2"])

            maps["tau_ns"][bin_y, bin_x] = fit["tau"] * 1e9
            maps["beta"][bin_y, bin_x] = fit["beta"]
            maps["A"][bin_y, bin_x] = fit["A"]
            maps["C"][bin_y, bin_x] = fit["C"]

            maps["r2"][bin_y, bin_x] = r2_value
            maps["rmse"][bin_y, bin_x] = fit["rmse"]

            maps["fit_end_index"][bin_y, bin_x] = fit["fit_end_index"]
            maps["fit_end_time_ns"][bin_y, bin_x] = fit["fit_end_time"] * 1e9
            maps["cutoff"][bin_y, bin_x] = fit["cutoff"]

            # -----------------------------------------
            # R2 acceptance
            # -----------------------------------------

            if np.isfinite(r2_value) and r2_value >= min_r2:
                r2_accepted = True
                maps["r2_accepted"][bin_y, bin_x] = 1
            else:
                r2_accepted = False
                maps["r2_accepted"][bin_y, bin_x] = 0

            fit["r2_accepted"] = bool(r2_accepted)

            if apply_r2_threshold and not r2_accepted:
                fit["accepted"] = False
                fit["reason"] = "below_min_r2"

                if is_selected:
                    selected_decay_records.append(
                        _make_selected_decay_record(
                            bin_y=bin_y,
                            bin_x=bin_x,
                            bin_decay=bin_decay,
                            raw_time=time_axis,
                            preprocessed=preprocessed,
                            fit=fit,
                        )
                    )

                continue

            # -----------------------------------------
            # Final accepted fit
            # -----------------------------------------

            fit["accepted"] = True
            maps["success"][bin_y, bin_x] = 1

            if is_selected:
                selected_decay_records.append(
                    _make_selected_decay_record(
                        bin_y=bin_y,
                        bin_x=bin_x,
                        bin_decay=bin_decay,
                        raw_time=time_axis,
                        preprocessed=preprocessed,
                        fit=fit,
                    )
                )

    # -------------------------------------------------
    # Collect complete analysis result
    # -------------------------------------------------

    n_success = int(np.sum(maps["success"]))
    n_r2_accepted = int(np.sum(maps["r2_accepted"]))

    analysis_result = {
        "maps": maps,
        "selected_decays": selected_decay_records,
        "roi": dict(roi),
        "bin_size": dict(bin_size),
        "geometry": geometry,
        "settings": {
            "apply_photon_threshold": apply_photon_threshold,
            "min_total_photons": min_total_photons,
            "apply_r2_threshold": apply_r2_threshold,
            "min_r2": min_r2,
            "threshold_mode": threshold_mode,
            "fit_amplitude": fit_amplitude,
            "fit_background": fit_background,
            "background_fraction": background_fraction,
            "trim_end_bins": trim_end_bins,
            "selected_bins": selected_bins,
        },
    }

    if verbose:
        print()
        print("Map generation complete.")
        print(
            f"Detector image size: "
            f"{geometry['image_width']} x {geometry['image_height']}"
        )
        print(
            f"ROI size: "
            f"{geometry['roi_width']} x {geometry['roi_height']} detector pixels"
        )
        print(
            f"Analysis-bin size: "
            f"{geometry['bin_width']} x {geometry['bin_height']} detector pixels"
        )
        print(
            f"Output map size: "
            f"{geometry['output_width']} x {geometry['output_height']} analysis bins"
        )
        print(f"Analysis bins processed: {geometry['total_bins']}")
        print(f"R2-accepted fits: {n_r2_accepted}")
        print(f"Successful fits: {n_success}")
        print(f"Failed/skipped fits: {geometry['total_bins'] - n_success}")

    return analysis_result