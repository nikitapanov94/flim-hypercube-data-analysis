"""
preprocess_decay.py

Purpose
-------
Preprocess a single fluorescence decay curve.

Operations
----------
1. Optionally trim a selected number of final raw time bins.
2. Estimate the background from the last fraction of the remaining decay.
3. Subtract the background.
4. Normalize the decay.
5. Trim the curve so that time starts at the decay maximum.

Input
-----
decay_curve
time_axis

Output
------
preprocessed = {
    "time": ...,
    "intensity": ...,
    "stats": ...
}
"""

import numpy as np


DEFAULT_BACKGROUND_FRACTION = 0.05
DEFAULT_TRIM_END_BINS = 5
I1_TOLERANCE = 1e-12


def preprocess_decay(
    decay_curve,
    time_axis=None,
    background_fraction=DEFAULT_BACKGROUND_FRACTION,
    trim_end_bins=DEFAULT_TRIM_END_BINS,
    i1_tolerance=I1_TOLERANCE,
):
    """
    Preprocess one fluorescence decay.

    Parameters
    ----------
    decay_curve : array-like
        Raw fluorescence decay.

    time_axis : array-like, optional
        Physical time corresponding to each histogram bin.
        If None, a simple index axis is generated.

    background_fraction : float
        Fraction of the decay used to estimate the background
        after optional end-bin trimming.
        Default = 0.07, meaning the last 7% of the remaining decay.

    trim_end_bins : int
        Number of final raw time bins to remove before any
        preprocessing is performed.

        This trimming is applied before background estimation,
        background subtraction, normalization, and decay-maximum
        trimming.

        Use this when the final TCSPC bins contain acquisition
        artifacts, such as artificially low intensities.

        Default = 0, meaning no raw end-bin trimming is applied.

    i1_tolerance : float
        Tolerance used when locating the last point at the maximum.

    Returns
    -------
    preprocessed : dict
        Dictionary containing the preprocessed decay.

        Keys:
            "time"
                Time axis after raw end-bin trimming, decay-maximum
                trimming, and shifting so that the decay starts at
                t = 0.

            "intensity"
                Background-subtracted and normalized decay.

            "stats"
                Dictionary containing preprocessing statistics.
    """

    # -------------------------------------------------
    # Convert decay to a one-dimensional float array
    # -------------------------------------------------

    decay_curve = np.asarray(decay_curve, dtype=float)

    if decay_curve.ndim != 1:
        raise ValueError("decay_curve must be one-dimensional.")

    original_n_bins = len(decay_curve)

    if original_n_bins < 2:
        raise ValueError("Decay curve is too short.")

    # -------------------------------------------------
    # Time axis
    # -------------------------------------------------

    if time_axis is None:
        time_axis = np.arange(original_n_bins, dtype=float)
    else:
        time_axis = np.asarray(time_axis, dtype=float)

        if len(time_axis) != original_n_bins:
            raise ValueError(
                "time_axis and decay_curve must have the same length."
            )

    # -------------------------------------------------
    # Validate raw end-bin trimming setting
    # -------------------------------------------------

    trim_end_bins = int(trim_end_bins)

    if trim_end_bins < 0:
        raise ValueError("trim_end_bins must be greater than or equal to 0.")

    if trim_end_bins >= original_n_bins:
        raise ValueError(
            "trim_end_bins must be smaller than the number of decay bins."
        )

    # -------------------------------------------------
    # Optional raw end-bin trimming
    # -------------------------------------------------
    #
    # This is intentionally performed before background estimation.
    #
    # Some PTU decays may contain artificially low intensities in the
    # final TCSPC bins. If those bins are included in the baseline
    # region, they can underestimate the background mean. That, in turn,
    # can shift the background-corrected decay and affect normalization
    # and fitting.
    #
    # trim_end_bins = 0
    #     Keep the full decay.
    #
    # trim_end_bins > 0
    #     Remove that many final raw bins before any further processing.
    # -------------------------------------------------

    if trim_end_bins > 0:
        decay_curve = decay_curve[:-trim_end_bins]
        time_axis = time_axis[:-trim_end_bins]

    n_bins = len(decay_curve)

    if n_bins < 2:
        raise ValueError(
            "Decay curve is too short after raw end-bin trimming."
        )

    # -------------------------------------------------
    # Background estimation
    # -------------------------------------------------

    if not (0.0 < background_fraction < 1.0):
        raise ValueError(
            "background_fraction must be between 0 and 1."
        )

    tail_length = max(
        1,
        int(np.ceil(background_fraction * n_bins)),
    )

    tail = decay_curve[-tail_length:]

    background_mean = float(np.mean(tail))

    if tail_length > 1:
        background_std = float(np.std(tail, ddof=1))
    else:
        background_std = np.nan

    # -------------------------------------------------
    # Background subtraction
    # -------------------------------------------------

    corrected = decay_curve - background_mean

    peak = float(np.max(corrected))

    if not np.isfinite(peak) or peak <= 0:
        raise ValueError(
            "Decay has no positive signal after background subtraction."
        )

    intensity = corrected / peak

    # -------------------------------------------------
    # Trim at decay maximum
    # -------------------------------------------------

    one_mask = np.isclose(
        intensity,
        1.0,
        atol=i1_tolerance,
        rtol=0.0,
    )

    if np.any(one_mask):
        decay_start_index = int(np.where(one_mask)[0][-1])
    else:
        decay_start_index = int(np.argmax(intensity))

    intensity = intensity[decay_start_index:]
    time = time_axis[decay_start_index:]

    # Shift time so the decay begins at t = 0.
    time = time - time[0]

    # -------------------------------------------------
    # Output
    # -------------------------------------------------

    stats = {
        "background_mean": background_mean,
        "background_std": background_std,
        "peak": peak,

        # Number of raw bins in the original decay before trimming.
        "original_n_bins": int(original_n_bins),

        # Number of final raw bins removed before preprocessing.
        "trim_end_bins": int(trim_end_bins),

        # Number of bins remaining after raw end-bin trimming.
        "n_bins_after_end_trim": int(n_bins),

        # Number of bins used to estimate the background.
        "background_tail_length": int(tail_length),

        # Index of the decay maximum after raw end-bin trimming.
        # This is the index where the final preprocessed decay starts.
        "trim_index": int(decay_start_index),

        # Number of points in the final preprocessed decay.
        "n_points": int(len(intensity)),
    }

    preprocessed = {
        "time": time,
        "intensity": intensity,
        "stats": stats,
    }

    return preprocessed