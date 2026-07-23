"""
fit_decay.py

Purpose
-------
Fit one preprocessed fluorescence decay using a stretched
exponential model.

Model
-----
I(t) = A * exp[-(t / tau)^beta] + C

Input
-----
time
intensity

Output
------
fit_result
"""

import numpy as np
from scipy.optimize import curve_fit


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

# -------------------------------------------------------
# Fit-window threshold configuration
# -------------------------------------------------------
#
# The fit does NOT necessarily use the entire decay curve.
# Instead, the fitting range ends when the decay signal falls
# below a chosen cutoff for a certain number of consecutive bins.
#
# There are THREE available cutoff modes:
#
# 1) threshold_mode = "noise"
#
#    cutoff = NOISE_K * background_std / peak
#
#    This uses the background noise estimated during preprocessing.
#    This is the original approach from the legacy fitting code.
#
#    Use this for normal PTU pixel-by-pixel fitting:
#
#        fit_decay(
#            time,
#            intensity,
#            stats=preprocessed["stats"],
#            threshold_mode="noise",
#        )
#
#    To make the noise threshold stricter or looser, change NOISE_K.
#
#
# 2) threshold_mode = "fixed"
#
#    cutoff = FIXED_CUTOFF
#
#    This uses one fixed normalized intensity value for all curves.
#
#    Use this for legacy validation or if you want a simple fixed cutoff:
#
#        fit_decay(
#            time,
#            intensity,
#            stats=None,
#            threshold_mode="fixed",
#        )
#
#    To change the fixed threshold, change FIXED_CUTOFF.
#
#
# 3) threshold_mode = "none"
#
#    No cutoff is applied.
#
#    The entire valid decay curve is fitted.
#
#    Use this when you want to fit the full decay without stopping
#    based on either a noise-based or fixed threshold:
#
#        fit_decay(
#            time,
#            intensity,
#            stats=None,
#            threshold_mode="none",
#        )
#
#
# In the "noise" and "fixed" modes, the fit ends only after the
# intensity has stayed below the cutoff for CONSECUTIVE_BINS points.
#
# This avoids stopping the fit because of one noisy bin.
#
# In the "none" mode, CONSECUTIVE_BINS is ignored because no cutoff
# is applied.
# -------------------------------------------------------

NOISE_K = 3.0
FIXED_CUTOFF = 0.001
CONSECUTIVE_BINS = 20
MIN_FIT_POINTS = 50


# -------------------------------------------------------
# Model parameter configuration
# -------------------------------------------------------

TAU_BOUNDS = (1e-9, 1e-7) # (1e-9, 1e-7) # 1 ns to 100 ns, for Ag2S in water
BETA_BOUNDS = (0, 1)

A_BOUNDS = (0.99, 1.01)
C_BOUNDS = (-0.01, 0.01)

TAU_INIT = 50e-9 #50e-9 ns for Ag2S in water
BETA_INIT = 0.5
A_INIT = 1.0
C_INIT = 0.0


# -------------------------------------------------------
# Model
# -------------------------------------------------------

def stretched_exponential(t, A, tau, beta, C):
    """
    Stretched exponential decay model.

    Parameters
    ----------
    t : array-like
        Time axis.

    A : float
        Amplitude.

    tau : float
        Characteristic lifetime.

    beta : float
        Stretching exponent.

    C : float
        Constant offset.

    Returns
    -------
    intensity : np.ndarray
        Model intensity.
    """

    t = np.asarray(t, dtype=float)

    return A * np.exp(-((t / tau) ** beta)) + C


# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def r2_score(y, y_hat):
    """
    Compute coefficient of determination R^2.
    """

    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)

    valid = np.isfinite(y) & np.isfinite(y_hat)

    if np.sum(valid) < 2:
        return np.nan

    y = y[valid]
    y_hat = y_hat[valid]

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    if ss_tot == 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


def compute_fit_end_index(
    intensity,
    stats=None,
    threshold_mode="none",
    noise_k=NOISE_K,
    fixed_cutoff=FIXED_CUTOFF,
    consecutive_bins=CONSECUTIVE_BINS,
    min_fit_points=MIN_FIT_POINTS,
):
    """
    Determine where the decay should stop being fitted.

    The fit window starts at the beginning of the preprocessed decay
    and ends when the normalized intensity falls below a cutoff for
    a chosen number of consecutive bins.

    If threshold_mode="none", no cutoff is applied and the full
    valid decay curve is fitted.

    Parameters
    ----------
    intensity : array-like
        Preprocessed normalized decay.

    stats : dict or None
        Statistics returned by preprocess_decay().

        Required if threshold_mode="noise".

        Expected keys:
            stats["background_std"]
            stats["peak"]

    threshold_mode : str
        Controls how the fitting cutoff is calculated.

        Options:

        "noise"
            Use the original noise-based threshold:

                cutoff = noise_k * background_std / peak

            This requires stats from preprocess_decay().

        "fixed"
            Use one fixed normalized cutoff value:

                cutoff = fixed_cutoff

            This does not require stats.

        "none"
            Do not apply any cutoff.

            The fit uses the entire valid decay curve.

            This does not require stats.

    noise_k : float
        Multiplier for the noise-based cutoff.

        Example:
            noise_k = 3.0 means the fit ends when the decay falls
            below 3 times the normalized background noise.

    fixed_cutoff : float
        Fixed normalized intensity cutoff used when
        threshold_mode="fixed".

        Example:
            fixed_cutoff = 0.001 means the fit ends when intensity
            stays below 0.001.

    consecutive_bins : int
        Number of consecutive points that must be below the cutoff
        before the fit is stopped.

        Ignored when threshold_mode="none".

    min_fit_points : int
        Minimum number of points that must be retained for fitting.

    Returns
    -------
    fit_end_index : int
        Index where fitting should stop.

    cutoff : float
        Cutoff value used.

        If threshold_mode="none", cutoff is np.nan.
    """

    intensity = np.asarray(intensity, dtype=float)

    if intensity.ndim != 1:
        raise ValueError("intensity must be one-dimensional.")

    if len(intensity) == 0:
        raise ValueError("intensity is empty.")

    # ---------------------------------------------------
    # Determine cutoff
    # ---------------------------------------------------

    if threshold_mode == "none":

        fit_end_index = len(intensity)
        cutoff = np.nan

        return fit_end_index, float(cutoff)

    elif threshold_mode == "noise":

        if stats is None:
            raise ValueError(
                "threshold_mode='noise' requires stats from preprocess_decay(). "
                "Either pass stats=preprocessed['stats'] or use "
                "threshold_mode='fixed' or threshold_mode='none'."
            )

        background_std = stats.get("background_std", np.nan)
        peak = stats.get("peak", np.nan)

        if (
            not np.isfinite(background_std)
            or not np.isfinite(peak)
            or peak <= 0
        ):
            raise ValueError(
                "Invalid stats for noise-based threshold. "
                "Expected finite stats['background_std'] and positive stats['peak']."
            )

        cutoff = noise_k * background_std / peak

    elif threshold_mode == "fixed":

        cutoff = fixed_cutoff

    else:
        raise ValueError(
            "threshold_mode must be either 'noise', 'fixed', or 'none'."
        )

    below = intensity <= cutoff

    # ---------------------------------------------------
    # Find first sustained below-cutoff region
    # ---------------------------------------------------

    N = int(consecutive_bins)

    if N <= 1:
        below_indices = np.where(below)[0]

        if len(below_indices) == 0:
            fit_end_index = len(intensity)
        else:
            fit_end_index = int(below_indices[0])

        fit_end_index = max(fit_end_index, min_fit_points)
        fit_end_index = min(fit_end_index, len(intensity))

        return fit_end_index, float(cutoff)

    hits = np.convolve(
        below.astype(int),
        np.ones(N, dtype=int),
        mode="valid",
    )

    start_indices = np.where(hits == N)[0]

    if len(start_indices) == 0:
        fit_end_index = len(intensity)
    else:
        fit_end_index = int(start_indices[0])

    fit_end_index = max(fit_end_index, min_fit_points)
    fit_end_index = min(fit_end_index, len(intensity))

    return fit_end_index, float(cutoff)


# -------------------------------------------------------
# Main fitting function
# -------------------------------------------------------

def fit_decay(
    time,
    intensity,
    stats=None,
    threshold_mode="noise",
    fit_amplitude=True,
    fit_background=True,
):
    """
    Fit one preprocessed fluorescence decay.

    Parameters
    ----------
    time : array-like
        Time axis returned by preprocess_decay().

    intensity : array-like
        Normalized intensity returned by preprocess_decay().

    stats : dict, optional
        Statistics returned by preprocess_decay().

        Use this when threshold_mode="noise":

            fit_decay(
                time,
                intensity,
                stats=preprocessed["stats"],
                threshold_mode="noise",
            )

        stats is not required when threshold_mode="fixed" or
        threshold_mode="none".

    threshold_mode : str
        Controls how the fit endpoint is chosen.

        "noise"
            Original approach.

            The fit cutoff is calculated from the preprocessing
            background noise:

                cutoff = NOISE_K * background_std / peak

            This requires stats.

        "fixed"
            The fit cutoff is a fixed normalized intensity value:

                cutoff = FIXED_CUTOFF

            This does not require stats.

        "none"
            No cutoff is applied.

            The full valid decay curve is fitted.

            This does not require stats.

    fit_amplitude : bool
        If True, fit A.
        If False, fix A = 1.

    fit_background : bool
        If True, fit C.
        If False, fix C = 0.

    Returns
    -------
    fit_result : dict
        Dictionary containing fit parameters, quality metrics,
        fit range, and fitted curve.
    """

    time = np.asarray(time, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if time.ndim != 1:
        raise ValueError("time must be one-dimensional.")

    if intensity.ndim != 1:
        raise ValueError("intensity must be one-dimensional.")

    if len(time) != len(intensity):
        raise ValueError("time and intensity must have the same length.")

    valid = np.isfinite(time) & np.isfinite(intensity)

    time_valid = time[valid]
    intensity_valid = intensity[valid]

    if len(time_valid) < MIN_FIT_POINTS:
        return {
            "ok": False,
            "reason": "too_few_valid_points",
        }

    # ---------------------------------------------------
    # Determine fitting range
    # ---------------------------------------------------
    #
    # This is where the fitting threshold is applied.
    #
    # For normal pixel-wise PTU fitting, use:
    #
    #     threshold_mode="noise"
    #     stats=preprocessed["stats"]
    #
    # For a fixed cutoff, use:
    #
    #     threshold_mode="fixed"
    #     stats=None
    #
    # For fitting the full valid decay curve, use:
    #
    #     threshold_mode="none"
    #     stats=None
    #
    # The returned fit_end_index determines the final point
    # included in the fit.
    # ---------------------------------------------------

    fit_end_index, cutoff = compute_fit_end_index(
        intensity_valid,
        stats=stats,
        threshold_mode=threshold_mode,
    )

    fit_time = time_valid[:fit_end_index]
    fit_intensity = intensity_valid[:fit_end_index]

    if len(fit_time) < MIN_FIT_POINTS:
        return {
            "ok": False,
            "reason": "too_few_fit_points",
        }

    # ---------------------------------------------------
    # Define fitting model according to selected options
    # ---------------------------------------------------

    if fit_amplitude and fit_background:

        def model(t, A, tau, beta, C):
            return stretched_exponential(t, A, tau, beta, C)

        p0 = [
            A_INIT,
            TAU_INIT,
            BETA_INIT,
            C_INIT,
        ]

        bounds = (
            [
                A_BOUNDS[0],
                TAU_BOUNDS[0],
                BETA_BOUNDS[0],
                C_BOUNDS[0],
            ],
            [
                A_BOUNDS[1],
                TAU_BOUNDS[1],
                BETA_BOUNDS[1],
                C_BOUNDS[1],
            ],
        )

        popt, _ = curve_fit(
            model,
            fit_time,
            fit_intensity,
            p0=p0,
            bounds=bounds,
            maxfev=40000,
        )

        A, tau, beta, C = popt

    elif fit_amplitude and not fit_background:

        def model(t, A, tau, beta):
            return stretched_exponential(t, A, tau, beta, 0.0)

        p0 = [
            A_INIT,
            TAU_INIT,
            BETA_INIT,
        ]

        bounds = (
            [
                A_BOUNDS[0],
                TAU_BOUNDS[0],
                BETA_BOUNDS[0],
            ],
            [
                A_BOUNDS[1],
                TAU_BOUNDS[1],
                BETA_BOUNDS[1],
            ],
        )

        popt, _ = curve_fit(
            model,
            fit_time,
            fit_intensity,
            p0=p0,
            bounds=bounds,
            maxfev=40000,
        )

        A, tau, beta = popt
        C = 0.0

    elif not fit_amplitude and fit_background:

        def model(t, tau, beta, C):
            return stretched_exponential(t, 1.0, tau, beta, C)

        p0 = [
            TAU_INIT,
            BETA_INIT,
            C_INIT,
        ]

        bounds = (
            [
                TAU_BOUNDS[0],
                BETA_BOUNDS[0],
                C_BOUNDS[0],
            ],
            [
                TAU_BOUNDS[1],
                BETA_BOUNDS[1],
                C_BOUNDS[1],
            ],
        )

        popt, _ = curve_fit(
            model,
            fit_time,
            fit_intensity,
            p0=p0,
            bounds=bounds,
            maxfev=40000,
        )

        tau, beta, C = popt
        A = 1.0

    else:

        def model(t, tau, beta):
            return stretched_exponential(t, 1.0, tau, beta, 0.0)

        p0 = [
            TAU_INIT,
            BETA_INIT,
        ]

        bounds = (
            [
                TAU_BOUNDS[0],
                BETA_BOUNDS[0],
            ],
            [
                TAU_BOUNDS[1],
                BETA_BOUNDS[1],
            ],
        )

        popt, _ = curve_fit(
            model,
            fit_time,
            fit_intensity,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )

        tau, beta = popt
        A = 1.0
        C = 0.0

    # ---------------------------------------------------
    # Evaluate fit
    # ---------------------------------------------------

    fitted_curve = stretched_exponential(
        fit_time,
        A,
        tau,
        beta,
        C,
    )

    r2 = r2_score(
        fit_intensity,
        fitted_curve,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (fit_intensity - fitted_curve) ** 2
            )
        )
    )

    fit_result = {
        "ok": True,
        "A": float(A),
        "tau": float(tau),
        "beta": float(beta),
        "C": float(C),
        "r2": float(r2),
        "rmse": rmse,
        "cutoff": float(cutoff),
        "threshold_mode": threshold_mode,
        "fit_end_index": int(fit_end_index),
        "fit_end_time": float(fit_time[-1]),
        "fit_time": fit_time,
        "fitted_curve": fitted_curve,
    }

    return fit_result