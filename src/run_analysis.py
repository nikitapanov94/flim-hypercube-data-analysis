"""
run_analysis.py

Run the complete Version 2 FLIM analysis workflow.

Version 2 workflow
------------------
1. Select one or more PTU files.
2. Select an output folder.
3. Read each PTU file.
4. Generate FLIM parameter maps using analysis bins.
5. Save maps and analysis report.
6. Optionally export tau distribution.
7. Optionally export selected analysis-bin decay curves and fits.

Concept
-------
The analysis bin is the fundamental fitted spatial unit.

A detector pixel is simply the smallest possible analysis bin.
"""

from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from analysis.read_ptu import read_ptu
from analysis.generate_maps import generate_maps
from analysis.save_maps import save_maps
from analysis.export_tau_distribution import export_tau_distribution
from analysis.export_selected_decays import export_selected_decays
from analysis.export_snr_vs_photons import export_snr_vs_photons


# =====================================================
# GENERAL ANALYSIS SETTINGS
# =====================================================

SOFTWARE_VERSION = "FLIM Analysis v2.0"

VERBOSE = True


# =====================================================
# VERSION 2 SPATIAL ANALYSIS SETTINGS
# =====================================================

ROI = {
    "y_start": 0,
    "y_stop": 128,
    "x_start": 0,
    "x_stop": 128,
}

BIN_SIZE = {
    "height": 4,
    "width": 4,
}

SELECTED_BINS = [
    (16, 16), #check... this might be (y, x), set to (0,0) when extracting the "master" decay curve over the whole map.
]


# =====================================================
# FIT ACCEPTANCE THRESHOLD SETTINGS
# =====================================================
#
# There are two independent threshold concepts:
#
# 1. Photon threshold
#    ----------------
#    This is a pre-fit threshold.
#
#    If APPLY_PHOTON_THRESHOLD = True, an analysis bin is skipped before
#    preprocessing and fitting when its summed raw photon count is below
#    MIN_TOTAL_PHOTONS.
#
#    If APPLY_PHOTON_THRESHOLD = False, every analysis bin is fitted
#    regardless of total photon count.
#
#
# 2. R2 threshold
#    ------------
#    This is a post-fit threshold.
#
#    Every bin that passes the photon threshold is first preprocessed and fitted.
#    The fitted R2 value is then compared with MIN_R2.
#
#    The exported r2_accepted_map is always generated:
#
#        r2_accepted_map = 1  means R2 >= MIN_R2
#        r2_accepted_map = 0  means R2 <  MIN_R2, or no valid fit was obtained
#
#    If APPLY_R2_THRESHOLD = False:
#        R2 is used only diagnostically.
#        tau_ns_map contains all successfully fitted bins that passed the
#        photon threshold.
#        success_map = 1 for all successfully fitted bins, even if R2 < MIN_R2.
#
#    If APPLY_R2_THRESHOLD = True:
#        R2 becomes an actual acceptance criterion.
#        tau_ns_map still stores fitted tau values for diagnostic inspection,
#        but success_map = 1 only for bins with R2 >= MIN_R2.
#        Tau distribution export uses success_map, so it includes only
#        R2-accepted bins.
#
# In short:
#
#    r2_map           = actual fitted R2 values
#    r2_accepted_map  = whether each fit passes the R2 criterion
#    success_map      = whether each bin is included in final quantitative output
# =====================================================

APPLY_PHOTON_THRESHOLD = True
MIN_TOTAL_PHOTONS = 0

APPLY_R2_THRESHOLD = False
MIN_R2 = 0.95

# =====================================================
# PREPROCESSING SETTINGS
# =====================================================

TRIM_END_BINS = 5

BACKGROUND_FRACTION = 0.05


# =====================================================
# FITTING SETTINGS
# =====================================================

RUN_MAP_ANALYSIS = True
SAVE_MAPS = True

THRESHOLD_MODE = "none"
# Options:
#   "noise"
#   "fixed"
#   "none"

FIT_AMPLITUDE = True
FIT_BACKGROUND = True


# =====================================================
# OPTIONAL EXPORT SETTINGS
# =====================================================

RUN_TAU_DISTRIBUTION_EXPORT = True

RUN_SNR_VS_PHOTONS_EXPORT = True

RUN_SELECTED_DECAY_EXPORT = True


# =====================================================
# SELECT INPUT FILES
# =====================================================

root = tk.Tk()
root.withdraw()

ptu_files = filedialog.askopenfilenames(
    title="Select one or more PTU files",
    filetypes=[("PTU files", "*.ptu")],
)

if not ptu_files:
    raise SystemExit("No PTU files were selected.")


# =====================================================
# SELECT OUTPUT FOLDER
# =====================================================

output_root = filedialog.askdirectory(
    title="Select folder where results will be saved"
)

if not output_root:
    raise SystemExit("No output folder was selected.")

output_root = Path(output_root)


# =====================================================
# START ANALYSIS
# =====================================================

print()
print("========================================")
print("FLIM ANALYSIS VERSION 2")
print("========================================")
print()

print(f"{len(ptu_files)} PTU file(s) selected.\n")


# =====================================================
# PROCESS EACH PTU FILE
# =====================================================

for i, ptu_file in enumerate(ptu_files, start=1):

    print("----------------------------------------")
    print(f"Processing file {i} of {len(ptu_files)}")
    print("----------------------------------------")

    # -------------------------------------------------
    # Read PTU file
    # -------------------------------------------------

    ptu_data = read_ptu(ptu_file)

    measurement_name = Path(ptu_data["filename"]).stem
    metadata = ptu_data["metadata"]

    # -------------------------------------------------
    # Create output folder for this measurement
    # -------------------------------------------------

    measurement_output_folder = output_root / measurement_name

    measurement_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Analysis settings record
    # -------------------------------------------------

    analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    settings = {
        "software_version": SOFTWARE_VERSION,
        "analysis_date": analysis_time,
        "PTU_file": Path(ptu_file).name,

        "roi": ROI,
        "bin_size": BIN_SIZE,
        "selected_bins": SELECTED_BINS,

        "apply_photon_threshold": APPLY_PHOTON_THRESHOLD,
        "min_total_photons": MIN_TOTAL_PHOTONS,
        "apply_r2_threshold": APPLY_R2_THRESHOLD,
        "min_r2": MIN_R2,

        "trim_end_bins": TRIM_END_BINS,
        "background_fraction": BACKGROUND_FRACTION,

        "threshold_mode": THRESHOLD_MODE,
        "fit_amplitude": FIT_AMPLITUDE,
        "fit_background": FIT_BACKGROUND,

        "run_map_analysis": RUN_MAP_ANALYSIS,
        "save_maps": SAVE_MAPS,
        "run_tau_distribution_export": RUN_TAU_DISTRIBUTION_EXPORT,
        "run_snr_vs_photons_export": RUN_SNR_VS_PHOTONS_EXPORT,
        "run_selected_decay_export": RUN_SELECTED_DECAY_EXPORT,
    }

    # -------------------------------------------------
    # Generate FLIM maps using analysis bins
    # -------------------------------------------------

    analysis_result = None

    if RUN_MAP_ANALYSIS:

        analysis_result = generate_maps(
            ptu_data=ptu_data,
            roi=ROI,
            bin_size=BIN_SIZE,
            selected_bins=SELECTED_BINS,
            apply_photon_threshold=APPLY_PHOTON_THRESHOLD,
            min_total_photons=MIN_TOTAL_PHOTONS,
            apply_r2_threshold=APPLY_R2_THRESHOLD,
            min_r2=MIN_R2,
            threshold_mode=THRESHOLD_MODE,
            fit_amplitude=FIT_AMPLITUDE,
            fit_background=FIT_BACKGROUND,
            background_fraction=BACKGROUND_FRACTION,
            trim_end_bins=TRIM_END_BINS,
            verbose=VERBOSE,
        )

        if SAVE_MAPS:

            save_maps(
                analysis_result=analysis_result,
                settings=settings,
                metadata=metadata,
                output_folder=measurement_output_folder,
                measurement_name=measurement_name,
            )

    # -------------------------------------------------
    # Export tau distribution
    # -------------------------------------------------

    if RUN_TAU_DISTRIBUTION_EXPORT:

        if analysis_result is None:
            raise RuntimeError(
                "Tau distribution export requires analysis_result. "
                "Set RUN_MAP_ANALYSIS = True."
            )

        export_tau_distribution(
            analysis_result=analysis_result,
            output_folder=measurement_output_folder,
            measurement_name=measurement_name,
        )

    # -------------------------------------------------
    # Export signal-to-noise ratio versus total photons
    # -------------------------------------------------

    if RUN_SNR_VS_PHOTONS_EXPORT:

        if analysis_result is None:
            raise RuntimeError(
                "SNR versus total photons export requires "
                "analysis_result. Set RUN_MAP_ANALYSIS = True."
            )

        export_snr_vs_photons(
            analysis_result=analysis_result,
            output_folder=measurement_output_folder,
            measurement_name=measurement_name,
        )

    # -------------------------------------------------
    # Export selected analysis-bin decay curves and fits
    # -------------------------------------------------

    if RUN_SELECTED_DECAY_EXPORT:

        if analysis_result is None:
            raise RuntimeError(
                "Selected decay export requires analysis_result. "
                "Set RUN_MAP_ANALYSIS = True."
            )

        export_selected_decays(
            analysis_result=analysis_result,
            output_folder=measurement_output_folder,
            measurement_name=measurement_name,
        )

    print(f"Finished: {measurement_name}\n")


# =====================================================
# FINISHED
# =====================================================

print("========================================")
print("Analysis complete.")
print(f"Successfully processed {len(ptu_files)} PTU file(s).")
print(f"Results saved to:\n{output_root}")
print("========================================")