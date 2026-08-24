# Reproducibility package checklist

Maintainer: Yunis Torun

Analysis/revision period: 17–24 August 2026.

## Archived in the repository

- Randomized SDSS DR17 retrieval script (final version)
- Exact SQL queries used for class-specific randomized retrieval
- Download summary and dataset checksum information
- Repeated-CV summary and selected-model metadata
- Untouched final-test performance summaries and confusion matrices
- Paired F6 versus F7 development-validation comparison
- Spatial holdout region definitions, fold results, and protocol
- Magnitude-robustness summary
- Photometric-uncertainty robustness summary
- Grouped permutation summary and manuscript table
- Probability-calibration summary, manuscript table, and ECE bin-sensitivity results
- Prior-shift summary and Supplementary Table S4
- SHA-256 manifest for the final analysis scripts and archived sample files

## Files to archive before final manuscript submission

- `step1_repeated_cv_final_test.py`
- `step2_spatial_validation_v3.py`
- `step3_magnitude_robustness_v3.py`
- `step4_photometric_uncertainty_v3.py`
- `step5_grouped_permutation_v1.py`
- `step6_calibration_v2.py`
- `step7_prior_shift_v1.py`
- `randomized_object_ids.csv`
- exact software environment (`requirements.txt` and/or `environment.yml`)
- final manuscript figures and complete supplementary tables

The SHA-256 hashes of the final script versions and archived data files are listed in `FILE_MANIFEST_SHA256.csv` so that uploaded copies can be verified before release.
