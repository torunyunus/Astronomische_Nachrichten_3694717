# Supplementary analysis tables

This directory contains machine-readable supplementary tables associated with the SDSS DR17 classification analysis.

- `Supplementary_Table_S1_spatial_validation.csv` — results from five non-overlapping RA-defined development-set holdout regions for the F4 Random Forest and F7 LightGBM models.
- `Supplementary_Table_S2_magnitude_robustness.csv` — performance across five r-band magnitude bins, including class counts and 95% bootstrap confidence intervals.
- `Supplementary_Table_S3a_uncertainty_tertiles.csv` — performance across low, medium, and high `Err_r` uncertainty tertiles, including approximate r-band signal-to-noise values.
- `Supplementary_Table_S3b_star_magnitude_uncertainty.csv` — STAR recall jointly stratified by r-band magnitude bin and within-bin photometric-uncertainty tertile.
- `Supplementary_Table_S4_prevalence_sensitivity.csv` — class-prevalence sensitivity under a published SDSS DR17 spectroscopic-reference composition, reported for original and prior-adjusted decision rules.

All reported uncertainty intervals are derived using the resampling procedures implemented in the corresponding analysis scripts. The F4 and F7 entries refer to the fixed representative models selected from development-set validation.
