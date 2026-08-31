# Supplementary analysis tables

This directory contains the machine-readable supplementary tables associated with the final SDSS DR17 classification manuscript.

- `Supplementary_Table_S1_spatial_validation.csv` — detailed five-region spatial hold-out results for the F4 Random Forest and F7 LightGBM models, including accuracy, macro-F1, MCC, macro-OVR AUC, class-specific recall, and class support.
- `Supplementary_Table_S2_class_conditional_magnitude_robustness.csv` — class-conditional r-band magnitude robustness. Within each true class, the final-test objects are divided into five equal-count quintiles and recall is reported with 95% bootstrap confidence intervals.
- `Supplementary_Table_S3a_uncertainty_tertiles.csv` — performance across low, medium, and high `Err_r` uncertainty tertiles, including approximate r-band signal-to-noise values.
- `Supplementary_Table_S3b_star_magnitude_uncertainty.csv` — STAR recall jointly stratified by r-band magnitude bin and within-bin photometric-uncertainty tertile.
- `Supplementary_Table_S4_prevalence_sensitivity.csv` — class-prevalence sensitivity under a published SDSS DR17 spectroscopic-reference composition, reported for original and prior-adjusted decision rules.

The manuscript refers to these outputs collectively as Supplementary Tables S1–S4; the S3 analysis is distributed across two machine-readable CSV files because it contains two complementary panels of results.

All reported uncertainty intervals are derived using the resampling procedures implemented in the corresponding analysis scripts. F4 and F7 refer to the fixed representative models selected from development-set validation.
