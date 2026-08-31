# Machine-readable tables

This directory contains the manuscript-aligned principal tables for the final version of the SDSS DR17 classification study.

The numbered files correspond directly to Tables 1–8 in the manuscript:

- `Table_1_feature_configurations.csv` — definitions of F1–F7.
- `Table_2_model_hyperparameters.csv` — fixed classifier settings used across feature configurations.
- `Table_3_repeated_CV_macroF1_mean_sd.csv` — repeated development-validation macro-F1 mean ± SD for all model–feature combinations.
- `Table_4_highest_mean_CV_model.csv` — development-selected classifier with the highest mean macro-F1 for each feature configuration.
- `Table_5_final_test_performance.csv` — untouched final-test performance of the fixed F4 Random Forest and F7 LightGBM models.
- `Table_6_magnitude_robustness.csv` — magnitude-dependent robustness across five equal-count r-band bins.
- `Table_7_probability_calibration.csv` — raw, sigmoid, and isotonic probability-calibration results.
- `Table_8_grouped_permutation_importance.csv` — grouped permutation importance.

`F4_final_test_confusion_matrix.csv` and `F7_final_test_confusion_matrix.csv` provide the machine-readable counts used for Figure 2.

Detailed spatial, class-conditional magnitude, photometric-uncertainty, and prevalence-sensitivity results are stored in `supplementary/` and in the corresponding analysis directories.

The unnumbered `feature_configurations.csv` and `model_hyperparameters.csv` files are retained as compact machine-readable definitions used during repository development; the numbered Table 1 and Table 2 files are the manuscript-aligned final exports.
