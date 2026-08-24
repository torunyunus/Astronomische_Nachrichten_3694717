# Machine-readable tables

This directory contains compact tables used to document the classifier configurations and the principal numerical results.

Key files include:

- `feature_configurations.csv` — definitions of F1–F7.
- `model_hyperparameters.csv` — fixed classifier settings used across feature configurations.
- `Table_3_repeated_CV_macroF1_mean_sd.csv` — repeated development-validation macro-F1 mean and standard deviation.
- `Table_4_highest_mean_CV_model.csv` — highest mean development macro-F1 model within each feature configuration under the fixed settings.
- `F4_final_test_confusion_matrix.csv` and `F7_final_test_confusion_matrix.csv` — final-test confusion matrices.
- `Table_6_probability_calibration.csv` — probability-calibration metrics.
- `Table_7_grouped_permutation_importance.csv` — grouped permutation importance.

More detailed robustness and sensitivity tables are stored in the corresponding analysis directories and in `supplementary/`.
