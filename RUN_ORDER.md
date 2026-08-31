# Analysis execution order

The commands below reproduce the main analysis stages from the archived randomized SDSS DR17 sample.

The combined input file is stored as:

`01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz`

The corresponding identifier table is:

`01_data_retrieval/data/randomized_object_ids.csv.gz`

## 1. Validate the archived data

```powershell
python 01_data_retrieval/validate_data_archive.py --data 01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz --ids 01_data_retrieval/data/randomized_object_ids.csv.gz
```

## 2. Development validation and final-test evaluation

```powershell
python 02_validation/run_repeated_cv_final_test.py --data 01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz --out validation_output
```

This stage performs the fixed 80/20 stratified split, 5-fold × 3 repeated development cross-validation, representative-model selection, final model fitting, untouched final-test evaluation, model serialization, prediction export, and manuscript-aligned Table 3–5 exports.

The fixed downstream models are written as:

- `validation_output/F4_selected_model.joblib`
- `validation_output/F7_selected_model.joblib`

## 3. Spatial validation

```powershell
python 03_spatial_validation/run_spatial_validation.py --data 01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz --out spatial_validation_output
```

This stage evaluates the development-selected F4 Random Forest and F7 LightGBM classifier families across five non-overlapping RA hold-out regions and reports accuracy, macro-F1, MCC, macro-OVR AUC, class-specific recall, and class support for Supplementary Table S1.

## 4. Magnitude robustness

```powershell
python 04_magnitude_robustness/run_magnitude_robustness.py --data 01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz --f4-model validation_output/F4_selected_model.joblib --f7-model validation_output/F7_selected_model.joblib --out magnitude_robustness_output
```

This stage uses the fixed serialized representative models and exports the magnitude-bin results underlying manuscript Table 6 and the class-conditional results used for Supplementary Table S2.

## 5. Photometric-uncertainty analysis

```powershell
python 05_photometric_uncertainty/run_photometric_uncertainty.py --data 01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz --f4-model validation_output/F4_selected_model.joblib --f7-model validation_output/F7_selected_model.joblib --out photometric_uncertainty_output
```

This stage uses the same fixed F4/F7 models and exports the uncertainty-tertile and joint magnitude–uncertainty results used for Supplementary Table S3.

## 6. Grouped permutation importance

```powershell
python 06_grouped_permutation/run_grouped_permutation.py --data 01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz --f4-model validation_output/F4_selected_model.joblib --f7-model validation_output/F7_selected_model.joblib --out grouped_permutation_output
```

This stage exports the grouped-permutation results corresponding to manuscript Table 8 and Figure 5.

## 7. Probability calibration

```powershell
python 07_probability_calibration/run_probability_calibration.py --data 01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz --f4-model validation_output/F4_selected_model.joblib --f7-model validation_output/F7_selected_model.joblib --out probability_calibration_output
```

This stage fits calibration mappings exclusively from development-set out-of-fold probabilities, applies them to the fixed final models, and exports the results corresponding to manuscript Table 7 and Figure 4.

## 8. Class-prevalence sensitivity

```powershell
python 08_prior_shift/run_prior_shift.py --data 01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz --f4-model validation_output/F4_selected_model.joblib --f7-model validation_output/F7_selected_model.joblib --out prior_shift_output
```

This stage uses the fixed final models for the prevalence-sensitivity analysis reported in Supplementary Table S4.

## 9. Figure generation

```powershell
python figures/generate_figures.py
```

The figure-generation script follows the final manuscript numbering: Figure 1 (repeated-CV heatmap), Figure 2 (final-test confusion matrices), Figure 3 (magnitude robustness), Figure 4 (probability calibration), and Figure 5 (grouped permutation importance).

## Manuscript table numbering

The final manuscript-aligned principal tables are numbered as follows:

1. Table 1 — feature configurations
2. Table 2 — model hyperparameters
3. Table 3 — repeated-CV macro-F1 comparison
4. Table 4 — development-selected classifier per feature configuration
5. Table 5 — untouched final-test performance
6. Table 6 — magnitude robustness
7. Table 7 — probability calibration
8. Table 8 — grouped permutation importance

## Optional data retrieval

The archived dataset is sufficient to reproduce the reported analysis. To perform a new randomized SDSS DR17 retrieval instead, run:

```powershell
python 01_data_retrieval/step0_randomized_download_sdss_dr17_v3.py
```

Because SQL-level randomization uses `ORDER BY NEWID()`, a new retrieval will not necessarily contain the identical objects.

## Software environment

Core dependencies are listed in `requirements-core.txt`. To record the installed versions from the analysis environment, run:

```powershell
powershell -ExecutionPolicy Bypass -File environment/export_environment.ps1
```
