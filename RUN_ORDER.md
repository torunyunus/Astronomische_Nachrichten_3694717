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

This stage performs the fixed 80/20 stratified split, 5-fold × 3 repeated development cross-validation, final model fitting, final-test evaluation, model serialization, and prediction export.

## 3. Spatial validation

```powershell
python 03_spatial_validation/run_spatial_validation.py --data 01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz --out spatial_validation_output
```

## 4. Magnitude robustness

```powershell
python 04_magnitude_robustness/run_magnitude_robustness.py --data 01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz --out magnitude_robustness_output
```

## 5. Photometric-uncertainty analysis

```powershell
python 05_photometric_uncertainty/run_photometric_uncertainty.py --data 01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz --out photometric_uncertainty_output
```

## 6. Grouped permutation importance

```powershell
python 06_grouped_permutation/run_grouped_permutation.py --data 01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz --f4-model validation_output/F4_selected_model.joblib --f7-model validation_output/F7_selected_model.joblib --out grouped_permutation_output
```

## 7. Probability calibration

```powershell
python 07_probability_calibration/run_probability_calibration.py --data 01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz --out probability_calibration_output
```

## 8. Class-prevalence sensitivity

```powershell
python 08_prior_shift/run_prior_shift.py --data 01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz --out prior_shift_output
```

## 9. Figure generation

```powershell
python figures/generate_figures.py
```

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
