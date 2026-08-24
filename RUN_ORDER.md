# Analysis execution order

The commands below reproduce the main analysis stages from the randomized SDSS DR17 sample.

Assume the combined input file is available as:

`data/sdss_dr17_randomized_150k.csv`

## 1. Data retrieval

```powershell
python 01_data_retrieval/step0_randomized_download_sdss_dr17_v3.py
```

The randomized SQL retrieval uses `ORDER BY NEWID()`. The analysed object identifiers should therefore be archived together with the final dataset.

## 2. Development validation and final-test evaluation

```powershell
python 02_validation/run_repeated_cv_final_test.py --data data/sdss_dr17_randomized_150k.csv --out validation_output
```

This stage performs the fixed 80/20 stratified split, 5-fold × 3 repeated development cross-validation, final model fitting, final-test evaluation, model serialization, and prediction export.

## 3. Spatial validation

```powershell
python 03_spatial_validation/run_spatial_validation.py --data data/sdss_dr17_randomized_150k.csv --out spatial_validation_output
```

## 4. Magnitude robustness

```powershell
python 04_magnitude_robustness/run_magnitude_robustness.py --data data/sdss_dr17_randomized_150k.csv --out magnitude_robustness_output
```

## 5. Photometric-uncertainty analysis

```powershell
python 05_photometric_uncertainty/run_photometric_uncertainty.py --data data/sdss_dr17_randomized_150k.csv --out photometric_uncertainty_output
```

## 6. Grouped permutation importance

```powershell
python 06_grouped_permutation/run_grouped_permutation.py --data data/sdss_dr17_randomized_150k.csv --f4-model validation_output/F4_selected_model.joblib --f7-model validation_output/F7_selected_model.joblib --out grouped_permutation_output
```

## 7. Probability calibration

```powershell
python 07_probability_calibration/run_probability_calibration.py --data data/sdss_dr17_randomized_150k.csv --out probability_calibration_output
```

## 8. Class-prevalence sensitivity

```powershell
python 08_prior_shift/run_prior_shift.py --data data/sdss_dr17_randomized_150k.csv --out prior_shift_output
```

## Software environment

Core dependencies are listed in `requirements-core.txt`. To record the exact installed versions from the analysis environment, run:

```powershell
powershell -ExecutionPolicy Bypass -File environment/export_environment.ps1
```

The resulting `requirements.txt` and `environment.yml` can then be archived with the repository.
