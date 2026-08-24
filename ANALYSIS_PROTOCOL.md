# Analysis protocol

## Sample construction

The analysis sample contains 150,000 spectroscopically labelled SDSS DR17 sources: 50,000 GALAXY, 50,000 STAR, and 50,000 QSO objects. Class-specific SQL retrieval uses `ORDER BY NEWID()` and `zWarning = 0`. Photometric magnitudes are restricted to valid finite values in the analysis range.

Seven feature configurations are evaluated:

- F1: spectroscopic redshift
- F2: u, g, r, i, z magnitudes
- F3: seven colour indices
- F4: magnitudes + colours
- F5: magnitudes + spectroscopic redshift
- F6: colours + spectroscopic redshift
- F7: magnitudes + colours + spectroscopic redshift

## Fixed data split

The complete sample is split once using stratified sampling with `random_state=42`:

- development set: 120,000 objects (80%)
- final test set: 30,000 objects (20%)

The final test set is not used for feature-configuration or representative-model selection.

## Development validation

All seven feature configurations and five classifier families are evaluated on the development set using repeated stratified cross-validation:

- 5 folds
- 3 repeats
- 15 matched validation folds per model/configuration
- 35 model/configuration combinations
- 525 model fits

Macro-F1 is the primary model-comparison metric. Hyperparameters are fixed across feature configurations. The representative development-selected models are F4 Random Forest and F7 LightGBM.

## Final-test evaluation

Each selected representative model is fitted to the complete development set and evaluated on the untouched final test set. Reported final-test metrics include accuracy, macro-F1, Matthews correlation coefficient, macro one-vs-rest ROC AUC, class recall, and confusion matrices. Principal confidence intervals use 2,000 stratified bootstrap resamples with seed 42.

## Spatial validation

Spatial sensitivity is assessed inside the development set using five non-overlapping RA-defined regions. Each region is held out in turn while the model is trained on the other four regions.

## Magnitude robustness

The final test sample is divided into five r-band magnitude bins. Performance is reported by bin together with class composition and bootstrap confidence intervals. Class-conditional recall is also evaluated as a function of magnitude.

## Photometric uncertainty

Photometric uncertainty is assessed using `Err_r` tertiles and an approximate r-band signal-to-noise relation, `S/N ≈ 1.0857 / Err_r`. Joint magnitude-by-uncertainty analyses are used to separate faintness effects from photometric-error effects.

## Grouped permutation importance

Grouped permutation analysis is performed on the fixed F4 Random Forest and F7 LightGBM models. Magnitude, colour, full-photometric, and redshift blocks are permuted as groups. Each group uses 100 permutations with seed 42, and importance is measured as the decrease in macro-F1.

## Probability calibration

Calibration mappings are fitted exclusively from development-set out-of-fold predictions. Raw probabilities are compared with sigmoid and isotonic calibration. Evaluation includes multiclass log loss, Brier score, expected calibration error with 10, 15, and 20 equal-width confidence bins, macro-F1, and reliability curves. Confidence intervals use 2,000 bootstrap resamples.

## Class-prevalence sensitivity

Class-conditional confusion rates from the balanced final test set are reweighted to a published SDSS DR17 spectroscopic-reference composition. Original and prior-adjusted decision rules are compared. This analysis is a prevalence-sensitivity diagnostic and does not by itself establish transferability to a different survey population.

## Reproducibility

The workflow order and executable commands are listed in `RUN_ORDER.md`. Core dependencies are listed in `requirements-core.txt`. The environment export utility in `environment/export_environment.ps1` records the installed package versions and Conda environment used on the analysis computer.
