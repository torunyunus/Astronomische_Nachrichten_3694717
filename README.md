# Reliability-Aware Star–Galaxy–Quasar Classification in SDSS DR17

**Reliability-Aware Star–Galaxy–Quasar Classification in SDSS DR17: Feature Ablation, Magnitude Robustness, and Probability Calibration**

Authors: Yunis Torun, Yiğit Ütük, and Serkan Akkoyun.

Repository maintainer: Yunis Torun.

## Purpose

This repository contains the analysis code and reproducibility material for the SDSS DR17 star–galaxy–quasar classification study. The analysis uses a randomized, class-balanced sample of 150,000 spectroscopically labelled SDSS DR17 sources (50,000 GALAXY, 50,000 STAR, and 50,000 QSO objects).

## Repository structure

- `01_data_retrieval/` — randomized SDSS DR17 retrieval, exact SQL, archived analysis data, metadata, checksums, and data documentation.
- `02_validation/` — repeated stratified development validation and untouched final-test evaluation.
- `03_spatial_validation/` — spatially disjoint RA-region validation.
- `04_magnitude_robustness/` — magnitude-dependent robustness analysis.
- `05_photometric_uncertainty/` — photometric-error and approximate S/N sensitivity analysis.
- `06_grouped_permutation/` — grouped permutation importance.
- `07_probability_calibration/` — raw, sigmoid, and isotonic probability calibration.
- `08_prior_shift/` — class-prevalence and label-shift sensitivity analysis.
- `tables/` — manuscript-aligned machine-readable Tables 1–8 and final-test confusion matrices.
- `supplementary/` — machine-readable Supplementary Tables S1–S4.
- `figures/` — code for reproducing manuscript Figures 1–5 and the supplementary figure.
- `environment/` — exact Conda environment, complete package snapshot, and environment documentation.

## Core experimental design

The randomized 150,000-source sample is split once into a 120,000-object development set and a 30,000-object untouched final test set using a stratified split with `random_state=42`.

Model and feature-configuration comparison is performed only within the development set using repeated stratified cross-validation (5 folds × 3 repeats). The representative configurations retained from development validation are:

- **F4 Random Forest** — redshift-free photometric configuration (magnitudes + colours).
- **F7 LightGBM** — full spectroscopic-reference configuration (magnitudes + colours + spectroscopic redshift).

The selected models are serialized after fitting to the complete development set and are reused by the downstream final-test reliability analyses so that magnitude robustness, photometric-uncertainty analysis, grouped permutation importance, probability calibration, and prevalence sensitivity refer to the same fixed representative models.

The detailed experimental protocol is documented in `ANALYSIS_PROTOCOL.md`.

## Manuscript table and figure mapping

Principal table files in `tables/` follow the final manuscript numbering exactly:

- Table 1 — feature configurations
- Table 2 — model hyperparameters
- Table 3 — repeated-CV macro-F1 comparison
- Table 4 — development-selected classifier per feature configuration
- Table 5 — untouched final-test performance
- Table 6 — magnitude robustness
- Table 7 — probability calibration
- Table 8 — grouped permutation importance

`figures/generate_figures.py` follows the final manuscript figure numbering exactly:

- Figure 1 — repeated-development-validation macro-F1 heatmap
- Figure 2 — final-test confusion matrices
- Figure 3 — magnitude-dependent robustness
- Figure 4 — probability-calibration reliability curves
- Figure 5 — grouped permutation importance

## Randomized SQL retrieval and archived sample

Class-specific sampling uses SQL-level randomization with `ORDER BY NEWID()` and applies `zWarning = 0`. Because `NEWID()` is not seedable, rerunning the same SQL query does not guarantee retrieval of the identical 150,000 objects. The exact SQL text is included in this repository.

For exact numerical reconstruction, the analysed sample is archived in compressed form:

- `01_data_retrieval/data/randomized_object_ids.csv.gz`
- `01_data_retrieval/data/sdss_dr17_randomized_150k.csv.gz`

SHA-256 checksums for both compressed and uncompressed forms are listed in `01_data_retrieval/DATA_FILE_SHA256.csv`. The archive can be checked with `01_data_retrieval/validate_data_archive.py`.

## Interpretation notes

- F4 is the redshift-free photometric setting.
- F7 is a spectroscopic-reference setting; its performance is conditional on the availability of spectroscopic redshift in the selected SDSS spectroscopic sample.
- The study sample is deliberately class-balanced. Population-level probability interpretation therefore depends on the target class prior.
- Classifier-family rankings are conditional on the prespecified fixed-hyperparameter benchmark.

## Software environment

The exact analysis environment is archived as `environment/environment.yml`, and the complete package snapshot is provided in `environment/requirements.txt`. Principal package versions are summarized in `environment/REFERENCE_ENVIRONMENT.md` and `environment/requirements-reference.txt`.

## Execution

`RUN_ORDER.md` lists the commands for the complete analysis workflow, including the fixed-model arguments required by downstream reliability analyses. The archived `.csv.gz` files can be read directly by the analysis scripts. Figure-generation code is available in `figures/generate_figures.py`.

## Citation

A DOI/citation entry will be added after repository archival.
