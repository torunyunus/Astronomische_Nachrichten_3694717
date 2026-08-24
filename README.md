# Reliability-Aware Star–Galaxy–Quasar Classification in SDSS DR17

**Reliability-Aware Star–Galaxy–Quasar Classification in SDSS DR17: Feature Ablation, Magnitude Robustness, and Probability Calibration**

Authors: Yunis Torun, Yiğit Ütük, and Serkan Akkoyun.

Repository maintainer: Yunis Torun.

Analysis period represented in this repository: **17–24 August 2026**.

## Purpose

This repository contains the analysis code and reproducibility material for the SDSS DR17 star–galaxy–quasar classification study. The analysis uses a randomized, class-balanced sample of 150,000 spectroscopically labelled SDSS DR17 sources (50,000 GALAXY, 50,000 STAR, and 50,000 QSO objects).

## Repository structure

- `01_data_retrieval/` — randomized SDSS DR17 retrieval, exact SQL, sample metadata, and data-archive notes.
- `02_validation/` — repeated stratified development validation and untouched final-test evaluation.
- `03_spatial_validation/` — spatially disjoint RA-region validation.
- `04_magnitude_robustness/` — magnitude-dependent robustness analysis.
- `05_photometric_uncertainty/` — photometric-error and approximate S/N sensitivity analysis.
- `06_grouped_permutation/` — grouped permutation importance.
- `07_probability_calibration/` — raw, sigmoid, and isotonic probability calibration.
- `08_prior_shift/` — class-prevalence and label-shift sensitivity analysis.
- `tables/` — tables used in the manuscript and supplementary material.
- `figures/` — figure-generation code.

## Core experimental design

The randomized 150,000-source sample is split once into a 120,000-object development set and a 30,000-object untouched final test set using a stratified split with `random_state=42`.

Model and feature-configuration comparison is performed only within the development set using repeated stratified cross-validation (5 folds × 3 repeats). The representative configurations retained from development validation are:

- **F4 Random Forest** — redshift-free photometric configuration (magnitudes + colours).
- **F7 LightGBM** — full spectroscopic-reference configuration (magnitudes + colours + spectroscopic redshift).

## Randomized SQL retrieval

Class-specific sampling uses SQL-level randomization with `ORDER BY NEWID()` and applies `zWarning = 0`. Because `NEWID()` is not seedable, rerunning the same SQL query does not guarantee retrieval of the identical 150,000 objects. The exact SQL text is included in this repository. For exact numerical reconstruction, the analysed object-identifier table and combined randomized source table should be deposited with the repository or its associated data archive; the expected filenames and contents are described in `01_data_retrieval/DATA_ARCHIVE.md`.

The file-level SHA-256 checksums available for the analysis files are listed in `FILE_MANIFEST_SHA256.csv`.

## Interpretation notes

- F4 is the redshift-free photometric setting.
- F7 is a spectroscopic-reference setting; its performance is conditional on the availability of spectroscopic redshift in the selected SDSS spectroscopic sample.
- The study sample is deliberately class-balanced. Population-level probability interpretation therefore depends on the target class prior.
- Classifier-family rankings are conditional on the prespecified fixed-hyperparameter benchmark.

## Software environment

Core dependencies are listed in `requirements-core.txt`. The PowerShell utility in `environment/export_environment.ps1` records the exact installed package versions and Conda environment from the analysis computer.

## Execution

`RUN_ORDER.md` lists the commands for the complete analysis workflow. Figure-generation code is available in `figures/generate_figures.py`.

## Citation

A DOI/citation entry can be added after repository archival.
