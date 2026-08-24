# Random seeds and deterministic settings

The following settings are used throughout the analysis workflow unless otherwise stated.

| Component | Setting |
|---|---:|
| Python-side randomized source shuffle | 42 |
| Development/final-test stratified split | 42 |
| Repeated stratified cross-validation | 42 |
| Logistic Regression | 42 |
| Decision Tree | 42 |
| Random Forest | 42 |
| LightGBM | 42 |
| Stratified bootstrap confidence intervals | 42 |
| Grouped permutation analysis | 42 |
| Class-prevalence bootstrap analysis | 42 |

The SQL-level source selection uses `ORDER BY NEWID()`. SQL Server does not expose a fixed seed for this operation; therefore the exact analysed source identifiers must be retained separately for exact sample reconstruction.
