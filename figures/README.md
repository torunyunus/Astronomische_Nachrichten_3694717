# Figure generation

Run:

```bash
python figures/generate_figures.py
```

The script reads the machine-readable analysis tables in this repository and writes PNG and PDF outputs to the `figures/` directory.

Generated outputs:

- `Figure_1_repeated_CV_macroF1` — development-set macro-F1 across the seven feature configurations and five classifier families.
- `Figure_3_magnitude_robustness` — magnitude-dependent macro-F1 and STAR recall for F4 Random Forest and F7 LightGBM; x-axis labels include GALAXY, STAR, and QSO counts in each r-band bin.
- `Figure_4_reliability_curves` — two-panel reliability curves for F4 Random Forest and F7 LightGBM comparing raw, sigmoid, and isotonic probabilities with 95% Wilson confidence intervals.
- `Figure_5_grouped_permutation_importance` — grouped permutation importance measured as the decrease in macro-F1.
- `Supplementary_Figure_S1_STAR_class_conditional_recall` — STAR class-conditional recall across r-band magnitude bins.

The plotted values are read directly from the corresponding CSV result tables rather than entered manually.
