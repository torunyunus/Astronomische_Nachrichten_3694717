# Figure generation

Run:

```bash
python figures/generate_figures.py
```

The script reads the manuscript-aligned machine-readable tables in this repository and writes PNG and PDF outputs to the `figures/` directory.

Generated principal outputs follow the final manuscript numbering exactly:

- `Figure_1_repeated_CV_macroF1` — repeated-development-validation macro-F1 heatmap across the seven feature configurations and five classifier families.
- `Figure_2_confusion_matrices` — final-test confusion matrices for the fixed F4 Random Forest and F7 LightGBM models.
- `Figure_3_magnitude_robustness` — magnitude-dependent macro-F1 and STAR recall for F4 Random Forest and F7 LightGBM, including GALAXY/STAR/QSO counts for Q1–Q5.
- `Figure_4_reliability_curves` — reliability curves for F4 Random Forest and F7 LightGBM comparing raw, sigmoid, and isotonic probabilities with 95% Wilson confidence intervals.
- `Figure_5_grouped_permutation_importance` — grouped permutation importance measured as the decrease in macro-F1.
- `Supplementary_Figure_S1_STAR_class_conditional_recall` — STAR class-conditional recall across r-band magnitude bins.

The principal figure labels therefore correspond directly to Figures 1–5 in the final manuscript. Values are read from the archived machine-readable results rather than entered manually in the plotting code.
