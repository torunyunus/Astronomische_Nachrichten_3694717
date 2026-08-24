# Reference software environment

The analysis workflow is implemented in Python. The following package versions provide a reference environment compatible with the repository scripts:

- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- SciPy 1.17.0
- scikit-learn 1.8.0
- LightGBM 4.6.0
- Matplotlib 3.10.8
- joblib 1.5.3
- requests 2.32.5

For an exact snapshot of the installed `astro_scie` environment on the analysis computer, run `environment/export_environment.ps1`. This creates `requirements.txt` and `environment.yml` directly from the active installation.
