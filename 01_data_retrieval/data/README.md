# Archived analysis data

This directory contains the compressed analysis-data files used by the repository:

- `randomized_object_ids.csv.gz`
- `sdss_dr17_randomized_150k.csv.gz`

The analysis scripts use `pandas.read_csv` and can read the `.csv.gz` files directly; decompression is not required. SHA-256 values for both compressed and uncompressed forms are listed in `../DATA_FILE_SHA256.csv`.

To verify the archived files directly:

```powershell
python ../validate_data_archive.py --data sdss_dr17_randomized_150k.csv.gz --ids randomized_object_ids.csv.gz
```

The combined table contains 150,000 sources, balanced across GALAXY, STAR, and QSO (50,000 per class). The identifier table contains the corresponding `objid`, `specobjid`, and class fields for all 150,000 sources.
