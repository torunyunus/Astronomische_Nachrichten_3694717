# Archived analysis data

This directory is intended for the compressed analysis-data files:

- `randomized_object_ids.csv.gz`
- `sdss_dr17_randomized_150k.csv.gz`

After downloading, decompress them before running the analysis scripts. The SHA-256 values for both compressed and uncompressed forms are listed in `../DATA_FILE_SHA256.csv`.

To verify the decompressed files:

```powershell
python ../validate_data_archive.py --data sdss_dr17_randomized_150k.csv --ids randomized_object_ids.csv
```

The combined table contains 150,000 sources, balanced across GALAXY, STAR, and QSO (50,000 per class).
