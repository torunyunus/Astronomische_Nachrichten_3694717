# Data archive

The SDSS DR17 source catalogue is public. The randomized class-specific SQL queries used for this analysis are stored in `sql_queries_used.txt`.

SQL Server `ORDER BY NEWID()` is not seedable. Re-running the same queries reproduces the sampling procedure but does not guarantee the identical set of objects. Exact numerical reconstruction therefore uses the archived object identifiers and the combined randomized sample stored in `data/`.

Archived files:

- `data/randomized_object_ids.csv.gz` — compressed identifier table for the analysed 150,000 sources (`objid`, `specobjid`, and class).
- `data/sdss_dr17_randomized_150k.csv.gz` — compressed combined analysis table containing positional, photometric, uncertainty, spectroscopic-class, and redshift fields.

The randomized source table contains 50,000 GALAXY, 50,000 STAR, and 50,000 QSO objects. The Python-side shuffle and subsequent development/final-test split use `random_state=42`.

SHA-256 checksums for both compressed and uncompressed forms are listed in `DATA_FILE_SHA256.csv`. The utility `validate_data_archive.py` verifies compressed-file integrity, decompressed-content integrity, row counts, class balance, identifier consistency, and the expected input schema before analysis.

Direct archive validation:

```powershell
python validate_data_archive.py --data data/sdss_dr17_randomized_150k.csv.gz --ids data/randomized_object_ids.csv.gz
```
