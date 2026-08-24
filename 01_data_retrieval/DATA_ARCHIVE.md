# Data archive

The SDSS DR17 source catalogue is public. The randomized class-specific SQL queries used for this analysis are stored in `sql_queries_used.txt`.

SQL Server `ORDER BY NEWID()` is not seedable. Re-running the same queries therefore reproduces the sampling procedure but does not guarantee the identical set of objects. Exact numerical reproduction requires archiving the analysed object identifiers and the combined randomized sample together with the code.

Expected data files:

- `randomized_object_ids.csv` — identifiers for the analysed 150,000 sources.
- `sdss_dr17_randomized_150k.csv` — combined analysis table containing the photometric, uncertainty, positional, spectroscopic-class, and redshift fields used by the scripts.

The randomized source table contains 50,000 GALAXY, 50,000 STAR, and 50,000 QSO objects. The Python-side shuffle and subsequent development/final-test split use `random_state=42`.
