#!/usr/bin/env python3
# Maintainer: Yunis Torun
"""
step0_randomized_download_sdss_dr17.py

Data download for:
"Reliability-Aware Star–Galaxy–Quasar Classification in SDSS DR17"

Purpose
-------
Replace the previous class-wise:
    SELECT TOP 50000 ... ORDER BY p.objID
sampling with SQL-level randomized sampling:
    SELECT TOP 50000 ... ORDER BY NEWID()

The script also downloads SDSS photometric uncertainties
(Err_u, Err_g, Err_r, Err_i, Err_z) for the downstream
uncertainty analysis. These uncertainty columns are NOT automatically
added to the original F1–F7 feature sets.

Important
---------
1) The final CSV itself should be archived with the reproducibility package,
   because SQL Server NEWID() is random and not seedable.
2) The script writes the exact SQL queries used to a text file.
3) The downloaded dataset is shuffled reproducibly in Python with
   random_state=42 after class-wise random retrieval.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

SKYSERVER_SQL_URL = (
    "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
)

CLASSES = ("GALAXY", "STAR", "QSO")
ROWS_PER_CLASS = 50_000
RANDOM_STATE = 42
MAG_MIN_SQL = 0
MAG_MAX_SQL = 35

OUT_DIR = Path("sdss_dr17_random_sample")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_CSV = OUT_DIR / "sdss_dr17_randomized_150k.csv"
QUERY_LOG = OUT_DIR / "sql_queries_used.txt"
SUMMARY_JSON = OUT_DIR / "download_summary.json"

REQUEST_TIMEOUT = 300
MAX_RETRIES = 3


def build_query(class_name: str) -> str:
    """
    Randomly retrieve up to ROWS_PER_CLASS eligible sources for one class.

    NEWID() performs SQL-level random ordering, directly addressing the
    sampling concern associated with TOP N ordered by objID.
    """
    return f"""
SELECT TOP {ROWS_PER_CLASS}
    p.objID                 AS objid,
    p.ra                    AS ra,
    p.dec                   AS dec,
    p.u                     AS u,
    p.g                     AS g,
    p.r                     AS r,
    p.i                     AS i,
    p.z                     AS z,
    p.Err_u                 AS err_u,
    p.Err_g                 AS err_g,
    p.Err_r                 AS err_r,
    p.Err_i                 AS err_i,
    p.Err_z                 AS err_z,
    s.specObjID             AS specobjid,
    s.class                 AS class,
    s.z                     AS redshift,
    s.zErr                  AS redshift_err,
    s.plate                 AS plate,
    s.mjd                   AS mjd,
    s.fiberID               AS fiberid
FROM PhotoObj AS p
JOIN SpecObj AS s
    ON s.bestObjID = p.objID
WHERE
    s.class = '{class_name}'
    AND s.zWarning = 0
    AND p.u BETWEEN {MAG_MIN_SQL} AND {MAG_MAX_SQL}
    AND p.g BETWEEN {MAG_MIN_SQL} AND {MAG_MAX_SQL}
    AND p.r BETWEEN {MAG_MIN_SQL} AND {MAG_MAX_SQL}
    AND p.i BETWEEN {MAG_MIN_SQL} AND {MAG_MAX_SQL}
    AND p.z BETWEEN {MAG_MIN_SQL} AND {MAG_MAX_SQL}
ORDER BY NEWID()
""".strip()


def query_sdss(sql: str) -> pd.DataFrame:
    params = {"cmd": sql, "format": "csv"}
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  HTTP attempt {attempt}/{MAX_RETRIES} ...")
            response = requests.get(
                SKYSERVER_SQL_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            text = response.text.strip()
            if not text:
                raise RuntimeError("SkyServer returned an empty response.")
            lowered = text.lower()
            if lowered.startswith("error") or "sql syntax error" in lowered:
                raise RuntimeError(f"SkyServer SQL error:\n{text[:2000]}")
            lines = text.splitlines()
            while lines and lines[0].strip().lower().startswith("#table"):
                print(f"  Removing SkyServer metadata line: {lines[0].strip()}")
                lines.pop(0)
            csv_text = "\n".join(lines).strip()
            if not csv_text:
                raise RuntimeError(
                    "SkyServer response contained no CSV data after metadata removal."
                )
            try:
                df = pd.read_csv(io.StringIO(csv_text), low_memory=False)
            except (IndexError, pd.errors.ParserError):
                print("  Default CSV parser failed; retrying with Python engine ...")
                df = pd.read_csv(
                    io.StringIO(csv_text),
                    engine="python",
                    on_bad_lines="warn",
                )
            if list(df.columns) == ["#Table1"]:
                raise RuntimeError(
                    "SkyServer CSV response was not parsed correctly. "
                    "The response began with '#Table1' but no usable header was found."
                )
            if df.empty:
                raise RuntimeError("Query returned zero rows.")
            return df
        except Exception as exc:
            last_error = exc
            print(f"  Query attempt failed: {exc}")
            if attempt < MAX_RETRIES:
                wait_s = 10 * attempt
                print(f"  Retrying in {wait_s} s ...")
                time.sleep(wait_s)
    raise RuntimeError(f"Query failed after {MAX_RETRIES} attempts.") from last_error


def clean_class_frame(df: pd.DataFrame, requested_class: str) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = [
        "objid", "ra", "dec",
        "u", "g", "r", "i", "z",
        "err_u", "err_g", "err_r", "err_i", "err_z",
        "specobjid", "class", "redshift", "redshift_err",
        "plate", "mjd", "fiberid",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns for {requested_class}: {missing}\n"
            f"Returned columns: {list(df.columns)}"
        )
    numeric_cols = [
        "ra", "dec", "u", "g", "r", "i", "z",
        "err_u", "err_g", "err_r", "err_i", "err_z",
        "redshift", "redshift_err",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["class"] = (
        df["class"].astype(str).str.strip().str.upper().replace({"QUASAR": "QSO"})
    )
    mag_cols = ["u", "g", "r", "i", "z"]
    valid = df[required].notna().all(axis=1)
    for c in mag_cols:
        valid &= (df[c] > 0) & (df[c] < 35)
    valid &= df["class"].eq(requested_class)
    cleaned = df.loc[valid].copy()
    cleaned = cleaned.drop_duplicates(subset=["objid", "class"], keep="first")
    return cleaned


def main() -> None:
    print("=" * 72)
    print("SDSS DR17 randomized download")
    print("=" * 72)
    print(f"Target: {ROWS_PER_CLASS:,} rows per class")
    print(f"Classes: {', '.join(CLASSES)}")
    print("Sampling: SQL-level ORDER BY NEWID()")
    frames = []
    query_texts = []
    class_summary = {}
    for cls in CLASSES:
        print(f"\n[{cls}] Building query ...")
        sql = build_query(cls)
        query_texts.append(f"\n\n-- {cls}\n{sql}\n")
        print(f"[{cls}] Downloading randomized sample ...")
        raw = query_sdss(sql)
        print(f"[{cls}] Raw rows returned: {len(raw):,}")
        cleaned = clean_class_frame(raw, cls)
        print(f"[{cls}] Rows after preprocessing/deduplication: {len(cleaned):,}")
        class_path = OUT_DIR / f"sdss_dr17_random_{cls.lower()}.csv"
        cleaned.to_csv(class_path, index=False)
        class_summary[cls] = {
            "raw_rows": int(len(raw)),
            "clean_rows": int(len(cleaned)),
            "class_csv": str(class_path),
        }
        frames.append(cleaned)
    print("\nCombining classes ...")
    full = pd.concat(frames, ignore_index=True)
    dup_objid = full["objid"].duplicated(keep=False)
    n_cross_duplicates = int(dup_objid.sum())
    if n_cross_duplicates:
        dup_path = OUT_DIR / "cross_class_duplicate_objids.csv"
        full.loc[dup_objid].sort_values("objid").to_csv(dup_path, index=False)
        print(
            f"WARNING: {n_cross_duplicates:,} rows participate in duplicated objIDs. "
            f"Saved to {dup_path}"
        )
        full = full.drop_duplicates(subset=["objid", "class"], keep="first")
    full = full.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    print("\nFinal class counts:")
    counts = full["class"].value_counts().reindex(CLASSES, fill_value=0)
    print(counts.to_string())
    full.to_csv(FINAL_CSV, index=False)
    QUERY_LOG.write_text(
        (
            "SDSS DR17 randomized sampling queries\n"
            "Sampling mechanism: ORDER BY NEWID()\n"
            "NOTE: NEWID() is random and not seedable; archive the resulting "
            "CSV and object IDs for exact reproducibility.\n"
            + "".join(query_texts)
        ),
        encoding="utf-8",
    )
    summary = {
        "skyserver_endpoint": SKYSERVER_SQL_URL,
        "rows_per_class_requested": ROWS_PER_CLASS,
        "classes": list(CLASSES),
        "sql_sampling": "ORDER BY NEWID()",
        "python_shuffle_random_state": RANDOM_STATE,
        "sql_magnitude_constraint": f"BETWEEN {MAG_MIN_SQL} AND {MAG_MAX_SQL}",
        "python_magnitude_constraint": "0 < m < 35",
        "quality_constraint": "s.zWarning = 0",
        "photometric_uncertainties_downloaded": [
            "err_u", "err_g", "err_r", "err_i", "err_z"
        ],
        "redshift_uncertainty_downloaded": "redshift_err",
        "final_rows": int(len(full)),
        "final_class_counts": {k: int(v) for k, v in counts.items()},
        "cross_class_duplicate_rows_detected": n_cross_duplicates,
        "class_downloads": class_summary,
        "final_csv": str(FINAL_CSV),
        "sql_query_log": str(QUERY_LOG),
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nDone.")
    print(f"Final dataset : {FINAL_CSV}")
    print(f"SQL log       : {QUERY_LOG}")
    print(f"Summary       : {SUMMARY_JSON}")
    print("\nNext analysis step:")
    print("Compare old vs randomized samples in r, redshift, RA/Dec, and class support.")


if __name__ == "__main__":
    main()
