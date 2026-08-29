# Maintainer: Yunis Torun

from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "objid", "ra", "dec", "u", "g", "r", "i", "z",
    "err_u", "err_g", "err_r", "err_i", "err_z",
    "specobjid", "class", "redshift", "redshift_err",
    "plate", "mjd", "fiberid",
]
EXPECTED_COUNTS = {"GALAXY": 50000, "STAR": 50000, "QSO": 50000}

EXPECTED_HASHES = {
    "sdss_dr17_randomized_150k.csv": "c9c62ab3ec627d01e5da7dcba63a0f664ce37c027108bd3fe7a187923843bfe5",
    "sdss_dr17_randomized_150k.csv.gz": "32b55835601a3818b0e664436b4f3f4f50a168a32c09cf799322cef203602059",
    "randomized_object_ids.csv": "a74888197d77256ce332754047bc773cf8f7cf007699e0c7d5f832cfe5e12ed5",
    "randomized_object_ids.csv.gz": "14ffa9c0033aea398818303c56cd49ba7f1f1c29fbb4697cb9732226af665ffe",
}

EXPECTED_DECOMPRESSED_HASHES = {
    "sdss_dr17_randomized_150k.csv.gz": EXPECTED_HASHES["sdss_dr17_randomized_150k.csv"],
    "randomized_object_ids.csv.gz": EXPECTED_HASHES["randomized_object_ids.csv"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_decompressed_gzip(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_hash(path: Path) -> None:
    expected = EXPECTED_HASHES.get(path.name)
    if expected is None:
        raise RuntimeError(f"Unrecognized archive filename: {path.name}")

    actual = sha256(path)
    print(f"{path.name} sha256:", actual)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path.name}")

    if path.suffix == ".gz":
        expected_uncompressed = EXPECTED_DECOMPRESSED_HASHES[path.name]
        actual_uncompressed = sha256_decompressed_gzip(path)
        print(f"{path.name} decompressed sha256:", actual_uncompressed)
        if actual_uncompressed != expected_uncompressed:
            raise RuntimeError(f"Decompressed SHA-256 mismatch for {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to sdss_dr17_randomized_150k.csv or .csv.gz")
    ap.add_argument("--ids", required=True, help="Path to randomized_object_ids.csv or .csv.gz")
    args = ap.parse_args()

    data_path = Path(args.data)
    ids_path = Path(args.ids)

    verify_hash(data_path)
    verify_hash(ids_path)

    df = pd.read_csv(data_path)
    ids = pd.read_csv(ids_path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing analysis-table columns: {missing}")

    if len(df) != 150000:
        raise RuntimeError(f"Expected 150000 analysis rows, found {len(df)}")
    if len(ids) != 150000:
        raise RuntimeError(f"Expected 150000 identifier rows, found {len(ids)}")

    counts = df["class"].astype(str).str.upper().value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Unexpected class counts: {counts}")

    required_id_cols = {"objid", "specobjid", "class"}
    if not required_id_cols.issubset(ids.columns):
        raise RuntimeError("Identifier table must contain objid, specobjid, and class.")

    if df["objid"].duplicated().any():
        raise RuntimeError("Duplicate objid values detected in the analysis table.")
    if df["specobjid"].duplicated().any():
        raise RuntimeError("Duplicate specobjid values detected in the analysis table.")

    numeric = df.select_dtypes(include="number")
    if numeric.isna().any().any():
        raise RuntimeError("Missing numeric values detected in the analysis table.")

    a = df[["objid", "specobjid", "class"]].copy()
    b = ids[["objid", "specobjid", "class"]].copy()
    a["class"] = a["class"].astype(str).str.upper()
    b["class"] = b["class"].astype(str).str.upper()
    a = a.sort_values(["objid", "specobjid", "class"]).reset_index(drop=True)
    b = b.sort_values(["objid", "specobjid", "class"]).reset_index(drop=True)
    if not a.equals(b):
        raise RuntimeError("Identifier table does not match the analysis table.")

    print("Archive validation: PASS")
    print("Rows: 150000")
    print("Class counts:", counts)
    print("Schema columns:", len(EXPECTED_COLUMNS))


if __name__ == "__main__":
    main()
