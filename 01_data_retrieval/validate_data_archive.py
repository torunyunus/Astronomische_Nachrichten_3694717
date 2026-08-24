# Maintainer: Yunis Torun
# Analysis period: 17-24 August 2026

from __future__ import annotations

import argparse
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
EXPECTED_DATA_SHA256 = "c9c62ab3ec627d01e5da7dcba63a0f664ce37c027108bd3fe7a187923843bfe5"
EXPECTED_IDS_SHA256 = "a74888197d77256ce332754047bc773cf8f7cf007699e0c7d5f832cfe5e12ed5"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to sdss_dr17_randomized_150k.csv")
    ap.add_argument("--ids", required=True, help="Path to randomized_object_ids.csv")
    args = ap.parse_args()

    data_path = Path(args.data)
    ids_path = Path(args.ids)

    data_hash = sha256(data_path)
    ids_hash = sha256(ids_path)
    print("data sha256:", data_hash)
    print("ids  sha256:", ids_hash)

    if data_hash != EXPECTED_DATA_SHA256:
        raise RuntimeError("Analysis-table SHA-256 does not match the archived reference.")
    if ids_hash != EXPECTED_IDS_SHA256:
        raise RuntimeError("Identifier-table SHA-256 does not match the archived reference.")

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
