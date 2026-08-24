# Maintainer: Yunis Torun
# Analysis period: 17-24 August 2026

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from analysis_common import RANDOM_STATE, prepare_dataframe, make_model


def equal_count_bins(values, n_bins):
    order = np.argsort(np.asarray(values), kind="mergesort")
    out = np.empty(len(order), dtype=int)
    for bid, idx in enumerate(np.array_split(order, n_bins), start=1):
        out[idx] = bid
    return out


def bootstrap_metrics(y, pred, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    cls = np.unique(y)
    idx_by_class = {c: np.flatnonzero(y == c) for c in cls}
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(idx_by_class[c], size=len(idx_by_class[c]), replace=True)
            for c in cls
        ])
        yt, yp = y[idx], pred[idx]
        rec = recall_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        rows.append([
            accuracy_score(yt, yp),
            f1_score(yt, yp, average="macro", zero_division=0),
            matthews_corrcoef(yt, yp),
            rec[0], rec[1], rec[2],
        ])
    a = np.asarray(rows)
    return np.quantile(a, 0.025, axis=0), np.quantile(a, 0.975, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="photometric_uncertainty_output")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, fs = prepare_dataframe(pd.read_csv(args.data))
    if "err_r" not in df.columns:
        raise ValueError("Input CSV must contain err_r.")
    for c in ["err_u", "err_g", "err_r", "err_i", "err_z"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    dev, test = train_test_split(
        df, test_size=0.20, stratify=df["target"], random_state=RANDOM_STATE
    )
    dev = dev.reset_index(drop=True)
    test = test.reset_index(drop=True)
    test = test[np.isfinite(test["err_r"].to_numpy(dtype=float)) & (test["err_r"].to_numpy(dtype=float) > 0)].reset_index(drop=True)
    test["uncertainty_tertile_id"] = equal_count_bins(test["err_r"].to_numpy(), 3)
    test["r_bin_id"] = equal_count_bins(test["r"].to_numpy(), 5)

    selected = {
        "F4": ("F4 Mag+Colors", "Random Forest"),
        "F7": ("F7 Full", "LightGBM"),
    }
    overall_rows = []
    joint_rows = []
    star_rows = []

    for tag, (fs_name, model_name) in selected.items():
        model = make_model(model_name)
        model.fit(dev[fs[fs_name]].to_numpy(dtype=float), dev["target"].to_numpy())
        pred_all = model.predict(test[fs[fs_name]].to_numpy(dtype=float))
        proba_all = model.predict_proba(test[fs[fs_name]].to_numpy(dtype=float))

        for tid, name in [(1, "Low"), (2, "Medium"), (3, "High")]:
            m = test["uncertainty_tertile_id"].eq(tid).to_numpy()
            d = test.loc[m]
            y, pred, proba = d["target"].to_numpy(), pred_all[m], proba_all[m]
            rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
            lo, hi = bootstrap_metrics(y, pred, args.bootstrap, RANDOM_STATE + tid + (100 if tag == "F7" else 0))
            counts = d["class"].value_counts()
            overall_rows.append({
                "feature_configuration": tag,
                "model": model_name,
                "uncertainty_tertile_id": tid,
                "uncertainty_tertile": name,
                "n": len(d),
                "err_r_min": d["err_r"].min(),
                "err_r_max": d["err_r"].max(),
                "err_r_median": d["err_r"].median(),
                "snr_r_median": 1.0857 / d["err_r"].median(),
                "galaxy_n": int(counts.get("GALAXY", 0)),
                "star_n": int(counts.get("STAR", 0)),
                "qso_n": int(counts.get("QSO", 0)),
                "accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                "mcc": matthews_corrcoef(y, pred),
                "auc_macro_ovr": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
                "recall_galaxy": rec[0], "recall_star": rec[1], "recall_qso": rec[2],
                "accuracy_ci95_low": lo[0], "accuracy_ci95_high": hi[0],
                "macro_f1_ci95_low": lo[1], "macro_f1_ci95_high": hi[1],
                "mcc_ci95_low": lo[2], "mcc_ci95_high": hi[2],
                "recall_galaxy_ci95_low": lo[3], "recall_galaxy_ci95_high": hi[3],
                "recall_star_ci95_low": lo[4], "recall_star_ci95_high": hi[4],
                "recall_qso_ci95_low": lo[5], "recall_qso_ci95_high": hi[5],
            })

        for rbin in range(1, 6):
            ids = np.flatnonzero(test["r_bin_id"].eq(rbin).to_numpy())
            local = test.iloc[ids].copy()
            local["within_r_uncertainty_id"] = equal_count_bins(local["err_r"].to_numpy(), 3)
            for tid, name in [(1, "Low"), (2, "Medium"), (3, "High")]:
                loc = local["within_r_uncertainty_id"].eq(tid).to_numpy()
                idx = ids[loc]
                d = test.iloc[idx]
                y, pred = d["target"].to_numpy(), pred_all[idx]
                rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
                joint_rows.append({
                    "feature_configuration": tag,
                    "model": model_name,
                    "r_bin_id": rbin,
                    "r_bin": f"Q{rbin}",
                    "uncertainty_tertile_id": tid,
                    "uncertainty_tertile": name,
                    "n": len(d),
                    "r_median": d["r"].median(),
                    "err_r_median": d["err_r"].median(),
                    "snr_r_median": 1.0857 / d["err_r"].median(),
                    "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                    "recall_galaxy": rec[0], "recall_star": rec[1], "recall_qso": rec[2],
                })

                s = d["target"].eq(1).to_numpy()
                if s.any():
                    star_rows.append({
                        "feature_configuration": tag,
                        "model": model_name,
                        "r_bin_id": rbin,
                        "r_bin": f"Q{rbin}",
                        "uncertainty_tertile_id": tid,
                        "uncertainty_tertile": name,
                        "n_star": int(s.sum()),
                        "star_recall": float(np.mean(pred[s] == 1)),
                        "r_median": d.loc[s, "r"].median(),
                        "err_r_median": d.loc[s, "err_r"].median(),
                    })

    pd.DataFrame(overall_rows).to_csv(out / "uncertainty_tertile_results.csv", index=False)
    pd.DataFrame(joint_rows).to_csv(out / "magnitude_uncertainty_results.csv", index=False)
    pd.DataFrame(star_rows).to_csv(out / "star_magnitude_uncertainty_results.csv", index=False)

    relation = (
        test.groupby("class", as_index=False)
        .agg(n=("err_r", "size"), r_median=("r", "median"), err_r_median=("err_r", "median"))
    )
    relation["snr_r_median"] = 1.0857 / relation["err_r_median"]
    relation.to_csv(out / "r_errr_relationship_by_class.csv", index=False)


if __name__ == "__main__":
    main()
