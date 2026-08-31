# Maintainer: Yunis Torun

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

from analysis_common import RANDOM_STATE, prepare_dataframe, make_model


def normalize_rows(p, eps=1e-15):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return p / p.sum(axis=1, keepdims=True)


def multiclass_brier(y, p):
    onehot = np.eye(3)[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((normalize_rows(p) - onehot) ** 2, axis=1)))


def top_label_ece(y, p, n_bins):
    p = normalize_rows(p)
    pred = np.argmax(p, axis=1)
    conf = np.max(p, axis=1)
    correct = (pred == np.asarray(y)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bid = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bid == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def metrics(y, p):
    p = normalize_rows(p)
    pred = np.argmax(p, axis=1)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier_score": multiclass_brier(y, p),
        "ece_10": top_label_ece(y, p, 10),
        "ece_15": top_label_ece(y, p, 15),
        "ece_20": top_label_ece(y, p, 20),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def stable_sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def binary_loss(a, b, x, y, ridge):
    z = a * x + b
    return float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * ridge * a * a)


def fit_platt(prob, target, max_iter=100, tol=1e-10, ridge=1e-6):
    eps = 1e-7
    p = np.clip(np.asarray(prob, dtype=float), eps, 1.0 - eps)
    y = np.asarray(target, dtype=float)
    x = np.log(p / (1.0 - p))
    a, b = 1.0, 0.0
    current = binary_loss(a, b, x, y, ridge)
    for _ in range(max_iter):
        q = stable_sigmoid(a * x + b)
        diff = q - y
        w = np.clip(q * (1.0 - q), 1e-12, None)
        ga = float(np.mean(diff * x) + ridge * a)
        gb = float(np.mean(diff))
        haa = float(np.mean(w * x * x) + ridge)
        hab = float(np.mean(w * x))
        hbb = float(np.mean(w) + 1e-12)
        det = haa * hbb - hab * hab
        if not np.isfinite(det) or abs(det) < 1e-18:
            break
        da = (hbb * ga - hab * gb) / det
        db = (-hab * ga + haa * gb) / det
        if max(abs(da), abs(db)) < tol:
            break
        scale = 1.0
        accepted = False
        for _ in range(30):
            na, nb = a - scale * da, b - scale * db
            loss = binary_loss(na, nb, x, y, ridge)
            if np.isfinite(loss) and loss <= current + 1e-14:
                a, b, current = na, nb, loss
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    return a, b


def fit_sigmoid_maps(oof, y):
    return [fit_platt(oof[:, k], (y == k).astype(float)) for k in range(3)]


def apply_sigmoid(maps, p):
    eps = 1e-7
    out = np.zeros_like(p, dtype=float)
    for k, (a, b) in enumerate(maps):
        pk = np.clip(p[:, k], eps, 1.0 - eps)
        x = np.log(pk / (1.0 - pk))
        out[:, k] = stable_sigmoid(a * x + b)
    return normalize_rows(out)


def fit_isotonic_maps(oof, y):
    maps = []
    for k in range(3):
        iso = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        iso.fit(oof[:, k], (y == k).astype(float))
        maps.append(iso)
    return maps


def apply_isotonic(maps, p):
    out = np.column_stack([maps[k].predict(p[:, k]) for k in range(3)])
    return normalize_rows(out)


def oof_probabilities(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full((len(y), 3), np.nan)
    for tr, va in cv.split(X, y):
        m = clone(model)
        m.fit(X[tr], y[tr])
        oof[va] = normalize_rows(m.predict_proba(X[va]))
    return oof


def bootstrap_ci(y, method_probs, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    idx_by_class = {k: np.flatnonzero(y == k) for k in [0, 1, 2]}
    store = {name: [] for name in method_probs}
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(idx_by_class[k], size=len(idx_by_class[k]), replace=True)
            for k in [0, 1, 2]
        ])
        for name, p in method_probs.items():
            store[name].append(metrics(y[idx], p[idx]))
    out = {}
    for name, rows in store.items():
        d = pd.DataFrame(rows)
        out[name] = {
            c: (float(d[c].quantile(0.025)), float(d[c].quantile(0.975)))
            for c in d.columns
        }
    return out


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return center - half, center + half


def reliability_rows(config, method, y, p, n_bins=15):
    p = normalize_rows(p)
    pred = np.argmax(p, axis=1)
    conf = np.max(p, axis=1)
    correct = pred == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bid = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bid == b
        n = int(m.sum())
        if n:
            s = int(correct[m].sum())
            lo, hi = wilson(s, n)
            mean_confidence = float(conf[m].mean())
            empirical_accuracy = float(correct[m].mean())
        else:
            lo = hi = mean_confidence = empirical_accuracy = np.nan
        rows.append({
            "feature_configuration": config,
            "method": method,
            "n_bins": n_bins,
            "bin_id": b + 1,
            "bin_lower": edges[b],
            "bin_upper": edges[b + 1],
            "n": n,
            "mean_confidence": mean_confidence,
            "empirical_accuracy": empirical_accuracy,
            "accuracy_ci95_low": lo,
            "accuracy_ci95_high": hi,
        })
    return rows


def write_table_7(out: Path, results: pd.DataFrame) -> None:
    rows = []
    for r in results.itertuples(index=False):
        rows.append({
            "Configuration": r.feature_configuration,
            "Model": r.model,
            "Method": r.method,
            "Macro-F1": f"{r.macro_f1:.4f}",
            "Log loss (95% CI)": (
                f"{r.log_loss:.4f} ({r.log_loss_ci95_low:.4f}–{r.log_loss_ci95_high:.4f})"
            ),
            "Brier score (95% CI)": (
                f"{r.brier_score:.4f} ({r.brier_score_ci95_low:.4f}–{r.brier_score_ci95_high:.4f})"
            ),
            "ECE15 (95% CI)": (
                f"{r.ece_15:.4f} ({r.ece_15_ci95_low:.4f}–{r.ece_15_ci95_high:.4f})"
            ),
            "ECE10": f"{r.ece_10:.4f}",
            "ECE20": f"{r.ece_20:.4f}",
        })
    pd.DataFrame(rows).to_csv(out / "Table_7_probability_calibration.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--f4-model", required=True, help="Fixed F4 model from validation stage")
    ap.add_argument("--f7-model", required=True, help="Fixed F7 model from validation stage")
    ap.add_argument("--out", default="probability_calibration_output")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, fs = prepare_dataframe(pd.read_csv(args.data))
    dev, test = train_test_split(
        df, test_size=0.20, stratify=df["target"], random_state=RANDOM_STATE
    )
    dev, test = dev.reset_index(drop=True), test.reset_index(drop=True)

    selected = {
        "F4": ("F4 Mag+Colors", "Random Forest", args.f4_model),
        "F7": ("F7 Full", "LightGBM", args.f7_model),
    }
    result_rows = []
    ece_rows = []
    curve_rows = []

    for config, (fs_name, model_name, model_path) in selected.items():
        Xd = dev[fs[fs_name]].to_numpy(dtype=float)
        yd = dev["target"].to_numpy()
        Xt = test[fs[fs_name]].to_numpy(dtype=float)
        yt = test["target"].to_numpy()

        base = make_model(model_name)
        oof = oof_probabilities(base, Xd, yd)
        sigmoid = fit_sigmoid_maps(oof, yd)
        isotonic = fit_isotonic_maps(oof, yd)

        final_model = joblib.load(model_path)
        raw = normalize_rows(final_model.predict_proba(Xt))
        probs = {
            "Raw": raw,
            "Sigmoid": apply_sigmoid(sigmoid, raw),
            "Isotonic": apply_isotonic(isotonic, raw),
        }
        ci = bootstrap_ci(
            yt,
            probs,
            args.bootstrap,
            RANDOM_STATE + (100000 if config == "F7" else 0),
        )

        for method, p in probs.items():
            met = metrics(yt, p)
            row = {
                "feature_configuration": config,
                "model": model_name,
                "method": method,
                **met,
            }
            for metric_name in [
                "log_loss", "brier_score", "ece_15", "accuracy", "macro_f1"
            ]:
                row[f"{metric_name}_ci95_low"] = ci[method][metric_name][0]
                row[f"{metric_name}_ci95_high"] = ci[method][metric_name][1]
            result_rows.append(row)
            ece_rows.append({
                "feature_configuration": config,
                "model": model_name,
                "method": method,
                "ece_10": met["ece_10"],
                "ece_15": met["ece_15"],
                "ece_20": met["ece_20"],
            })
            curve_rows.extend(reliability_rows(config, method, yt, p, 15))

    results = pd.DataFrame(result_rows)
    results.to_csv(out / "calibration_results.csv", index=False)
    pd.DataFrame(ece_rows).to_csv(out / "ece_sensitivity.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / "reliability_curve_data.csv", index=False)
    write_table_7(out, results)


if __name__ == "__main__":
    main()
