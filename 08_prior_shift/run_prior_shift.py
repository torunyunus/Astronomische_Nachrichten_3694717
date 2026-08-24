# Maintainer: Yunis Torun
# Analysis period: 17-24 August 2026

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from analysis_common import RANDOM_STATE, prepare_dataframe, make_model

BALANCED_PRIOR = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
SDSS_REFERENCE_PRIOR = np.array([0.3724535025530711, 0.34686081192418167, 0.28068568552274725], dtype=float)


def normalize_rows(p):
    p = np.asarray(p, dtype=float)
    s = p.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return p / s


def prior_adjust_probabilities(p, target_prior, source_prior=BALANCED_PRIOR):
    p = normalize_rows(p)
    adjusted = p * (np.asarray(target_prior) / np.asarray(source_prior))[None, :]
    return normalize_rows(adjusted)


def population_cm(cm_counts, priors):
    cm = np.asarray(cm_counts, dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_probs = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    return np.asarray(priors, dtype=float)[:, None] * row_probs


def metrics_from_cm(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    total = cm.sum()
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) > 0)
    accuracy = tp.sum() / total
    macro_f1 = f1.mean()
    weighted_f1 = np.sum(f1 * support) / support.sum()

    t_sum = support
    p_sum = predicted
    cov_ytyp = tp.sum() * total - np.dot(t_sum, p_sum)
    cov_ypyp = total * total - np.dot(p_sum, p_sum)
    cov_ytyt = total * total - np.dot(t_sum, t_sum)
    denom = np.sqrt(cov_ytyt * cov_ypyp)
    mcc = cov_ytyp / denom if denom > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mcc": float(mcc),
        "precision_galaxy": float(precision[0]),
        "precision_star": float(precision[1]),
        "precision_qso": float(precision[2]),
        "recall_galaxy": float(recall[0]),
        "recall_star": float(recall[1]),
        "recall_qso": float(recall[2]),
        "f1_galaxy": float(f1[0]),
        "f1_star": float(f1[1]),
        "f1_qso": float(f1[2]),
    }


def bootstrap_population(cm_counts, priors, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    cm_counts = np.asarray(cm_counts, dtype=int)
    rows = []
    for _ in range(n_boot):
        sampled = np.zeros((3, 3), dtype=int)
        for k in range(3):
            n = int(cm_counts[k].sum())
            probs = cm_counts[k] / n
            sampled[k] = rng.multinomial(n, probs)
        rows.append(metrics_from_cm(population_cm(sampled, priors)))
    d = pd.DataFrame(rows)
    return {
        c: (float(d[c].quantile(0.025)), float(d[c].quantile(0.975)))
        for c in d.columns
    }


def evaluate_scenario(config, model_name, y, pred, proba, scenario, priors, decision_rule, n_boot, seed):
    if decision_rule == "Prior-adjusted":
        pred_eval = np.argmax(prior_adjust_probabilities(proba, priors), axis=1)
    else:
        pred_eval = pred
    cm = confusion_matrix(y, pred_eval, labels=[0, 1, 2])
    met = metrics_from_cm(population_cm(cm, priors))
    ci = bootstrap_population(cm, priors, n_boot, seed)
    row = {
        "feature_configuration": config,
        "model": model_name,
        "scenario": scenario,
        "decision_rule": decision_rule,
        "galaxy_prior": priors[0],
        "star_prior": priors[1],
        "qso_prior": priors[2],
        **met,
    }
    for name, (lo, hi) in ci.items():
        row[f"{name}_ci95_low"] = lo
        row[f"{name}_ci95_high"] = hi
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="prior_shift_output")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, fs = prepare_dataframe(pd.read_csv(args.data))
    dev, test = train_test_split(
        df, test_size=0.20, stratify=df["target"], random_state=RANDOM_STATE
    )
    dev, test = dev.reset_index(drop=True), test.reset_index(drop=True)
    y = test["target"].to_numpy()

    selected = {
        "F4": ("F4 Mag+Colors", "Random Forest"),
        "F7": ("F7 Full", "LightGBM"),
    }
    scenarios = [
        ("Balanced study prior", BALANCED_PRIOR),
        ("Published SDSS DR17 spectroscopic reference", SDSS_REFERENCE_PRIOR),
    ]
    rows = []

    for config, (fs_name, model_name) in selected.items():
        model = make_model(model_name)
        model.fit(dev[fs[fs_name]].to_numpy(dtype=float), dev["target"].to_numpy())
        Xt = test[fs[fs_name]].to_numpy(dtype=float)
        pred = model.predict(Xt)
        proba = model.predict_proba(Xt)
        for sidx, (scenario, priors) in enumerate(scenarios):
            for ridx, rule in enumerate(["Original", "Prior-adjusted"]):
                rows.append(evaluate_scenario(
                    config, model_name, y, pred, proba, scenario, priors, rule,
                    args.bootstrap,
                    RANDOM_STATE + sidx * 1000 + ridx * 100 + (10000 if config == "F7" else 0),
                ))

    pd.DataFrame(rows).to_csv(out / "prior_shift_results.csv", index=False)


if __name__ == "__main__":
    main()
