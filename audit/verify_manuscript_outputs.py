# Maintainer: Yunis Torun
"""Verify regenerated Tables 3–8 and Figure 1–5 numerical sources against the current manuscript."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

FAIL = []


def close(actual, expected, decimals=4, label=""):
    tol = 0.5 * 10 ** (-decimals) + 1e-12
    if not np.isfinite(float(actual)) or abs(float(actual) - float(expected)) > tol:
        FAIL.append(f"{label}: got {actual}, expected {expected}")


def exact(actual, expected, label=""):
    if str(actual) != str(expected):
        FAIL.append(f"{label}: got {actual!r}, expected {expected!r}")


# Table 3: repeated-CV mean macro-F1 ± SD.
exp3 = {
    "F1": {"Logistic Regression": (0.9133, 0.0011), "Decision Tree": (0.9245, 0.0011), "GaussianNB": (0.9365, 0.0009), "Random Forest": (0.9283, 0.0012), "LightGBM": (0.9368, 0.0011)},
    "F2": {"Logistic Regression": (0.6982, 0.0020), "Decision Tree": (0.8415, 0.0029), "GaussianNB": (0.5899, 0.0024), "Random Forest": (0.8812, 0.0022), "LightGBM": (0.8601, 0.0017)},
    "F3": {"Logistic Regression": (0.6412, 0.0017), "Decision Tree": (0.8373, 0.0023), "GaussianNB": (0.5680, 0.0018), "Random Forest": (0.8730, 0.0020), "LightGBM": (0.8708, 0.0022)},
    "F4": {"Logistic Regression": (0.6983, 0.0020), "Decision Tree": (0.8609, 0.0015), "GaussianNB": (0.6788, 0.0018), "Random Forest": (0.8954, 0.0017), "LightGBM": (0.8929, 0.0020)},
    "F5": {"Logistic Regression": (0.9462, 0.0009), "Decision Tree": (0.9778, 0.0009), "GaussianNB": (0.9379, 0.0020), "Random Forest": (0.9832, 0.0003), "LightGBM": (0.9827, 0.0005)},
    "F6": {"Logistic Regression": (0.9382, 0.0008), "Decision Tree": (0.9770, 0.0007), "GaussianNB": (0.9452, 0.0010), "Random Forest": (0.9823, 0.0004), "LightGBM": (0.9827, 0.0008)},
    "F7": {"Logistic Regression": (0.9462, 0.0009), "Decision Tree": (0.9791, 0.0006), "GaussianNB": (0.9391, 0.0015), "Random Forest": (0.9847, 0.0004), "LightGBM": (0.9848, 0.0004)},
}

d3 = pd.read_csv("validation_output/repeated_cv_summary.csv")
for _, r in d3.iterrows():
    feature = str(r["feature_configuration"]).split()[0]
    model = r["model"]
    if feature in exp3 and model in exp3[feature]:
        expected_mean, expected_sd = exp3[feature][model]
        close(r["macro_f1_mean"], expected_mean, 4, f"Table3 {feature} {model} mean")
        close(r["macro_f1_sd"], expected_sd, 4, f"Table3 {feature} {model} sd")
    else:
        FAIL.append(f"Table3 unexpected row: {r.to_dict()}")
if len(d3) != 35:
    FAIL.append(f"Table3 row count {len(d3)} != 35")


# Table 4: development-selected model per feature configuration.
exp4 = {
    "F1": ("LightGBM", 0.9368, 0.0011),
    "F2": ("Random Forest", 0.8812, 0.0022),
    "F3": ("Random Forest", 0.8730, 0.0020),
    "F4": ("Random Forest", 0.8954, 0.0017),
    "F5": ("Random Forest", 0.9832, 0.0003),
    "F6": ("LightGBM", 0.9827, 0.0008),
    "F7": ("LightGBM", 0.9848, 0.0004),
}
for feature, (expected_model, expected_mean, expected_sd) in exp4.items():
    sub = d3[d3["feature_configuration"].astype(str).str.startswith(feature + " ")]
    row = sub.loc[sub["macro_f1_mean"].idxmax()]
    exact(row["model"], expected_model, f"Table4 {feature} model")
    close(row["macro_f1_mean"], expected_mean, 4, f"Table4 {feature} mean")
    close(row["macro_f1_sd"], expected_sd, 4, f"Table4 {feature} sd")


# Table 5: untouched final-test performance.
exp5 = {
    "F4": ("Random Forest", 0.8948, 0.8946, 0.8912, 0.8982, 0.8424, 0.9784),
    "F7": ("LightGBM", 0.9851, 0.9851, 0.9836, 0.9864, 0.9777, 0.9987),
}
d5 = pd.read_csv("validation_output/untouched_final_test_results_F4_F7.csv")
for _, r in d5.iterrows():
    feature = str(r["feature_configuration"]).split()[0]
    if feature not in exp5:
        FAIL.append(f"Table5 unexpected configuration {feature}")
        continue
    model, acc, macro_f1, ci_low, ci_high, mcc, auc = exp5[feature]
    exact(r["model"], model, f"Table5 {feature} model")
    for column, value in [
        ("accuracy", acc),
        ("macro_f1", macro_f1),
        ("macro_f1_ci95_low", ci_low),
        ("macro_f1_ci95_high", ci_high),
        ("mcc", mcc),
        ("auc_macro_ovr", auc),
    ]:
        close(r[column], value, 4, f"Table5 {feature} {column}")


# Figure 2: exact confusion-matrix counts.
for feature, expected in {
    "F4": np.array([[9194, 506, 300], [706, 8575, 719], [430, 495, 9075]]),
    "F7": np.array([[9861, 8, 131], [1, 9999, 0], [307, 0, 9693]]),
}.items():
    cm = pd.read_csv(
        f"validation_output/{feature}_final_test_confusion_matrix.csv", index_col=0
    ).to_numpy(dtype=int)
    if not np.array_equal(cm, expected):
        FAIL.append(
            f"Figure2 {feature} confusion matrix mismatch: {cm.tolist()} != {expected.tolist()}"
        )


# Table 6: magnitude robustness.
exp6 = {
    ("F4", "Q1"): (11.120, 17.747, 2620, 3259, 121, 0.9001, 0.8772, 0.9211, 0.9691, 0.9622, 0.9756, 0.9782, 0.9730, 0.9831, 0.6612, 0.5785, 0.7438),
    ("F4", "Q2"): (17.747, 19.374, 1738, 2868, 1394, 0.9454, 0.9392, 0.9513, 0.9264, 0.9137, 0.9384, 0.9651, 0.9582, 0.9718, 0.9383, 0.9254, 0.9512),
    ("F4", "Q3"): (19.374, 20.389, 1834, 1551, 2615, 0.8851, 0.8768, 0.8929, 0.9253, 0.9133, 0.9368, 0.8311, 0.8117, 0.8498, 0.9048, 0.8929, 0.9159),
    ("F4", "Q4"): (20.389, 21.102, 1969, 1198, 2833, 0.8225, 0.8115, 0.8333, 0.9279, 0.9162, 0.9391, 0.6269, 0.5977, 0.6544, 0.9001, 0.8892, 0.9114),
    ("F4", "Q5"): (21.102, 30.062, 1839, 1124, 3037, 0.7610, 0.7486, 0.7726, 0.8271, 0.8102, 0.8439, 0.5151, 0.4858, 0.5436, 0.9124, 0.9019, 0.9220),
    ("F7", "Q1"): (11.120, 17.747, 2620, 3259, 121, 0.9452, 0.9270, 0.9623, 0.9962, 0.9935, 0.9985, 0.9997, 0.9991, 1.0000, 0.7603, 0.6860, 0.8347),
    ("F7", "Q2"): (17.747, 19.374, 1738, 2868, 1394, 0.9926, 0.9902, 0.9950, 0.9908, 0.9862, 0.9954, 1.0000, 1.0000, 1.0000, 0.9864, 0.9799, 0.9921),
    ("F7", "Q3"): (19.374, 20.389, 1834, 1551, 2615, 0.9854, 0.9825, 0.9883, 0.9891, 0.9842, 0.9935, 1.0000, 1.0000, 1.0000, 0.9713, 0.9648, 0.9778),
    ("F7", "Q4"): (20.389, 21.102, 1969, 1198, 2833, 0.9867, 0.9839, 0.9893, 0.9893, 0.9843, 0.9934, 1.0000, 1.0000, 1.0000, 0.9746, 0.9686, 0.9802),
    ("F7", "Q5"): (21.102, 30.062, 1839, 1124, 3037, 0.9732, 0.9694, 0.9771, 0.9608, 0.9516, 0.9701, 1.0000, 1.0000, 1.0000, 0.9631, 0.9562, 0.9700),
}
d6 = pd.read_csv("magnitude_robustness_output/magnitude_bin_results.csv")
columns6 = [
    ("r_min", 3), ("r_max", 3), ("galaxy_n", 0), ("star_n", 0), ("qso_n", 0),
    ("macro_f1", 4), ("macro_f1_ci95_low", 4), ("macro_f1_ci95_high", 4),
    ("recall_galaxy", 4), ("recall_galaxy_ci95_low", 4), ("recall_galaxy_ci95_high", 4),
    ("recall_star", 4), ("recall_star_ci95_low", 4), ("recall_star_ci95_high", 4),
    ("recall_qso", 4), ("recall_qso_ci95_low", 4), ("recall_qso_ci95_high", 4),
]
for _, r in d6.iterrows():
    key = (r["feature_configuration"], r["r_bin"])
    if key not in exp6:
        FAIL.append(f"Table6 unexpected row {key}")
        continue
    for (column, decimals), value in zip(columns6, exp6[key]):
        if decimals == 0:
            if int(r[column]) != int(value):
                FAIL.append(f"Table6 {key} {column}: got {r[column]}, expected {value}")
        else:
            close(r[column], value, decimals, f"Table6 {key} {column}")
if len(d6) != 10:
    FAIL.append(f"Table6 row count {len(d6)} != 10")


# Table 7: probability calibration.
exp7 = {
    ("F4", "Raw"): (0.8946, 0.2877, 0.2776, 0.2980, 0.1546, 0.1507, 0.1585, 0.0104, 0.0081, 0.0135, 0.0104, 0.0104),
    ("F4", "Sigmoid"): (0.8945, 0.2793, 0.2715, 0.2868, 0.1543, 0.1504, 0.1583, 0.0043, 0.0036, 0.0084, 0.0040, 0.0056),
    ("F4", "Isotonic"): (0.8941, 0.2756, 0.2684, 0.2824, 0.1543, 0.1502, 0.1584, 0.0049, 0.0040, 0.0090, 0.0038, 0.0044),
    ("F7", "Raw"): (0.9851, 0.0457, 0.0423, 0.0492, 0.0241, 0.0221, 0.0260, 0.0017, 0.0013, 0.0032, 0.0004, 0.0013),
    ("F7", "Sigmoid"): (0.9851, 0.0457, 0.0423, 0.0491, 0.0241, 0.0221, 0.0260, 0.0018, 0.0012, 0.0033, 0.0004, 0.0013),
    ("F7", "Isotonic"): (0.9850, 0.0466, 0.0426, 0.0510, 0.0241, 0.0222, 0.0260, 0.0019, 0.0013, 0.0034, 0.0006, 0.0016),
}
d7 = pd.read_csv("probability_calibration_output/calibration_results.csv")
columns7 = [
    "macro_f1", "log_loss", "log_loss_ci95_low", "log_loss_ci95_high",
    "brier_score", "brier_score_ci95_low", "brier_score_ci95_high",
    "ece_15", "ece_15_ci95_low", "ece_15_ci95_high", "ece_10", "ece_20",
]
for _, r in d7.iterrows():
    key = (r["feature_configuration"], r["method"])
    if key not in exp7:
        FAIL.append(f"Table7 unexpected row {key}")
        continue
    for column, value in zip(columns7, exp7[key]):
        close(r[column], value, 4, f"Table7 {key} {column}")
if len(d7) != 6:
    FAIL.append(f"Table7 row count {len(d7)} != 6")


# Table 8: grouped permutation importance.
exp8 = {
    ("F4", "Magnitude group"): (0.1543, 0.0019, 0.1509, 0.1581),
    ("F4", "Color group"): (0.5301, 0.0027, 0.5245, 0.5340),
    ("F4", "Photometric block"): (0.5616, 0.0026, 0.5574, 0.5668),
    ("F7", "Magnitude group"): (0.0157, 0.0006, 0.0146, 0.0169),
    ("F7", "Color group"): (0.0947, 0.0012, 0.0923, 0.0970),
    ("F7", "Photometric block"): (0.1319, 0.0016, 0.1291, 0.1353),
    ("F7", "Spectroscopic redshift"): (0.5836, 0.0024, 0.5794, 0.5884),
}
d8 = pd.read_csv("grouped_permutation_output/grouped_permutation_summary.csv")
columns8 = [
    "mean_delta_macro_f1", "sd_delta_macro_f1", "q025_delta_macro_f1", "q975_delta_macro_f1"
]
for _, r in d8.iterrows():
    key = (r["feature_configuration"], r["group"])
    if key not in exp8:
        FAIL.append(f"Table8 unexpected row {key}")
        continue
    for column, value in zip(columns8, exp8[key]):
        close(r[column], value, 4, f"Table8 {key} {column}")
if len(d8) != 7:
    FAIL.append(f"Table8 row count {len(d8)} != 7")


# Figure 1–5 must be regenerated and non-empty. Numerical sources are checked above.
figures = [
    "figures/Figure_1_repeated_CV_macroF1.png",
    "figures/Figure_2_confusion_matrices.png",
    "figures/Figure_3_magnitude_robustness.png",
    "figures/Figure_4_reliability_curves.png",
    "figures/Figure_5_grouped_permutation_importance.png",
]
for figure in figures:
    path = Path(figure)
    if not path.exists() or path.stat().st_size < 1000:
        FAIL.append(f"Figure missing or empty: {figure}")


if FAIL:
    print("REPRODUCIBILITY AUDIT: FAIL")
    for failure in FAIL:
        print(" -", failure)
    sys.exit(1)

print("REPRODUCIBILITY AUDIT: PASS")
print("Tables 3–8 and Figure 1–5 numerical sources match the current manuscript at reported precision.")
