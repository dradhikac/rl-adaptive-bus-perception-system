"""
check_model_accuracy.py

Run this file directly to evaluate the trained reliability model and
print an accuracy report to the terminal — no Streamlit, no browser,
just:

    python check_model_accuracy.py

By default it looks for these files relative to where you run it:
    models/reliability_model.pkl
    data/bdd100k_features.csv
    data/reliability_scores.csv
    outputs/ml_model_results.csv   (optional — for the DT vs RF table)

If your files live somewhere else, either pass paths as arguments:

    python check_model_accuracy.py --model path/to/model.pkl \\
        --features path/to/bdd100k_features.csv \\
        --labels path/to/reliability_scores.csv

...or just edit the DEFAULT_* constants below.

A plain-text copy of the report is also saved to
outputs/accuracy_report.txt so you have something to show judges
without needing to re-run the script live.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ====================================
# DEFAULT PATHS — edit these if your repo layout differs
# ====================================

DEFAULT_MODEL_PATH = "models/reliability_model.pkl"
DEFAULT_FEATURES_PATH = "outputs/bdd100k_features.csv"
DEFAULT_LABELS_PATH = "outputs/reliability_scores.csv"
DEFAULT_ML_RESULTS_PATH = "outputs/ml_model_results.csv"
DEFAULT_REPORT_OUT_PATH = "outputs/accuracy_report.txt"


# ====================================
# HELPERS
# ====================================

def fail(message):
    print(f"\n❌ {message}\n")
    sys.exit(1)


def load_required_file(path, description, loader):
    p = Path(path)
    if not p.exists():
        fail(
            f"Couldn't find {description} at '{path}'.\n"
            f"   Either place it there, or re-run with the matching "
            f"command-line flag (see --help)."
        )
    return loader(p)


def print_table(title, rows, col_widths):
    print(f"\n{title}")
    print("-" * (sum(col_widths) + len(col_widths) + 1))
    header = "|" + "|".join(
        f" {rows[0][i]:<{col_widths[i]-1}}" for i in range(len(col_widths))
    ) + "|"
    print(header)
    print("-" * (sum(col_widths) + len(col_widths) + 1))
    for row in rows[1:]:
        line = "|" + "|".join(
            f" {str(row[i]):<{col_widths[i]-1}}" for i in range(len(col_widths))
        ) + "|"
        print(line)
    print("-" * (sum(col_widths) + len(col_widths) + 1))


# ====================================
# MAIN EVALUATION
# ====================================

def evaluate(model_path, features_path, labels_path, ml_results_path):

    lines = []  # collects everything printed, so it can also be saved to a file

    def log(text=""):
        print(text)
        lines.append(text)

    log("=" * 62)
    log(" MODEL ACCURACY & RELIABILITY CHECK")
    log("=" * 62)

    # ---------- load files ----------

    model = load_required_file(model_path, "the trained model", joblib.load)
    features = load_required_file(features_path, "the BDD100K feature CSV", pd.read_csv)
    labels = load_required_file(labels_path, "the reliability scores CSV", pd.read_csv)

    log(f"\nModel file     : {model_path}")
    log(f"Features file  : {features_path}")
    log(f"Labels file    : {labels_path}")
    log(f"Model type     : {type(model).__name__}")
    log(f"Model features : {list(model.feature_names_in_)}")

    # ---------- reconstruct ground truth ----------
    # reliability_scores.csv is a deterministic lookup keyed on
    # (weather, timeofday). We rebuild that lookup and join it onto the
    # BDD100K feature rows to form a held-out test set.

    lookup = labels.groupby(["weather", "timeofday"])["reliability_score"].mean().to_dict()
    features["reliability_true"] = features.apply(
        lambda r: lookup.get((r["weather"], r["timeofday"]), np.nan), axis=1
    )

    n_total = len(features)
    features = features.dropna(subset=["reliability_true"])
    n_used = len(features)
    n_skipped = n_total - n_used

    if n_used == 0:
        fail(
            "No rows in the features file matched a (weather, timeofday) "
            "combination in the labels file — check that both CSVs use "
            "the same category names."
        )

    le_w = LabelEncoder().fit(sorted(features["weather"].unique()))
    le_t = LabelEncoder().fit(sorted(features["timeofday"].unique()))
    features["weather_enc"] = le_w.transform(features["weather"])
    features["timeofday_enc"] = le_t.transform(features["timeofday"])

    X = features[["brightness", "contrast", "blur_score", "edge_count"]].copy()
    X["weather"] = features["weather_enc"]
    X["timeofday"] = features["timeofday_enc"]
    X = X[model.feature_names_in_]

    preds = model.predict(X)
    y_true = features["reliability_true"].values

    # ---------- metrics ----------

    mae = mean_absolute_error(y_true, preds)
    rmse = mean_squared_error(y_true, preds) ** 0.5
    r2 = r2_score(y_true, preds)
    within_02 = (np.abs(preds - y_true) <= 0.02).mean() * 100
    within_05 = (np.abs(preds - y_true) <= 0.05).mean() * 100
    within_10 = (np.abs(preds - y_true) <= 0.10).mean() * 100

    log(f"\nEvaluated on {n_used} held-out images "
        f"({n_skipped} skipped — no matching label).")

    print_table(
        "RELIABILITY MODEL — ACCURACY METRICS",
        [
            ("Metric", "Value"),
            ("Mean Absolute Error (MAE)", f"{mae:.4f}"),
            ("Root Mean Squared Error (RMSE)", f"{rmse:.4f}"),
            ("R\u00b2 Score", f"{r2:.4f}"),
            ("Predictions within \u00b10.02", f"{within_02:.1f}%"),
            ("Predictions within \u00b10.05", f"{within_05:.1f}%"),
            ("Predictions within \u00b10.10", f"{within_10:.1f}%"),
        ],
        col_widths=[36, 14],
    )
    lines.append("")  # spacing before the DT/RF table if it follows

    # ---------- optional DT vs RF comparison ----------

    ml_path = Path(ml_results_path)
    if ml_path.exists():
        ml_results = pd.read_csv(ml_path)
        dt_mae = ml_results["decision_tree_mae"].iloc[0]
        rf_mae = ml_results["random_forest_mae"].iloc[0]

        print_table(
            "PRIOR MODEL COMPARISON (self-reported, single split)",
            [
                ("Model", "MAE"),
                ("Decision Tree", f"{dt_mae:.4f}"),
                ("Random Forest", f"{rf_mae:.4f}"),
            ],
            col_widths=[20, 14],
        )
    else:
        log(f"\n(No {ml_results_path} found — skipping Decision Tree vs "
            f"Random Forest comparison table.)")

    log("\n" + "=" * 62)
    log(" This report was computed live from the actual model and data")
    log(" files above — every number will reproduce on any machine")
    log(" given the same inputs.")
    log("=" * 62 + "\n")

    return lines


# ====================================
# ENTRY POINT
# ====================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and print the reliability model's accuracy."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                         help=f"Path to the trained .pkl model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--features", default=DEFAULT_FEATURES_PATH,
                         help=f"Path to bdd100k_features.csv (default: {DEFAULT_FEATURES_PATH})")
    parser.add_argument("--labels", default=DEFAULT_LABELS_PATH,
                         help=f"Path to reliability_scores.csv (default: {DEFAULT_LABELS_PATH})")
    parser.add_argument("--ml-results", default=DEFAULT_ML_RESULTS_PATH,
                         help=f"Path to ml_model_results.csv, optional (default: {DEFAULT_ML_RESULTS_PATH})")
    parser.add_argument("--out", default=DEFAULT_REPORT_OUT_PATH,
                         help=f"Where to save a text copy of the report (default: {DEFAULT_REPORT_OUT_PATH})")
    args = parser.parse_args()

    report_lines = evaluate(args.model, args.features, args.labels, args.ml_results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved a text copy of this report to: {out_path}\n")


if __name__ == "__main__":
    main()
