"""
Evaluate the saved denial model against the full synthetic dataset.

Prints classification metrics, decile lift, and segment-level calibration —
the checks an RCM team would run before trusting the scorer in workflow.

Run: python ml/evaluate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from predict import load_artifacts, prepare_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic" / "claims_50k.csv"


def main() -> None:
    model, encoders = load_artifacts()
    df = pd.read_csv(DATA_PATH)
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df["claim_month"] = df["claim_date"].dt.month
    df["claim_quarter"] = df["claim_date"].dt.quarter

    X = prepare_features(df, encoders)
    y = df["denial_flag"].astype(int)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("=" * 60)
    print("FULL-DATASET EVALUATION")
    print("=" * 60)
    print(f"AUC-ROC: {roc_auc_score(y, proba):.4f}")
    print("\nClassification report:")
    print(classification_report(y, pred, target_names=["approved/pending", "denied"]))
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y, pred))

    # Decile lift: do high scores capture most denials?
    deciles = pd.qcut(proba, 10, labels=False, duplicates="drop")
    lift = pd.DataFrame({"decile": deciles, "denied": y}).groupby("decile")["denied"].agg(["mean", "sum", "count"])
    base = y.mean()
    print("\nDecile analysis (decile 9 = highest predicted risk):")
    print(f"{'decile':>6} {'denial_rate':>12} {'lift':>6} {'denials':>9}")
    for d, row in lift.iterrows():
        print(f"{d:>6} {row['mean']:>11.1%} {row['mean'] / base:>5.1f}x {int(row['sum']):>9,}")

    # Calibration by payer type — flag any segment where the model drifts
    df["proba"] = proba
    print("\nCalibration by payer type (predicted vs actual):")
    cal = df.groupby("payer_type").agg(predicted=("proba", "mean"), actual=("denial_flag", "mean"))
    for ptype, row in cal.iterrows():
        print(f"  {ptype:<12} predicted {row['predicted']:.1%}  actual {row['actual']:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
