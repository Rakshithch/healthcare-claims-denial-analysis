"""
Train an XGBoost claim-denial classifier.

Pipeline:
  1. Load data/synthetic/claims_50k.csv
  2. Feature engineering (label-encode categoricals, derive icd10_chapter,
     claim_month/quarter)
  3. Stratified 80/20 train/test split
  4. Randomized hyperparameter search with 3-fold CV over the documented grid
  5. Evaluate (AUC-ROC, precision, recall, F1, confusion matrix)
  6. SHAP global importance (bar + beeswarm plots, top-5 JSON)
  7. Persist model, encoders, and metrics to ml/artifacts/

Run: python ml/train_model.py
"""

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — we only save figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic" / "claims_50k.csv"
ARTIFACTS = ROOT / "ml" / "artifacts"

# Categorical columns that get a LabelEncoder (saved for inference)
CATEGORICAL_FEATURES = [
    "payer_type",
    "provider_type",
    "icd10_chapter",
    "cpt_code",
    "patient_age_bucket",
    "place_of_service",
    "state",
]
NUMERIC_FEATURES = [
    "prior_auth_required",
    "prior_auth_obtained",
    "claim_amount",
    "claim_month",
    "claim_quarter",
]
TARGET = "denial_flag"

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 0.9, 1.0],
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from raw claim columns (no encoding yet)."""
    out = df.copy()
    # ICD-10 chapter proxy: first 3 chars (category level, e.g. E11, I10)
    out["icd10_chapter"] = out["icd10_primary"].astype(str).str[:3]
    out["claim_date"] = pd.to_datetime(out["claim_date"])
    out["claim_month"] = out["claim_date"].dt.month
    out["claim_quarter"] = out["claim_date"].dt.quarter
    out["prior_auth_required"] = out["prior_auth_required"].astype(int)
    out["prior_auth_obtained"] = out["prior_auth_obtained"].astype(int)
    out["place_of_service"] = out["place_of_service"].astype(str)
    # defensive: drop rows missing the target or amount
    out = out.dropna(subset=[TARGET, "claim_amount"])
    return out


def encode(df: pd.DataFrame, encoders: dict | None = None):
    """Label-encode categoricals. Fits new encoders when none are passed."""
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    fitted = encoders or {}
    for col in CATEGORICAL_FEATURES:
        if encoders is None:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            fitted[col] = le
        else:
            le = fitted[col]
            # unseen categories map to -1 rather than crashing at inference
            known = {v: i for i, v in enumerate(le.classes_)}
            X[col] = X[col].astype(str).map(known).fillna(-1).astype(int)
    return X, fitted


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    df = build_features(pd.read_csv(DATA_PATH))
    X, encoders = encode(df)
    y = df[TARGET].astype(int)
    print(f"  {len(X):,} claims | positive rate {y.mean():.1%} | {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print("Hyperparameter search (RandomizedSearchCV, 20 candidates x 3 folds) ...")
    search = RandomizedSearchCV(
        xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
        param_distributions=PARAM_GRID,
        n_iter=20,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"  best params: {search.best_params_}")
    print(f"  best CV AUC: {search.best_score_:.4f}")

    # ----- evaluation ---------------------------------------------------
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "auc_roc": round(float(roc_auc_score(y_test, proba)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "best_params": search.best_params_,
        "cv_auc": round(float(search.best_score_), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": list(X.columns),
    }

    # ----- SHAP explainability --------------------------------------------
    print("Computing SHAP values ...")
    explainer = shap.TreeExplainer(model)
    sample = X_test.sample(n=min(3000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    # global importance = mean |SHAP|
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
    importance = importance.sort_values(ascending=False)
    metrics["top_5_features"] = {
        k: round(float(v), 4) for k, v in importance.head(5).items()
    }

    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.title("Global Feature Importance (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "shap_importance_bar.png", dpi=150)
    plt.close("all")

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.title("SHAP Summary (beeswarm)")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "shap_summary_beeswarm.png", dpi=150)
    plt.close("all")

    # ----- persist artifacts -----------------------------------------------
    with open(ARTIFACTS / "xgb_denial_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(ARTIFACTS / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(ARTIFACTS / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(ARTIFACTS / "feature_importance.json", "w") as f:
        json.dump({k: round(float(v), 4) for k, v in importance.items()}, f, indent=2)

    # ----- report -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL TRAINING REPORT")
    print("=" * 60)
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}  (target > 0.82)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    tn, fp, fn, tp = np.array(metrics["confusion_matrix"]).ravel()
    print(f"Confusion: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
    print("\nTop 5 features (mean |SHAP|):")
    for feat, val in metrics["top_5_features"].items():
        print(f"  {feat:<22} {val:.4f}")
    print(f"\nArtifacts written to {ARTIFACTS}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
