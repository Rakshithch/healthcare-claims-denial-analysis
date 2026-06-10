"""
Inference helpers for the denial-risk model.

Used by the Streamlit risk scorer page and usable standalone:

  python ml/predict.py            # scores a sample claim
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# Must match ml/train_model.py exactly
CATEGORICAL_FEATURES = [
    "payer_type", "provider_type", "icd10_chapter", "cpt_code",
    "patient_age_bucket", "place_of_service", "state",
]
NUMERIC_FEATURES = [
    "prior_auth_required", "prior_auth_obtained",
    "claim_amount", "claim_month", "claim_quarter",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

RISK_LEVELS = [  # (threshold, label)
    (0.25, "LOW"),
    (0.50, "MEDIUM"),
    (0.75, "HIGH"),
    (1.01, "VERY HIGH"),
]


def load_artifacts():
    """Load the trained model and fitted encoders."""
    with open(ARTIFACTS / "xgb_denial_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


def prepare_features(claims: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Encode a dataframe of raw claim fields into model-ready features.

    Expects columns: payer_type, provider_type, icd10_primary, cpt_code,
    patient_age_bucket, place_of_service, state, prior_auth_required,
    prior_auth_obtained, claim_amount, claim_month, claim_quarter.
    """
    df = claims.copy()
    if "icd10_chapter" not in df:
        df["icd10_chapter"] = df["icd10_primary"].astype(str).str[:3]
    df["place_of_service"] = df["place_of_service"].astype(str)
    df["prior_auth_required"] = df["prior_auth_required"].astype(int)
    df["prior_auth_obtained"] = df["prior_auth_obtained"].astype(int)

    X = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        le = encoders[col]
        known = {v: i for i, v in enumerate(le.classes_)}
        # unseen categories -> -1 (model treats as its own branch)
        X[col] = X[col].astype(str).map(known).fillna(-1).astype(int)
    return X


def risk_level(probability: float) -> str:
    """Map a denial probability to a business-friendly risk band."""
    for threshold, label in RISK_LEVELS:
        if probability < threshold:
            return label
    return "VERY HIGH"


def score_claims(claims: pd.DataFrame, model=None, encoders=None) -> pd.DataFrame:
    """Score raw claims; returns input + denial_probability + risk_level."""
    if model is None or encoders is None:
        model, encoders = load_artifacts()
    X = prepare_features(claims, encoders)
    proba = model.predict_proba(X)[:, 1]
    out = claims.copy()
    out["denial_probability"] = np.round(proba, 4)
    out["risk_level"] = [risk_level(p) for p in proba]
    return out


def explain_claim(claim: pd.DataFrame, model=None, encoders=None, top_n: int = 3):
    """Per-claim SHAP explanation: top_n (feature, shap_value) risk drivers."""
    import shap

    if model is None or encoders is None:
        model, encoders = load_artifacts()
    X = prepare_features(claim, encoders)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)[0]
    contrib = sorted(zip(FEATURES, sv), key=lambda t: abs(t[1]), reverse=True)
    return contrib[:top_n]


if __name__ == "__main__":
    sample = pd.DataFrame([{
        "payer_type": "medicaid",
        "provider_type": "specialist",
        "icd10_primary": "M17.11",
        "cpt_code": "27447",
        "patient_age_bucket": "50-64",
        "place_of_service": "21",
        "state": "TX",
        "prior_auth_required": True,
        "prior_auth_obtained": False,
        "claim_amount": 32000.0,
        "claim_month": 7,
        "claim_quarter": 3,
    }])
    scored = score_claims(sample)
    print(f"Denial probability: {scored['denial_probability'].iloc[0]:.1%}")
    print(f"Risk level:         {scored['risk_level'].iloc[0]}")
    print("Top drivers:")
    for feat, val in explain_claim(sample):
        print(f"  {feat:<22} SHAP={val:+.3f}")
