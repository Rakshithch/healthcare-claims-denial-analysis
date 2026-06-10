"""Page 4 — Claim Denial Risk Scorer: live XGBoost predictions with SHAP drivers."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # streamlit_app/
sys.path.insert(0, str(ROOT / "ml"))                          # ml/ for predict helpers

from utils.charts import gauge
from utils.data_loader import load_claims, load_model
from predict import explain_claim, prepare_features, risk_level

st.set_page_config(page_title="Claim Risk Scorer", page_icon="🎯", layout="wide")
st.title("🎯 Claim Denial Risk Scorer")
st.markdown("Score a claim **before submission** using the trained XGBoost model "
            "(AUC 0.83). The model explains its top risk drivers via SHAP.")

model, encoders = load_model()
if model is None:
    st.error("Model artifacts not found. Run `python ml/train_model.py` first.")
    st.stop()

claims = load_claims()

# Human-friendly labels for dropdowns, built from the actual training data
icd_options = (claims[["icd10_primary", "icd10_description"]].drop_duplicates()
               .sort_values("icd10_primary"))
cpt_options = (claims[["cpt_code", "cpt_description"]].drop_duplicates()
               .sort_values("cpt_code"))
POS_LABELS = {
    "11": "11 — Office", "21": "21 — Inpatient Hospital",
    "22": "22 — Outpatient Hospital", "23": "23 — Emergency Room",
    "24": "24 — Ambulatory Surgical Center", "31": "31 — Skilled Nursing Facility",
}

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("risk_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        payer_type = st.selectbox("Payer Type", sorted(claims["payer_type"].unique()))
        provider_type = st.selectbox("Provider Type", sorted(claims["provider_type"].unique()))
        icd10 = st.selectbox(
            "Primary ICD-10 Code", icd_options["icd10_primary"],
            format_func=lambda c: f"{c} — " + icd_options.set_index("icd10_primary")
            .loc[c, "icd10_description"][:45],
        )
    with c2:
        cpt = st.selectbox(
            "CPT Code", cpt_options["cpt_code"],
            format_func=lambda c: f"{c} — " + cpt_options.set_index("cpt_code")
            .loc[c, "cpt_description"][:45],
        )
        auth_required = st.selectbox("Prior Auth Required", ["No", "Yes"])
        auth_obtained = st.selectbox("Prior Auth Obtained", ["No", "Yes"])
    with c3:
        age_bucket = st.selectbox("Patient Age Group", ["0-17", "18-34", "35-49", "50-64", "65+"])
        pos = st.selectbox("Place of Service", list(POS_LABELS), format_func=POS_LABELS.get)
        state = st.selectbox("State", sorted(claims["state"].unique()),
                             index=sorted(claims["state"].unique()).index("CA"))
    claim_amount = st.slider("Claim Amount ($)", 0, 50_000, 5_000, step=250)
    submitted = st.form_submit_button("🔮 Assess Denial Risk", width="stretch")

# ---------------------------------------------------------------------------
# Prediction + explanation
# ---------------------------------------------------------------------------
if submitted:
    claim = pd.DataFrame([{
        "payer_type": payer_type,
        "provider_type": provider_type,
        "icd10_primary": icd10,
        "cpt_code": cpt,
        "patient_age_bucket": age_bucket,
        "place_of_service": pos,
        "state": state,
        "prior_auth_required": auth_required == "Yes",
        "prior_auth_obtained": auth_obtained == "Yes",
        "claim_amount": float(max(claim_amount, 1)),  # model trained on positive amounts
        "claim_month": 6,    # neutral mid-year defaults for a pre-submission score
        "claim_quarter": 2,
    }])

    X = prepare_features(claim, encoders)
    proba = float(model.predict_proba(X)[0, 1])
    level = risk_level(proba)

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(gauge(proba), width="stretch")
        badge_color = {"LOW": "#2ecc71", "MEDIUM": "#f1c40f",
                       "HIGH": "#e67e22", "VERY HIGH": "#e74c3c"}[level]
        st.markdown(
            f"<div style='text-align:center'><span style='background:{badge_color};"
            f"color:#0f172a;padding:8px 28px;border-radius:999px;font-weight:800;"
            f"font-size:1.2rem'>RISK: {level}</span></div>",
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Top denial risk factors")
        FRIENDLY = {
            "payer_type": "Payer type", "provider_type": "Provider type",
            "icd10_chapter": "Diagnosis category", "cpt_code": "Procedure (CPT)",
            "patient_age_bucket": "Patient age group", "place_of_service": "Place of service",
            "state": "State", "prior_auth_required": "Prior auth required",
            "prior_auth_obtained": "Prior auth obtained", "claim_amount": "Claim amount",
            "claim_month": "Claim month", "claim_quarter": "Claim quarter",
        }
        drivers = explain_claim(claim, model, encoders, top_n=3)
        for feat, val in drivers:
            direction = "🔺 raises" if val > 0 else "🟢 lowers"
            st.markdown(f"- **{FRIENDLY.get(feat, feat)}** {direction} denial risk "
                        f"(SHAP {val:+.2f})")

        st.subheader("Recommended actions")
        recs = []
        if auth_required == "Yes" and auth_obtained == "No":
            recs.append("**Obtain prior authorization before submitting.** This is the "
                        "single biggest denial driver — auth-missing claims deny at ~85%.")
        if payer_type == "medicaid" and provider_type == "specialist":
            recs.append("Medicaid + specialist claims see elevated denials — verify plan "
                        "coverage and referral requirements first.")
        if claim_amount > 15_000:
            recs.append("High-dollar claim: attach clinical documentation and operative "
                        "notes proactively; expect heightened payer review.")
        if proba >= 0.5:
            recs.append("Route to the pre-submission review queue and double-check "
                        "ICD-10/CPT pairing for medical-necessity alignment.")
        if not recs:
            recs.append("Risk profile looks clean — submit normally and monitor the "
                        "835 remittance for adjustments.")
        for r in recs:
            st.markdown(f"- {r}")
