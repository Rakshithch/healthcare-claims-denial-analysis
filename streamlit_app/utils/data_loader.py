"""Cached data access for the Streamlit dashboard (fully offline — CSV only)."""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "synthetic"
ARTIFACTS = ROOT / "ml" / "artifacts"


@st.cache_data(show_spinner=False)
def load_claims() -> pd.DataFrame:
    """Load the synthetic claims with derived date fields."""
    df = pd.read_csv(DATA_DIR / "claims_50k.csv", parse_dates=["claim_date", "service_date"])
    df["claim_year"] = df["claim_date"].dt.year
    df["claim_month"] = df["claim_date"].dt.month
    df["claim_quarter"] = df["claim_date"].dt.quarter
    df["year_month"] = df["claim_date"].dt.to_period("M").astype(str)
    df["denial_reason_code"] = df["denial_reason_code"].fillna("")
    return df


@st.cache_data(show_spinner=False)
def load_providers() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "providers.csv")


@st.cache_data(show_spinner=False)
def load_payers() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "payers.csv")


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model + encoders; None if not yet trained."""
    import pickle

    model_path = ARTIFACTS / "xgb_denial_model.pkl"
    enc_path = ARTIFACTS / "encoders.pkl"
    if not (model_path.exists() and enc_path.exists()):
        return None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(enc_path, "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Shared sidebar filters; returns the filtered claims frame."""
    st.sidebar.markdown("### 🔍 Filters")
    dmin, dmax = df["claim_date"].min().date(), df["claim_date"].max().date()
    date_range = st.sidebar.date_input(
        "Claim date range", value=(dmin, dmax), min_value=dmin, max_value=dmax
    )
    payer_sel = st.sidebar.multiselect(
        "Payer type", sorted(df["payer_type"].unique()),
        default=sorted(df["payer_type"].unique()),
    )
    provider_sel = st.sidebar.multiselect(
        "Provider type", sorted(df["provider_type"].unique()),
        default=sorted(df["provider_type"].unique()),
    )

    out = df[df["payer_type"].isin(payer_sel) & df["provider_type"].isin(provider_sel)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        out = out[(out["claim_date"] >= start) & (out["claim_date"] <= end)]
    return out
