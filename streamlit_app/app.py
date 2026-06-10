"""Healthcare Claims Denial Analytics — Streamlit entry point (home page)."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make utils importable
from utils.data_loader import load_claims, sidebar_filters

st.set_page_config(
    page_title="Healthcare Claims Denial Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — dark professional theme touches
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
    }
    [data-testid="stMetricValue"] { font-size: 2rem; color: #38bdf8; }
    [data-testid="stMetricLabel"] { color: #94a3b8; }
    .tech-badge {
        display: inline-block; padding: 4px 12px; margin: 3px;
        border-radius: 999px; font-size: 0.8rem; font-weight: 600;
        background: #1e293b; border: 1px solid #38bdf8; color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🏥 Denial Analytics")
st.sidebar.markdown("Synthetic 837/835-style claims · 2022–2024")
st.sidebar.divider()

claims = load_claims()
filtered = sidebar_filters(claims)

st.sidebar.divider()
st.sidebar.caption("Built with Snowflake · dbt · XGBoost · Streamlit")

# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------
st.title("🏥 Healthcare Claims Denial Analytics")
st.markdown(
    "End-to-end denial-rate analysis and ML risk scoring over **50,000 synthetic "
    "claims**. Use the sidebar to filter, and the pages on the left to explore "
    "denial drivers, provider performance, trends, and the live risk scorer."
)

# KPI cards
denied = filtered[filtered["denial_flag"] == 1]
n_days = max((filtered["claim_date"].max() - filtered["claim_date"].min()).days, 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Claims Analyzed", f"{len(filtered):,}")
c2.metric("Overall Denial Rate", f"{filtered['denial_flag'].mean():.1%}" if len(filtered) else "—")
c3.metric("Revenue at Risk", f"${denied['claim_amount'].sum() / 1e6:,.1f}M")
c4.metric("Avg Claims / Day", f"{len(filtered) / n_days:,.0f}")

st.divider()

left, right = st.columns([3, 2])
with left:
    st.subheader("About this project")
    st.markdown("""
- **Problem** — U.S. providers lose an estimated **$262B annually** to claim denials.
  Most denials are preventable: missing prior authorization, coding mismatches,
  and payer-specific policy quirks.
- **Pipeline** — synthetic claims generator → Snowflake (RAW → STAGING → MARTS star
  schema) → dbt models with tests → EDA → XGBoost denial classifier with SHAP
  explainability → this dashboard.
- **What to try** — open **Claim Risk Scorer** and submit a Medicaid specialist claim
  with prior auth required but not obtained. Watch the gauge.
""")
with right:
    st.subheader("Tech stack")
    st.markdown(
        '<span class="tech-badge">Python</span>'
        '<span class="tech-badge">SQL</span>'
        '<span class="tech-badge">Snowflake</span>'
        '<span class="tech-badge">dbt</span>'
        '<span class="tech-badge">XGBoost</span>'
        '<span class="tech-badge">SHAP</span>'
        '<span class="tech-badge">scikit-learn</span>'
        '<span class="tech-badge">Plotly</span>'
        '<span class="tech-badge">Streamlit</span>'
        '<span class="tech-badge">pandas</span>',
        unsafe_allow_html=True,
    )
    st.subheader("Data note")
    st.info("All data is **synthetic** — generated with realistic payer/denial "
            "economics. No PHI/PII anywhere in this project.")
