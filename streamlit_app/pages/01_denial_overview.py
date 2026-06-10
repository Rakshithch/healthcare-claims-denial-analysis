"""Page 1 — Denial overview: rates by payer/provider and top reasons."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.charts import denial_rate_bar, denial_reasons_bar
from utils.data_loader import load_claims, sidebar_filters

st.set_page_config(page_title="Denial Overview", page_icon="📊", layout="wide")
st.title("📊 Denial Overview")

claims = load_claims()
df = sidebar_filters(claims)
if df.empty:
    st.warning("No claims match the current filters.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Claims in view", f"{len(df):,}")
c2.metric("Denial rate", f"{df['denial_flag'].mean():.1%}")
c3.metric("Revenue at risk",
          f"${df.loc[df['denial_flag'] == 1, 'claim_amount'].sum() / 1e6:,.1f}M")
st.divider()

left, right = st.columns(2)
with left:
    st.plotly_chart(denial_rate_bar(df, "payer_type", "Denial Rate by Payer Type"),
                    width="stretch")
with right:
    st.plotly_chart(denial_rate_bar(df, "provider_type", "Denial Rate by Provider Type"),
                    width="stretch")

st.plotly_chart(denial_reasons_bar(df), width="stretch")

with st.expander("Denial reason code reference"):
    denied = df[df["denial_flag"] == 1]
    ref = (denied.groupby("denial_reason_code")
           .agg(description=("denial_reason_description", "first"),
                denials=("claim_id", "count"),
                revenue_at_risk=("claim_amount", "sum"))
           .sort_values("denials", ascending=False))
    ref["revenue_at_risk"] = ref["revenue_at_risk"].map("${:,.0f}".format)
    st.dataframe(ref, width="stretch")
