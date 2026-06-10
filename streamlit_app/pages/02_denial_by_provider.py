"""Page 2 — Provider performance: sortable table + volume vs denial scatter."""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.charts import LAYOUT
from utils.data_loader import load_claims, load_providers, sidebar_filters

st.set_page_config(page_title="Denial by Provider", page_icon="🩺", layout="wide")
st.title("🩺 Denial by Provider")

claims = load_claims()
df = sidebar_filters(claims)
if df.empty:
    st.warning("No claims match the current filters.")
    st.stop()

# Provider-level aggregation (mirrors mart_denial_by_provider)
providers = load_providers()
perf = (
    df.groupby(["provider_id", "provider_type"])
    .agg(total_claims=("claim_id", "count"),
         denied_claims=("denial_flag", "sum"),
         denial_rate=("denial_flag", "mean"),
         avg_claim_amount=("claim_amount", "mean"))
    .reset_index()
)
revenue = (df[df["denial_flag"] == 1]
           .groupby("provider_id")["claim_amount"].sum()
           .rename("revenue_at_risk"))
perf = perf.merge(revenue, on="provider_id", how="left").fillna({"revenue_at_risk": 0})
perf = perf.merge(providers[["provider_id", "provider_name", "state"]],
                  on="provider_id", how="left")

# Extra page-level filter on top of the shared sidebar ones
ptype = st.selectbox("Provider type", ["All"] + sorted(perf["provider_type"].unique()))
if ptype != "All":
    perf = perf[perf["provider_type"] == ptype]

st.subheader("Claim Volume vs Denial Rate")
st.caption("Bubble size = revenue at risk. Providers in the upper-right are the costliest problem.")
# Only providers with enough volume for a stable rate
plot_df = perf[perf["total_claims"] >= 20]
fig = px.scatter(
    plot_df,
    x="total_claims", y="denial_rate",
    size="revenue_at_risk", color="provider_type",
    hover_name="provider_name",
    hover_data={"denial_rate": ":.1%", "revenue_at_risk": ":$,.0f", "state": True},
    size_max=40,
)
fig.update_layout(xaxis_title="Total claims", yaxis_title="Denial rate",
                  yaxis_tickformat=".0%", height=520, **LAYOUT)
st.plotly_chart(fig, width="stretch")

st.subheader("Provider Performance Table")
st.caption("Click any column header to sort.")
table = perf.sort_values("revenue_at_risk", ascending=False).copy()
table["denial_rate"] = (table["denial_rate"] * 100).round(1)
table["avg_claim_amount"] = table["avg_claim_amount"].round(0)
table["revenue_at_risk"] = table["revenue_at_risk"].round(0)
st.dataframe(
    table[["provider_id", "provider_name", "provider_type", "state",
           "total_claims", "denied_claims", "denial_rate",
           "avg_claim_amount", "revenue_at_risk"]],
    width="stretch", height=420,
    column_config={
        "denial_rate": st.column_config.NumberColumn("Denial rate (%)", format="%.1f%%"),
        "avg_claim_amount": st.column_config.NumberColumn("Avg claim ($)", format="$%d"),
        "revenue_at_risk": st.column_config.NumberColumn("Revenue at risk ($)", format="$%d"),
    },
)
