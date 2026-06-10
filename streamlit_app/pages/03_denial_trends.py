"""Page 3 — Trends: monthly denial rate, YoY comparison, prior-auth impact."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.charts import LAYOUT, monthly_trend
from utils.data_loader import load_claims, sidebar_filters

st.set_page_config(page_title="Denial Trends", page_icon="📈", layout="wide")
st.title("📈 Denial Trends (2022–2024)")

claims = load_claims()
df = sidebar_filters(claims)
if df.empty:
    st.warning("No claims match the current filters.")
    st.stop()

# Monthly trend with 3-month rolling average
st.plotly_chart(monthly_trend(df), width="stretch")

left, right = st.columns(2)

with left:
    st.subheader("Year-over-Year Denial Rate")
    yearly = (df.groupby("claim_year")
              .agg(denial_rate=("denial_flag", "mean"), claims=("claim_id", "count"))
              .reset_index())
    fig = go.Figure(go.Bar(
        x=yearly["claim_year"].astype(str),
        y=yearly["denial_rate"] * 100,
        marker_color=["#3498db", "#9b59b6", "#e67e22"][: len(yearly)],
        text=[f"{r:.1%}" for r in yearly["denial_rate"]],
        textposition="outside",
        customdata=yearly["claims"],
        hovertemplate="%{x}: %{y:.1f}%% (%{customdata:,} claims)<extra></extra>",
    ))
    fig.update_layout(xaxis_title="", yaxis_title="Denial rate (%)",
                      yaxis_range=[0, yearly["denial_rate"].max() * 130], **LAYOUT)
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Prior Auth Impact Over Time")
    auth_req = df[df["prior_auth_required"]].copy()
    auth_req["auth_status"] = auth_req["prior_auth_obtained"].map(
        {True: "Auth obtained", False: "Auth NOT obtained"})
    trend = (auth_req.groupby(["year_month", "auth_status"])["denial_flag"]
             .mean().reset_index())
    fig = go.Figure()
    for status, color in [("Auth obtained", "#2ecc71"), ("Auth NOT obtained", "#e74c3c")]:
        sub = trend[trend["auth_status"] == status]
        fig.add_trace(go.Scatter(
            x=sub["year_month"], y=sub["denial_flag"] * 100,
            mode="lines", name=status, line=dict(color=color, width=2),
            hovertemplate="%{x}: %{y:.1f}%%<extra></extra>",
        ))
    fig.update_layout(xaxis_title="", yaxis_title="Denial rate (%)",
                      legend=dict(orientation="h", y=1.15), **LAYOUT)
    st.plotly_chart(fig, width="stretch")

st.info("The gap between the green and red lines is the **prior authorization story**: "
        "claims requiring auth that never obtained it deny at ~85% — month after month. "
        "An auth-verification workstep before submission is the single highest-ROI fix.")
