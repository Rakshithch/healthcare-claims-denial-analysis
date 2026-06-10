"""Reusable plotly chart builders with consistent dashboard styling."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Risk color bands used across the app
def rate_color(rate: float) -> str:
    """Red >30%, orange 20-30%, green <20% — matches the EDA convention."""
    if rate > 0.30:
        return "#e74c3c"
    if rate > 0.20:
        return "#f39c12"
    return "#2ecc71"


LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=10, r=10, t=50, b=10),
)


def denial_rate_bar(df: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    """Horizontal bar of denial rate by a grouping column, risk-colored."""
    agg = (
        df.groupby(group_col)
        .agg(denial_rate=("denial_flag", "mean"), claims=("denial_flag", "size"))
        .reset_index()
        .sort_values("denial_rate")
    )
    fig = go.Figure(go.Bar(
        x=agg["denial_rate"] * 100,
        y=agg[group_col],
        orientation="h",
        marker_color=[rate_color(r) for r in agg["denial_rate"]],
        text=[f"{r:.1%}" for r in agg["denial_rate"]],
        textposition="outside",
        customdata=agg["claims"],
        hovertemplate="%{y}: %{x:.1f}%% (%{customdata:,} claims)<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="Denial rate (%)", yaxis_title="",
                      xaxis_range=[0, agg["denial_rate"].max() * 115], **LAYOUT)
    return fig


def denial_reasons_bar(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar of top denial reason codes with counts + share."""
    denied = df[df["denial_flag"] == 1]
    counts = denied["denial_reason_code"].value_counts().head(top_n)
    desc = denied.groupby("denial_reason_code")["denial_reason_description"].first()
    share = counts / counts.sum()
    fig = go.Figure(go.Bar(
        x=counts.values[::-1],
        y=counts.index[::-1],
        orientation="h",
        marker_color="#e74c3c",
        text=[f"{c:,} ({s:.0%})" for c, s in zip(counts.values[::-1], share.values[::-1])],
        textposition="outside",
        customdata=[desc.get(code, "")[:80] for code in counts.index[::-1]],
        hovertemplate="<b>%{y}</b><br>%{customdata}<br>%{x:,} denials<extra></extra>",
    ))
    fig.update_layout(title=f"Top {top_n} Denial Reason Codes",
                      xaxis_title="Denied claims", yaxis_title="",
                      xaxis_range=[0, counts.max() * 1.2], **LAYOUT)
    return fig


def monthly_trend(df: pd.DataFrame) -> go.Figure:
    """Monthly denial-rate line with a 3-month rolling average overlay."""
    monthly = (
        df.groupby("year_month")
        .agg(denial_rate=("denial_flag", "mean"), claims=("denial_flag", "size"))
        .reset_index()
        .sort_values("year_month")
    )
    monthly["rolling_3mo"] = monthly["denial_rate"].rolling(3, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["year_month"], y=monthly["denial_rate"] * 100,
        mode="lines+markers", name="Monthly denial rate",
        line=dict(color="#3498db", width=2),
        hovertemplate="%{x}: %{y:.1f}%%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=monthly["year_month"], y=monthly["rolling_3mo"] * 100,
        mode="lines", name="3-month rolling avg",
        line=dict(color="#f39c12", width=3, dash="dash"),
        hovertemplate="%{x}: %{y:.1f}%% (3-mo avg)<extra></extra>",
    ))
    layout = {**LAYOUT, "margin": dict(l=10, r=10, t=90, b=10)}
    fig.update_layout(title=dict(text="Monthly Denial Rate Trend", y=0.97, yanchor="top"),
                      xaxis_title="", yaxis_title="Denial rate (%)",
                      legend=dict(orientation="h", y=1.04, yanchor="bottom"), **layout)
    return fig


def gauge(probability: float) -> go.Figure:
    """Risk gauge for the claim scorer (0-100% denial probability)."""
    pct = probability * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 56}},
        title={"text": "Denial Probability", "font": {"size": 18}},
        gauge=dict(
            axis=dict(range=[0, 100], ticksuffix="%"),
            bar=dict(color="#ffffff", thickness=0.25),
            steps=[
                dict(range=[0, 25], color="#2ecc71"),
                dict(range=[25, 50], color="#f1c40f"),
                dict(range=[50, 75], color="#e67e22"),
                dict(range=[75, 100], color="#e74c3c"),
            ],
            threshold=dict(line=dict(color="white", width=3), value=pct),
        ),
    ))
    fig.update_layout(height=340, **LAYOUT)
    return fig
