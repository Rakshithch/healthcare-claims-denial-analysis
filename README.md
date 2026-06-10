# 🏥 Healthcare Claims Denial Rate Analysis & Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Snowflake](https://img.shields.io/badge/Snowflake-Data%20Warehouse-29B5E8?logo=snowflake&logoColor=white)
![dbt](https://img.shields.io/badge/dbt-Transformations-FF694B?logo=dbt&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.83-EB5424)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/data-100%25%20synthetic-2ecc71)

> End-to-end revenue-cycle analytics: synthetic 837/835-style claims → Snowflake/dbt
> star schema → EDA → XGBoost denial-risk model with SHAP → interactive Streamlit dashboard.

## 💸 The Problem

U.S. healthcare providers lose an estimated **$262 billion annually** to denied claims.
The brutal part: **most denials are preventable** — missing prior authorizations,
ICD-10/CPT coding mismatches, and payer-specific policy quirks that could be caught
*before* submission. This project builds the analytics and ML tooling a revenue-cycle
team would use to find, quantify, and prevent those losses.

## 🏗️ Architecture

```
┌──────────────────┐     ┌──────────────────────────────────────────┐
│  Data Generation │     │              Snowflake                   │
│  ──────────────  │     │  ┌───────┐   ┌─────────┐   ┌─────────┐   │
│  50k synthetic   │────▶│  │  RAW  │──▶│ STAGING │──▶│  MARTS  │   │
│  claims (837/835 │COPY │  └───────┘   └─────────┘   └─────────┘   │
│  patterns, CARC  │INTO │      ▲            ▲      star schema +   │
│  codes)          │     │      └── dbt run ─┘      analytic views  │
└──────────────────┘     └──────────────────────────────────────────┘
         │                                  │
         │ CSV (offline demo path)          │ (warehouse path)
         ▼                                  ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   EDA Notebooks  │     │  ML Pipeline     │     │  Streamlit App   │
│  ──────────────  │     │  ──────────────  │     │  ──────────────  │
│  13-section EDA, │     │  XGBoost + CV    │────▶│  4 pages + live  │
│  feature eng.    │     │  SHAP, AUC 0.83  │ pkl │  risk scorer     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Data generation | Python, NumPy, pandas (realistic denial economics baked in) |
| Warehouse | ❄️ Snowflake — RAW/STAGING/MARTS, star schema, analytic views |
| Transformation | dbt — staging + marts models, schema tests (`not_null`, `unique`, `accepted_values`, `relationships`) |
| Analysis | Jupyter, seaborn/matplotlib |
| ML | XGBoost, scikit-learn (RandomizedSearchCV), SHAP explainability |
| Dashboard | Streamlit + Plotly (fully interactive, dark theme) |

## 🔑 Key Findings

1. **Prior authorization is the #1 controllable denial driver** — claims requiring auth
   that didn't obtain it deny at **~84%** vs **~27%** when obtained (3x multiplier).
2. **Payer mix sets the baseline**: self-pay ~40% and Medicaid ~35% denial rates vs
   Medicare ~15% — a 2.3–2.7x spread that should drive payer-weighted forecasting.
3. **CO-4 + CO-11 coding mismatches account for over half of all denials** — a claim-scrubber
   rules gap, not a clinical problem.
4. **~$78M in billed charges denied over 36 months** (~$26M/year) across 50k claims.
5. The XGBoost scorer reaches **AUC 0.83**, with prior-auth status, claim amount, and
   payer type as the top SHAP drivers — matching domain intuition.

## 🚀 Run It Locally

```bash
# 1. Clone and set up
git clone <repo-url> && cd healthcare-claims-denial-analysis
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only — XGBoost needs OpenMP:
brew install libomp

# 2. Generate the synthetic dataset (50k claims)
python scripts/generate_synthetic_data.py

# 3. Train the denial-risk model (writes ml/artifacts/)
python ml/train_model.py

# 4. Launch the dashboard — fully offline, no Snowflake needed
streamlit run streamlit_app/app.py
```

Optional warehouse leg:

```bash
# Snowflake: run sql/snowflake_setup.sql, then
export SNOWFLAKE_ACCOUNT=... SNOWFLAKE_USER=... SNOWFLAKE_PASSWORD=...
python scripts/load_to_snowflake.py
cd dbt_project && dbt deps && dbt run && dbt test
```

## 📁 Project Structure

```
healthcare-claims-denial-analysis/
├── scripts/                 # synthetic data generator + Snowflake loader
├── sql/                     # Snowflake DDL, stage, COPY INTO, analytic views
├── dbt_project/             # staging + marts models with schema tests
├── notebooks/               # 01_EDA (13 sections), 02_feature_engineering
├── ml/                      # train / predict / evaluate + artifacts
├── streamlit_app/           # 4-page interactive dashboard
│   └── pages/               #   overview · providers · trends · risk scorer
└── docs/                    # architecture deep-dive + LinkedIn post
```

## 📸 Screenshots

> _Placeholder — add screenshots of the dashboard home, trends page, and risk scorer gauge._

## 🔗 Links

- 📝 LinkedIn post: [docs/linkedin_post.md](docs/linkedin_post.md) _(post link placeholder)_
- 📐 Architecture deep-dive: [docs/architecture.md](docs/architecture.md)

---
*All data is synthetic — generated with realistic payer/denial economics. No PHI/PII.*
