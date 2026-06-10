# Healthcare Claims Denial Analysis — Project Guide

## What this project is
A portfolio-grade healthcare claims denial analytics platform: synthetic 837/835-style
claims data → Snowflake/dbt warehouse models → EDA → XGBoost denial-risk model →
Streamlit dashboard.

## Key conventions
- Python env: `.venv/` (Python 3.11). Activate with `source .venv/bin/activate`.
- All data is **synthetic** — no PHI/PII anywhere. Keep it that way.
- The Streamlit app must run fully offline from `data/synthetic/claims_50k.csv`
  (no Snowflake connection required for the demo).
- Charts: plotly in Streamlit, seaborn/matplotlib in notebooks.
- Denial economics baked into the generator (do not break these when editing):
  overall denial rate ≈ 27%; medicare 15% / medicaid 35% / commercial 25% / self_pay 40%;
  prior-auth-required-but-not-obtained claims deny at ≈ 85%.

## How to run
```bash
source .venv/bin/activate
python scripts/generate_synthetic_data.py   # regenerate data
python ml/train_model.py                    # retrain model (writes ml/artifacts/)
streamlit run streamlit_app/app.py          # launch dashboard
```

## Layout
- `scripts/` — data generation + Snowflake loader
- `sql/` — raw Snowflake DDL/COPY/view setup
- `dbt_project/` — staging + marts models (Snowflake target)
- `notebooks/` — EDA and feature engineering
- `ml/` — train/predict/evaluate + saved artifacts
- `streamlit_app/` — multi-page dashboard (pages auto-discovered from `pages/`)
- `docs/` — architecture + LinkedIn post

## Gotchas
- `ml/artifacts/*.pkl` are gitignored; run `ml/train_model.py` after cloning.
- The risk scorer page (`streamlit_app/pages/04_claim_risk_scorer.py`) loads the
  pickled model + encoders — feature names there must match `ml/train_model.py`.
- dbt `profiles.yml` is gitignored; copy `profiles.yml.example` and fill credentials.
