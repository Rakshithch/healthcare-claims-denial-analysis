# Architecture

## Full diagram

```
                        ┌─────────────────────────────────────────┐
                        │     scripts/generate_synthetic_data.py  │
                        │  50,000 claims · 500 providers · 20     │
                        │  payers · CARC denial codes · realistic │
                        │  payer/auth/coding denial economics     │
                        └───────────────┬─────────────────────────┘
                                        │ CSVs in data/synthetic/
              ┌─────────────────────────┼──────────────────────────┐
              │ (warehouse path)        │                          │ (offline path)
              ▼                         │                          ▼
┌──────────────────────────┐           │            ┌───────────────────────────┐
│  Snowflake               │           │            │  Local analytics          │
│  ┌────────────────────┐  │           │            │                           │
│  │ RAW                │◀─┼─ PUT + COPY INTO       │  notebooks/01_EDA.ipynb   │
│  │  CLAIMS/PROVIDERS/ │  │  (sql/snowflake_      │  notebooks/02_feature_    │
│  │  PAYERS            │  │   setup.sql)          │  engineering.ipynb        │
│  └─────────┬──────────┘  │                        └─────────────┬─────────────┘
│            │ dbt run     │                                      │
│  ┌─────────▼──────────┐  │                        ┌─────────────▼─────────────┐
│  │ STAGING            │  │                        │  ml/train_model.py        │
│  │  stg_claims        │  │                        │  XGBoost + Randomized     │
│  │  stg_providers     │  │                        │  SearchCV + SHAP          │
│  │  stg_payers        │  │                        │  → ml/artifacts/*.pkl,    │
│  └─────────┬──────────┘  │                        │    model_metrics.json     │
│            │             │                        └─────────────┬─────────────┘
│  ┌─────────▼──────────┐  │                                      │
│  │ MARTS              │  │                        ┌─────────────▼─────────────┐
│  │  FACT_CLAIMS       │  │                        │  streamlit_app/           │
│  │  DIM_PROVIDER      │  │                        │  app.py (KPIs + filters)  │
│  │  DIM_PAYER         │  │                        │  01 denial overview       │
│  │  DIM_DIAGNOSIS     │  │                        │  02 denial by provider    │
│  │  DIM_DATE          │  │                        │  03 denial trends         │
│  │  + V_* views       │  │                        │  04 claim risk scorer ←pkl│
│  └────────────────────┘  │                        └───────────────────────────┘
└──────────────────────────┘
```

## Data flow

1. **Generation** — `scripts/generate_synthetic_data.py` builds 50k claims with a
   logit-based denial model: payer base rates (medicare 15% → self_pay 40%),
   per-CPT/ICD/state policy effects, and hard correlations (missing prior auth ≈ 85%
   denial; ICD/CPT mismatch → CO-4/CO-11). Per-payer calibration keeps segment denial
   rates on target while leaving learnable signal for the ML model.
2. **Load** — `sql/snowflake_setup.sql` creates the database/schemas/tables, an
   internal stage, and `COPY INTO` statements; `scripts/load_to_snowflake.py` is the
   programmatic alternative using `write_pandas`.
3. **Transform** — dbt builds `stg_*` views (typing, cleaning, derived fields like
   `days_to_process` and `is_denied`) and mart tables (denial analysis by segment,
   provider, payer, monthly trends with window functions). Schema tests enforce keys,
   accepted values, and relationships.
4. **Analyze** — the EDA notebook walks 13 sections from KPI cards to a correlation
   heatmap; the feature-engineering notebook documents the encoding strategy and runs
   an explicit **leakage check** (post-adjudication fields excluded).
5. **Model** — `ml/train_model.py` label-encodes 7 categoricals + 5 numerics, runs a
   randomized search over the n_estimators/max_depth/learning_rate/subsample grid with
   3-fold CV, and persists model + encoders + metrics + SHAP plots. `ml/evaluate.py`
   adds decile lift and per-payer calibration checks.
6. **Serve** — the Streamlit app reads the CSVs directly (offline demo) and loads the
   pickled model in the risk scorer page, returning a probability gauge, risk band,
   top-3 SHAP drivers, and recommended actions.

## Components

| Component | Responsibility | Key design choice |
|---|---|---|
| Data generator | Realistic claims with learnable denial signal | Logit model + per-payer calibration, fixed seeds for reproducibility |
| Snowflake SQL | Warehouse DDL + star schema + views | RAW→STAGING→MARTS separation mirrors a production medallion layout |
| dbt | Tested, documented transformations | Staging as views, marts as tables; dbt_utils surrogate keys |
| ML pipeline | Denial probability scoring | Only pre-submission features (no leakage); LabelEncoder + unseen→-1 for inference safety |
| Streamlit | Stakeholder-facing analytics + live scoring | Zero warehouse dependency so anyone can run the demo |
