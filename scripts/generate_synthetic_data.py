"""
Synthetic healthcare claims generator.

Produces 50,000 claims with realistic denial economics modeled on industry
patterns (837/835 EDI fields, CARC denial codes, payer-mix denial rates):

  - Overall denial rate target: ~27%
  - Payer-level base denial rates: medicare 15%, medicaid 35%,
    commercial 25%, self_pay 40%
  - Prior auth required but NOT obtained -> ~85% denial rate
  - Medicaid + specialist combination -> elevated denial rate
  - ICD-10/CPT clinical mismatches -> CO-4 denials
  - High-cost procedures -> more likely to require prior auth

Outputs:
  data/synthetic/claims_50k.csv
  data/synthetic/providers.csv  (500 providers)
  data/synthetic/payers.csv     (20 payers)
"""

import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_CLAIMS = 50_000
N_PROVIDERS = 500
N_PAYERS = 20
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"

# ---------------------------------------------------------------------------
# Reference data: real ICD-10 / CPT codes and CARC denial reason codes
# ---------------------------------------------------------------------------

ICD10_CODES = {
    # code: (description, relative volume weight)
    "E11.9":   ("Type 2 diabetes mellitus without complications", 9),
    "I10":     ("Essential (primary) hypertension", 10),
    "J96.0":   ("Acute respiratory failure", 3),
    "K92.1":   ("Melena", 2),
    "N18.6":   ("End stage renal disease", 3),
    "F32.9":   ("Major depressive disorder, single episode, unspecified", 6),
    "M54.5":   ("Low back pain", 8),
    "Z51.11":  ("Encounter for antineoplastic chemotherapy", 4),
    "C50.919": ("Malignant neoplasm of unspecified site of unspecified female breast", 3),
    "I50.9":   ("Heart failure, unspecified", 5),
    "J18.9":   ("Pneumonia, unspecified organism", 5),
    "K57.30":  ("Diverticulosis of large intestine without perforation or abscess", 2),
    "E78.5":   ("Hyperlipidemia, unspecified", 7),
    "G20":     ("Parkinson's disease", 2),
    "M17.11":  ("Unilateral primary osteoarthritis, right knee", 4),
    "I63.9":   ("Cerebral infarction, unspecified", 3),
    "N39.0":   ("Urinary tract infection, site not specified", 6),
    "R07.9":   ("Chest pain, unspecified", 6),
    "Z79.899": ("Other long term (current) drug therapy", 4),
    "J44.1":   ("Chronic obstructive pulmonary disease with acute exacerbation", 4),
}

CPT_CODES = {
    # code: (description, low $, high $, weight)
    "99213": ("Office visit, established patient, low complexity", 90, 180, 14),
    "99214": ("Office visit, established patient, moderate complexity", 130, 260, 12),
    "99232": ("Subsequent hospital care, moderate complexity", 150, 320, 6),
    "99291": ("Critical care, first 30-74 minutes", 450, 1100, 3),
    "27447": ("Total knee arthroplasty", 18000, 42000, 2),
    "43239": ("Upper GI endoscopy with biopsy", 1200, 3800, 4),
    "70553": ("MRI brain without and with contrast", 1500, 4200, 4),
    "93306": ("Echocardiogram, complete with Doppler", 700, 2200, 5),
    "36415": ("Routine venipuncture", 8, 30, 10),
    "80053": ("Comprehensive metabolic panel", 25, 90, 10),
    "71046": ("Chest X-ray, 2 views", 80, 280, 7),
    "43235": ("Upper GI endoscopy, diagnostic", 900, 2900, 3),
    "64483": ("Transforaminal epidural injection, lumbar", 900, 2600, 3),
    "27130": ("Total hip arthroplasty", 20000, 48000, 2),
    "90837": ("Psychotherapy, 60 minutes", 140, 320, 5),
    "99285": ("Emergency department visit, high severity", 700, 2400, 5),
    "92928": ("Percutaneous coronary stent placement", 9000, 28000, 2),
    "33533": ("Coronary artery bypass, single arterial graft", 28000, 65000, 1),
    "43264": ("ERCP with stone removal", 3500, 9500, 1),
    "61510": ("Craniotomy for tumor excision", 32000, 75000, 1),
}

# CPT codes expensive/elective enough that payers commonly require prior auth
HIGH_COST_CPT = {"27447", "27130", "92928", "33533", "61510", "70553", "64483", "43264", "99291"}

# Clinically plausible ICD-10 pairings per CPT. A claim whose diagnosis is NOT
# in this set is a coding mismatch -> prime candidate for a CO-4 denial.
CPT_VALID_ICD10 = {
    "27447": {"M17.11", "M54.5"},
    "27130": {"M17.11", "M54.5"},
    "92928": {"I50.9", "R07.9", "I10", "I63.9"},
    "33533": {"I50.9", "R07.9", "I10"},
    "70553": {"G20", "I63.9", "F32.9", "C50.919"},
    "93306": {"I50.9", "I10", "R07.9"},
    "43239": {"K92.1", "K57.30"},
    "43235": {"K92.1", "K57.30"},
    "43264": {"K92.1", "K57.30"},
    "64483": {"M54.5", "M17.11"},
    "90837": {"F32.9"},
    "61510": {"C50.919", "G20", "I63.9"},
}

DENIAL_REASONS = {
    # CARC code: (description, base weight)
    "CO-4":  ("The procedure code is inconsistent with the modifier used or a required modifier is missing", 14),
    "CO-11": ("The diagnosis is inconsistent with the procedure", 12),
    "CO-22": ("This care may be covered by another payer per coordination of benefits", 10),
    "CO-29": ("The time limit for filing has expired", 9),
    "CO-97": ("The benefit for this service is included in the payment/allowance for another service", 13),
    "PR-1":  ("Deductible amount", 16),
    "PR-2":  ("Coinsurance amount", 11),
    "OA-23": ("The impact of prior payer(s) adjudication including payments and/or adjustments", 8),
}

# CO-197-style "no prior auth" denials are mapped to CO-4/CO-11 buckets here to
# keep the requested 8-code universe; auth failures skew toward CO-11.

PAYER_TYPES = ["medicare", "medicaid", "commercial", "self_pay"]
PAYER_TYPE_P = [0.30, 0.25, 0.35, 0.10]
PAYER_BASE_DENIAL = {"medicare": 0.15, "medicaid": 0.35, "commercial": 0.25, "self_pay": 0.40}

PROVIDER_TYPES = ["hospital", "physician", "specialist", "lab", "imaging"]
PROVIDER_TYPE_P = [0.20, 0.35, 0.25, 0.10, 0.10]

AGE_BUCKETS = ["0-17", "18-34", "35-49", "50-64", "65+"]
PLACE_OF_SERVICE = ["11", "21", "22", "23", "24", "31"]  # office, inpatient, outpatient, ER, ASC, SNF

# All 50 states, weighted roughly by population
STATES = {
    "CA": 39, "TX": 30, "FL": 22, "NY": 19, "PA": 13, "IL": 12, "OH": 12, "GA": 11,
    "NC": 11, "MI": 10, "NJ": 9, "VA": 9, "WA": 8, "AZ": 7, "TN": 7, "MA": 7,
    "IN": 7, "MO": 6, "MD": 6, "WI": 6, "CO": 6, "MN": 6, "SC": 5, "AL": 5,
    "LA": 5, "KY": 4, "OR": 4, "OK": 4, "CT": 4, "UT": 3, "IA": 3, "NV": 3,
    "AR": 3, "MS": 3, "KS": 3, "NM": 2, "NE": 2, "ID": 2, "WV": 2, "HI": 1,
    "NH": 1, "ME": 1, "MT": 1, "RI": 1, "DE": 1, "SD": 1, "ND": 1, "AK": 1,
    "VT": 1, "WY": 1,
}

PAYER_NAMES = {
    "medicare": ["Medicare Part B", "Medicare Advantage - UHC", "Medicare Advantage - Humana",
                 "Medicare Advantage - Aetna", "Medicare Advantage - Cigna", "Medicare Part A"],
    "medicaid": ["State Medicaid", "Medicaid MCO - Centene", "Medicaid MCO - Molina",
                 "Medicaid MCO - Anthem", "Medicaid MCO - UHC Community"],
    "commercial": ["UnitedHealthcare", "Anthem BCBS", "Aetna", "Cigna",
                   "Humana Commercial", "Kaiser Permanente", "BCBS Federal"],
    "self_pay": ["Self Pay", "Self Pay - Payment Plan"],
}


def build_payers() -> pd.DataFrame:
    """20 payers spread across the four payer types."""
    rows = []
    i = 1
    for ptype, names in PAYER_NAMES.items():
        for name in names:
            rows.append({"payer_id": f"PAY-{i:02d}", "payer_name": name, "payer_type": ptype})
            i += 1
    return pd.DataFrame(rows)  # 6+5+7+2 = 20


def build_providers() -> pd.DataFrame:
    """500 providers with type and state."""
    types = RNG.choice(PROVIDER_TYPES, size=N_PROVIDERS, p=PROVIDER_TYPE_P)
    states = RNG.choice(list(STATES), size=N_PROVIDERS,
                        p=np.array(list(STATES.values())) / sum(STATES.values()))
    return pd.DataFrame({
        "provider_id": [f"PRV-{i:04d}" for i in range(1, N_PROVIDERS + 1)],
        "provider_name": [f"Provider {i:04d}" for i in range(1, N_PROVIDERS + 1)],
        "provider_type": types,
        "state": states,
    })


def build_claims(providers: pd.DataFrame, payers: pd.DataFrame) -> pd.DataFrame:
    n = N_CLAIMS

    # --- entity assignment -------------------------------------------------
    prov_idx = RNG.integers(0, len(providers), n)
    provider_id = providers["provider_id"].to_numpy()[prov_idx]
    provider_type = providers["provider_type"].to_numpy()[prov_idx]
    state = providers["state"].to_numpy()[prov_idx]  # claim state follows provider

    payer_type = RNG.choice(PAYER_TYPES, size=n, p=PAYER_TYPE_P)
    # pick a payer_id consistent with the payer type
    payer_id = np.empty(n, dtype=object)
    for ptype in PAYER_TYPES:
        ids = payers.loc[payers["payer_type"] == ptype, "payer_id"].to_numpy()
        mask = payer_type == ptype
        payer_id[mask] = RNG.choice(ids, size=mask.sum())

    patient_id = np.array([f"P-{i:06d}" for i in RNG.integers(1, 20_001, n)])

    # --- dates --------------------------------------------------------------
    start = date(2022, 1, 1)
    claim_offsets = RNG.integers(0, (date(2024, 12, 31) - start).days + 1, n)
    claim_date = np.array([start + timedelta(days=int(d)) for d in claim_offsets])
    service_lag = RNG.integers(1, 8, n)  # service 1-7 days before claim submission
    service_date = np.array([cd - timedelta(days=int(l)) for cd, l in zip(claim_date, service_lag)])

    # --- clinical codes -----------------------------------------------------
    icd_codes = list(ICD10_CODES)
    icd_w = np.array([ICD10_CODES[c][1] for c in icd_codes], dtype=float)
    icd10 = RNG.choice(icd_codes, size=n, p=icd_w / icd_w.sum())

    cpt_codes = list(CPT_CODES)
    cpt_w = np.array([CPT_CODES[c][3] for c in cpt_codes], dtype=float)
    cpt = RNG.choice(cpt_codes, size=n, p=cpt_w / cpt_w.sum())

    # ICD/CPT mismatch flag: CPT has a plausible-diagnosis list and this claim's
    # diagnosis isn't on it (E/M visits and labs accept any diagnosis)
    mismatch = np.array([
        c in CPT_VALID_ICD10 and d not in CPT_VALID_ICD10[c]
        for c, d in zip(cpt, icd10)
    ])

    # --- amounts ------------------------------------------------------------
    lo = np.array([CPT_CODES[c][1] for c in cpt], dtype=float)
    hi = np.array([CPT_CODES[c][2] for c in cpt], dtype=float)
    claim_amount = np.round(RNG.uniform(lo, hi), 2)

    # --- prior auth ---------------------------------------------------------
    # High-cost CPTs require auth far more often; calibrated so ~40% of all
    # claims require prior auth overall.
    is_high_cost = np.isin(cpt, list(HIGH_COST_CPT))
    pa_prob = np.where(is_high_cost, 0.90, 0.28)
    prior_auth_required = RNG.random(n) < pa_prob
    # 85% of claims that need auth actually obtained it
    prior_auth_obtained = prior_auth_required & (RNG.random(n) < 0.85)

    # --- demographics -------------------------------------------------------
    # Age skews by payer: medicare mostly 65+, medicaid skews younger
    age_bucket = np.empty(n, dtype=object)
    age_p = {
        "medicare":   [0.00, 0.01, 0.03, 0.16, 0.80],
        "medicaid":   [0.28, 0.27, 0.22, 0.17, 0.06],
        "commercial": [0.10, 0.24, 0.30, 0.30, 0.06],
        "self_pay":   [0.06, 0.38, 0.30, 0.22, 0.04],
    }
    for ptype, p in age_p.items():
        mask = payer_type == ptype
        age_bucket[mask] = RNG.choice(AGE_BUCKETS, size=mask.sum(), p=p)

    pos = RNG.choice(PLACE_OF_SERVICE, size=n, p=[0.42, 0.18, 0.16, 0.12, 0.07, 0.05])

    # --- denial probability model -------------------------------------------
    # Payer base rate + systematic, learnable effects (per-code payer policy
    # quirks, geography, demographics, cost) + the documented correlations.
    base = np.vectorize(PAYER_BASE_DENIAL.get)(payer_type)
    logit = np.log(base / (1 - base))

    # Stable per-code/per-segment effects (fixed seed -> consistent "policy")
    eff_rng = np.random.default_rng(7)
    cpt_eff = dict(zip(cpt_codes, eff_rng.normal(0, 0.55, len(cpt_codes))))
    icd_eff = dict(zip(icd_codes, eff_rng.normal(0, 0.45, len(icd_codes))))
    state_eff = dict(zip(STATES, eff_rng.normal(0, 0.35, len(STATES))))
    pos_eff = {"11": -0.20, "21": 0.30, "22": 0.10, "23": 0.45, "24": 0.05, "31": 0.25}
    age_eff = {"0-17": -0.35, "18-34": 0.25, "35-49": 0.05, "50-64": -0.05, "65+": -0.30}

    logit += np.vectorize(cpt_eff.get)(cpt)
    logit += np.vectorize(icd_eff.get)(icd10)
    logit += np.vectorize(state_eff.get)(state)
    logit += np.vectorize(pos_eff.get)(pos)
    logit += np.vectorize(age_eff.get)(age_bucket)
    # Pricier claims draw more payer scrutiny (log-scaled, standardized)
    log_amt = np.log(claim_amount)
    logit += 0.35 * (log_amt - log_amt.mean()) / log_amt.std()

    auth_missing = prior_auth_required & ~prior_auth_obtained
    logit[mismatch] += 1.6                                # coding mismatch penalty
    logit[(payer_type == "medicaid") & (provider_type == "specialist")] += 0.70
    logit[is_high_cost & ~auth_missing] += 0.20           # scrutiny on expensive procedures
    logit += RNG.normal(0, 0.10, n)                       # small idiosyncratic noise

    p_denial = 1 / (1 + np.exp(-logit))
    # Missing prior auth dominates everything: ~85% denial, full stop
    p_denial[auth_missing] = np.clip(RNG.normal(0.85, 0.04, auth_missing.sum()), 0.7, 0.97)

    # Calibrate each payer type to its target denial rate by shifting the
    # logits of non-auth-missing claims (auth-missing stays pinned at ~85%).
    for _ in range(12):
        for ptype, target in PAYER_BASE_DENIAL.items():
            seg = payer_type == ptype
            adj = seg & ~auth_missing
            cur = p_denial[seg].mean()
            if adj.sum() == 0 or not (0.001 < cur < 0.999):
                continue
            shift = (np.log(target / (1 - target)) - np.log(cur / (1 - cur))) * 0.7
            lg = np.log(p_denial[adj] / (1 - p_denial[adj])) + shift
            p_denial[adj] = 1 / (1 + np.exp(-lg))

    denied = RNG.random(n) < p_denial
    # ~5% of non-denied claims still pending adjudication
    pending = ~denied & (RNG.random(n) < 0.05)
    status = np.where(denied, "denied", np.where(pending, "pending", "approved"))

    # --- denial reasons -----------------------------------------------------
    reason_codes = list(DENIAL_REASONS)
    reason_w = np.array([DENIAL_REASONS[c][1] for c in reason_codes], dtype=float)
    denial_code = np.full(n, "", dtype=object)
    d_idx = np.where(denied)[0]
    denial_code[d_idx] = RNG.choice(reason_codes, size=len(d_idx), p=reason_w / reason_w.sum())
    # Targeted overrides: mismatches mostly CO-4/CO-11; missing auth skews CO-11
    mm_denied = denied & mismatch
    denial_code[mm_denied] = RNG.choice(["CO-4", "CO-11"], size=mm_denied.sum(), p=[0.65, 0.35])
    auth_denied = denied & auth_missing & ~mismatch
    denial_code[auth_denied] = RNG.choice(
        ["CO-11", "CO-97", "CO-4"], size=auth_denied.sum(), p=[0.55, 0.30, 0.15])
    denial_desc = np.array([DENIAL_REASONS[c][0] if c else "" for c in denial_code])

    # --- payments -----------------------------------------------------------
    approved = status == "approved"
    allowed = np.zeros(n)
    paid = np.zeros(n)
    allowed[approved] = claim_amount[approved] * RNG.uniform(0.80, 0.95, approved.sum())
    paid[approved] = allowed[approved] * RNG.uniform(0.70, 0.90, approved.sum())

    return pd.DataFrame({
        "claim_id": [str(uuid.uuid4()) for _ in range(n)],
        "patient_id": patient_id,
        "provider_id": provider_id,
        "payer_id": payer_id,
        "claim_date": claim_date,
        "service_date": service_date,
        "icd10_primary": icd10,
        "icd10_description": [ICD10_CODES[c][0] for c in icd10],
        "cpt_code": cpt,
        "cpt_description": [CPT_CODES[c][0] for c in cpt],
        "claim_amount": claim_amount,
        "allowed_amount": np.round(allowed, 2),
        "paid_amount": np.round(paid, 2),
        "claim_status": status,
        "denial_reason_code": denial_code,
        "denial_reason_description": denial_desc,
        "provider_type": provider_type,
        "payer_type": payer_type,
        "patient_age_bucket": age_bucket,
        "state": state,
        "prior_auth_required": prior_auth_required,
        "prior_auth_obtained": prior_auth_obtained,
        "denial_flag": denied.astype(int),
        "place_of_service": pos,
    })


def print_summary(claims: pd.DataFrame) -> None:
    denial_rate = claims["denial_flag"].mean()
    print("=" * 60)
    print("SYNTHETIC CLAIMS GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total claims:        {len(claims):,}")
    print(f"Overall denial rate: {denial_rate:.1%}")
    print(f"Avg claim amount:    ${claims['claim_amount'].mean():,.2f}")
    print(f"Revenue at risk:     ${claims.loc[claims['denial_flag'] == 1, 'claim_amount'].sum():,.0f}")
    print("\nDenial rate by payer type:")
    print(claims.groupby("payer_type")["denial_flag"].mean()
          .sort_values(ascending=False).map("{:.1%}".format).to_string())
    print("\nTop 5 denial reason codes:")
    top = claims.loc[claims["denial_flag"] == 1, "denial_reason_code"].value_counts().head(5)
    for code, cnt in top.items():
        print(f"  {code:<6} {cnt:>6,}  ({DENIAL_REASONS[code][0][:60]})")
    auth = claims[claims["prior_auth_required"]]
    print("\nPrior auth impact (required claims):")
    print(f"  obtained:     {auth.loc[auth['prior_auth_obtained'], 'denial_flag'].mean():.1%} denied")
    print(f"  NOT obtained: {auth.loc[~auth['prior_auth_obtained'], 'denial_flag'].mean():.1%} denied")
    print("=" * 60)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payers = build_payers()
    providers = build_providers()
    claims = build_claims(providers, payers)

    claims.to_csv(OUT_DIR / "claims_50k.csv", index=False)
    providers.to_csv(OUT_DIR / "providers.csv", index=False)
    payers.to_csv(OUT_DIR / "payers.csv", index=False)
    print(f"Wrote {OUT_DIR / 'claims_50k.csv'}")
    print(f"Wrote {OUT_DIR / 'providers.csv'} ({len(providers)} providers)")
    print(f"Wrote {OUT_DIR / 'payers.csv'} ({len(payers)} payers)\n")
    print_summary(claims)


if __name__ == "__main__":
    main()
