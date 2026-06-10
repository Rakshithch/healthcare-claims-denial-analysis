# LinkedIn Post — ready to publish

---

💸 U.S. healthcare providers lose an estimated **$262 BILLION every year** to denied claims — and most of those denials are preventable.

I spent my nights building an end-to-end denial analytics platform to show exactly how a revenue-cycle team can find and stop those losses. 50,000 synthetic claims (modeled on real 837/835 EDI patterns and CARC denial codes), a full warehouse, and a live ML risk scorer.

🔍 **3 findings that would make any CFO sit up:**

1️⃣ **Prior authorization is the #1 controllable driver.** Claims that required auth but didn't get it denied at ~84% — vs ~27% when auth was obtained. A 3x multiplier hiding in a checkbox.

2️⃣ **Payer mix sets your baseline.** Self-pay (~40%) and Medicaid (~35%) denial rates run 2.3–2.7x the Medicare rate (~15%). Flat denial forecasting across payers is just wrong.

3️⃣ **Over half of all denials were coding mismatches** (CO-4 modifier issues + CO-11 diagnosis/procedure conflicts) — a claim-scrubber rules problem, not a clinical one. That was ~$26M/year in billed charges at risk in this dataset alone.

🛠️ **How it's built:**
• Python synthetic data generator with realistic denial economics
• Snowflake star schema (RAW → STAGING → MARTS) + analytic views
• dbt models with full schema tests
• XGBoost denial-risk classifier — AUC 0.83, SHAP explainability
• Streamlit dashboard with a live claim risk scorer: enter a claim, get a denial probability gauge, top risk drivers, and recommended fixes before submission

Coming from my work with 837/835 EDI and claims data at Change Healthcare, the most satisfying part was watching the model independently rediscover what every RCM analyst knows: check the prior auth. Then quantify it.

🔗 Full code + writeup on GitHub: https://github.com/Rakshithch/healthcare-claims-denial-analysis

What's the most preventable denial pattern you've seen in the wild? 👇

#HealthcareAnalytics #DataAnalytics #SQL #Snowflake #MachineLearning #Python #DataEngineering #Healthcare

---

**Posting tips:**
- Post Tue–Thu, 8–10am your timezone, for recruiter visibility.
- Put the GitHub link in the FIRST COMMENT (LinkedIn down-ranks posts with external links in the body).
- Attach 2–3 screenshots: the dashboard home KPIs, the risk scorer gauge, and the prior-auth impact chart.
