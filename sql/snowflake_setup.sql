-- ============================================================================
-- Healthcare Claims Denial Analysis — Snowflake setup
-- Creates database, schemas, RAW/STAGING tables, star-schema marts,
-- CSV stage + COPY INTO, and analytic views.
-- Run as a role with CREATE DATABASE privileges (e.g. SYSADMIN).
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. Database & schemas
-- ---------------------------------------------------------------------------
CREATE OR REPLACE DATABASE HEALTHCARE_CLAIMS_DENIAL;
USE DATABASE HEALTHCARE_CLAIMS_DENIAL;

CREATE OR REPLACE SCHEMA RAW;       -- landing zone, 1:1 with source CSVs
CREATE OR REPLACE SCHEMA STAGING;   -- typed/cleaned (mirrors dbt staging)
CREATE OR REPLACE SCHEMA MARTS;     -- star schema + reporting views

-- ---------------------------------------------------------------------------
-- 2. RAW tables (everything lands as-is; loose typing on purpose)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE RAW.CLAIMS (
    CLAIM_ID                    VARCHAR(36),
    PATIENT_ID                  VARCHAR(20),
    PROVIDER_ID                 VARCHAR(20),
    PAYER_ID                    VARCHAR(20),
    CLAIM_DATE                  DATE,
    SERVICE_DATE                DATE,
    ICD10_PRIMARY               VARCHAR(10),
    ICD10_DESCRIPTION           VARCHAR(200),
    CPT_CODE                    VARCHAR(10),
    CPT_DESCRIPTION             VARCHAR(200),
    CLAIM_AMOUNT                NUMBER(12,2),
    ALLOWED_AMOUNT              NUMBER(12,2),
    PAID_AMOUNT                 NUMBER(12,2),
    CLAIM_STATUS                VARCHAR(20),
    DENIAL_REASON_CODE          VARCHAR(10),
    DENIAL_REASON_DESCRIPTION   VARCHAR(300),
    PROVIDER_TYPE               VARCHAR(30),
    PAYER_TYPE                  VARCHAR(30),
    PATIENT_AGE_BUCKET          VARCHAR(10),
    STATE                       VARCHAR(2),
    PRIOR_AUTH_REQUIRED         BOOLEAN,
    PRIOR_AUTH_OBTAINED         BOOLEAN,
    DENIAL_FLAG                 NUMBER(1),
    PLACE_OF_SERVICE            VARCHAR(2)
);

CREATE OR REPLACE TABLE RAW.PROVIDERS (
    PROVIDER_ID     VARCHAR(20),
    PROVIDER_NAME   VARCHAR(100),
    PROVIDER_TYPE   VARCHAR(30),
    STATE           VARCHAR(2)
);

CREATE OR REPLACE TABLE RAW.PAYERS (
    PAYER_ID    VARCHAR(20),
    PAYER_NAME  VARCHAR(100),
    PAYER_TYPE  VARCHAR(30)
);

-- ---------------------------------------------------------------------------
-- 3. Stage + file format + COPY INTO
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FILE FORMAT RAW.CSV_FORMAT
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    SKIP_HEADER = 1
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    NULL_IF = ('', 'NULL')
    EMPTY_FIELD_AS_NULL = TRUE;

CREATE OR REPLACE STAGE RAW.CLAIMS_STAGE
    FILE_FORMAT = RAW.CSV_FORMAT
    COMMENT = 'Internal stage for synthetic claims CSV loads';

-- Upload from your machine first (SnowSQL):
--   PUT file:///<repo>/data/synthetic/claims_50k.csv @RAW.CLAIMS_STAGE;
--   PUT file:///<repo>/data/synthetic/providers.csv  @RAW.CLAIMS_STAGE;
--   PUT file:///<repo>/data/synthetic/payers.csv     @RAW.CLAIMS_STAGE;

COPY INTO RAW.CLAIMS    FROM @RAW.CLAIMS_STAGE/claims_50k.csv ON_ERROR = 'ABORT_STATEMENT';
COPY INTO RAW.PROVIDERS FROM @RAW.CLAIMS_STAGE/providers.csv  ON_ERROR = 'ABORT_STATEMENT';
COPY INTO RAW.PAYERS    FROM @RAW.CLAIMS_STAGE/payers.csv     ON_ERROR = 'ABORT_STATEMENT';

-- ---------------------------------------------------------------------------
-- 4. STAGING tables — typed, cleaned, derived fields
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE STAGING.STG_CLAIMS AS
SELECT
    CLAIM_ID,
    PATIENT_ID,
    PROVIDER_ID,
    PAYER_ID,
    CLAIM_DATE,
    SERVICE_DATE,
    UPPER(TRIM(ICD10_PRIMARY))                   AS ICD10_PRIMARY,
    ICD10_DESCRIPTION,
    TRIM(CPT_CODE)                               AS CPT_CODE,
    CPT_DESCRIPTION,
    CLAIM_AMOUNT,
    ALLOWED_AMOUNT,
    PAID_AMOUNT,
    LOWER(TRIM(CLAIM_STATUS))                    AS CLAIM_STATUS,
    NULLIF(TRIM(DENIAL_REASON_CODE), '')         AS DENIAL_REASON_CODE,
    NULLIF(TRIM(DENIAL_REASON_DESCRIPTION), '')  AS DENIAL_REASON_DESCRIPTION,
    LOWER(TRIM(PROVIDER_TYPE))                   AS PROVIDER_TYPE,
    LOWER(TRIM(PAYER_TYPE))                      AS PAYER_TYPE,
    PATIENT_AGE_BUCKET,
    UPPER(TRIM(STATE))                           AS STATE,
    PRIOR_AUTH_REQUIRED,
    PRIOR_AUTH_OBTAINED,
    DENIAL_FLAG,
    PLACE_OF_SERVICE,
    (CLAIM_STATUS = 'denied')                    AS IS_DENIED,
    YEAR(CLAIM_DATE)                             AS CLAIM_YEAR,
    MONTH(CLAIM_DATE)                            AS CLAIM_MONTH,
    QUARTER(CLAIM_DATE)                          AS CLAIM_QUARTER,
    DATEDIFF('day', SERVICE_DATE, CLAIM_DATE)    AS DAYS_TO_PROCESS
FROM RAW.CLAIMS
WHERE CLAIM_ID IS NOT NULL
  AND CLAIM_AMOUNT > 0
  AND SERVICE_DATE <= CLAIM_DATE;

CREATE OR REPLACE TABLE STAGING.STG_PROVIDERS AS
SELECT
    PROVIDER_ID,
    INITCAP(TRIM(PROVIDER_NAME))  AS PROVIDER_NAME,
    LOWER(TRIM(PROVIDER_TYPE))    AS PROVIDER_TYPE,
    UPPER(TRIM(STATE))            AS STATE
FROM RAW.PROVIDERS
WHERE PROVIDER_ID IS NOT NULL;

CREATE OR REPLACE TABLE STAGING.STG_PAYERS AS
SELECT
    PAYER_ID,
    TRIM(PAYER_NAME)            AS PAYER_NAME,
    LOWER(TRIM(PAYER_TYPE))     AS PAYER_TYPE
FROM RAW.PAYERS
WHERE PAYER_ID IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 5. MARTS — star schema
-- ---------------------------------------------------------------------------

-- Dimension: provider
CREATE OR REPLACE TABLE MARTS.DIM_PROVIDER AS
SELECT
    ROW_NUMBER() OVER (ORDER BY PROVIDER_ID) AS PROVIDER_KEY,
    PROVIDER_ID,
    PROVIDER_NAME,
    PROVIDER_TYPE,
    STATE
FROM STAGING.STG_PROVIDERS;

-- Dimension: payer
CREATE OR REPLACE TABLE MARTS.DIM_PAYER AS
SELECT
    ROW_NUMBER() OVER (ORDER BY PAYER_ID) AS PAYER_KEY,
    PAYER_ID,
    PAYER_NAME,
    PAYER_TYPE
FROM STAGING.STG_PAYERS;

-- Dimension: diagnosis (distinct ICD-10 codes seen in claims)
CREATE OR REPLACE TABLE MARTS.DIM_DIAGNOSIS AS
SELECT
    ROW_NUMBER() OVER (ORDER BY ICD10_PRIMARY) AS DIAGNOSIS_KEY,
    ICD10_PRIMARY                              AS ICD10_CODE,
    MAX(ICD10_DESCRIPTION)                     AS ICD10_DESCRIPTION,
    LEFT(ICD10_PRIMARY, 1)                     AS ICD10_CHAPTER_LETTER,
    LEFT(ICD10_PRIMARY, 3)                     AS ICD10_CATEGORY
FROM STAGING.STG_CLAIMS
GROUP BY ICD10_PRIMARY;

-- Dimension: date (covers the claim date span)
CREATE OR REPLACE TABLE MARTS.DIM_DATE AS
WITH SPAN AS (
    SELECT DATEADD('day', SEQ4(), '2022-01-01'::DATE) AS CAL_DATE
    FROM TABLE(GENERATOR(ROWCOUNT => 1200))
)
SELECT
    TO_NUMBER(TO_CHAR(CAL_DATE, 'YYYYMMDD')) AS DATE_KEY,
    CAL_DATE,
    YEAR(CAL_DATE)                            AS CAL_YEAR,
    QUARTER(CAL_DATE)                         AS CAL_QUARTER,
    MONTH(CAL_DATE)                           AS CAL_MONTH,
    MONTHNAME(CAL_DATE)                       AS CAL_MONTH_NAME,
    DAYOFWEEK(CAL_DATE)                       AS CAL_DAY_OF_WEEK,
    (DAYOFWEEK(CAL_DATE) IN (0, 6))           AS IS_WEEKEND
FROM SPAN
WHERE CAL_DATE <= '2025-03-31';

-- Fact: claims
CREATE OR REPLACE TABLE MARTS.FACT_CLAIMS AS
SELECT
    C.CLAIM_ID,
    TO_NUMBER(TO_CHAR(C.CLAIM_DATE, 'YYYYMMDD'))   AS CLAIM_DATE_KEY,
    TO_NUMBER(TO_CHAR(C.SERVICE_DATE, 'YYYYMMDD')) AS SERVICE_DATE_KEY,
    PR.PROVIDER_KEY,
    PA.PAYER_KEY,
    DG.DIAGNOSIS_KEY,
    C.PATIENT_ID,
    C.CPT_CODE,
    C.PLACE_OF_SERVICE,
    C.PATIENT_AGE_BUCKET,
    C.CLAIM_AMOUNT,
    C.ALLOWED_AMOUNT,
    C.PAID_AMOUNT,
    C.CLAIM_STATUS,
    C.DENIAL_REASON_CODE,
    C.PRIOR_AUTH_REQUIRED,
    C.PRIOR_AUTH_OBTAINED,
    C.DENIAL_FLAG,
    C.DAYS_TO_PROCESS
FROM STAGING.STG_CLAIMS C
LEFT JOIN MARTS.DIM_PROVIDER  PR ON C.PROVIDER_ID   = PR.PROVIDER_ID
LEFT JOIN MARTS.DIM_PAYER     PA ON C.PAYER_ID      = PA.PAYER_ID
LEFT JOIN MARTS.DIM_DIAGNOSIS DG ON C.ICD10_PRIMARY = DG.ICD10_CODE;

-- ---------------------------------------------------------------------------
-- 6. Analytic views
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW MARTS.V_DENIAL_RATE_BY_PAYER AS
SELECT
    P.PAYER_TYPE,
    P.PAYER_NAME,
    COUNT(*)                                   AS TOTAL_CLAIMS,
    SUM(F.DENIAL_FLAG)                         AS DENIED_CLAIMS,
    ROUND(AVG(F.DENIAL_FLAG) * 100, 2)         AS DENIAL_RATE_PCT,
    ROUND(SUM(IFF(F.DENIAL_FLAG = 1, F.CLAIM_AMOUNT, 0)), 2) AS REVENUE_AT_RISK
FROM MARTS.FACT_CLAIMS F
JOIN MARTS.DIM_PAYER P ON F.PAYER_KEY = P.PAYER_KEY
GROUP BY P.PAYER_TYPE, P.PAYER_NAME
ORDER BY DENIAL_RATE_PCT DESC;

CREATE OR REPLACE VIEW MARTS.V_DENIAL_RATE_BY_PROVIDER AS
SELECT
    PR.PROVIDER_ID,
    PR.PROVIDER_NAME,
    PR.PROVIDER_TYPE,
    PR.STATE,
    COUNT(*)                                   AS TOTAL_CLAIMS,
    SUM(F.DENIAL_FLAG)                         AS DENIED_CLAIMS,
    ROUND(AVG(F.DENIAL_FLAG) * 100, 2)         AS DENIAL_RATE_PCT,
    ROUND(AVG(F.CLAIM_AMOUNT), 2)              AS AVG_CLAIM_AMOUNT,
    ROUND(SUM(IFF(F.DENIAL_FLAG = 1, F.CLAIM_AMOUNT, 0)), 2) AS REVENUE_AT_RISK
FROM MARTS.FACT_CLAIMS F
JOIN MARTS.DIM_PROVIDER PR ON F.PROVIDER_KEY = PR.PROVIDER_KEY
GROUP BY PR.PROVIDER_ID, PR.PROVIDER_NAME, PR.PROVIDER_TYPE, PR.STATE
ORDER BY DENIAL_RATE_PCT DESC;

CREATE OR REPLACE VIEW MARTS.V_MONTHLY_DENIAL_TREND AS
SELECT
    D.CAL_YEAR,
    D.CAL_MONTH,
    D.CAL_MONTH_NAME,
    COUNT(*)                                   AS TOTAL_CLAIMS,
    SUM(F.DENIAL_FLAG)                         AS DENIED_CLAIMS,
    ROUND(AVG(F.DENIAL_FLAG) * 100, 2)         AS DENIAL_RATE_PCT,
    ROUND(AVG(AVG(F.DENIAL_FLAG)) OVER (
        ORDER BY D.CAL_YEAR, D.CAL_MONTH
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) * 100, 2)                                AS DENIAL_RATE_3MO_AVG_PCT
FROM MARTS.FACT_CLAIMS F
JOIN MARTS.DIM_DATE D ON F.CLAIM_DATE_KEY = D.DATE_KEY
GROUP BY D.CAL_YEAR, D.CAL_MONTH, D.CAL_MONTH_NAME
ORDER BY D.CAL_YEAR, D.CAL_MONTH;

CREATE OR REPLACE VIEW MARTS.V_TOP_DENIAL_REASONS AS
SELECT
    F.DENIAL_REASON_CODE,
    MAX(S.DENIAL_REASON_DESCRIPTION)           AS DENIAL_REASON_DESCRIPTION,
    COUNT(*)                                   AS DENIAL_COUNT,
    ROUND(COUNT(*) / SUM(COUNT(*)) OVER () * 100, 2) AS PCT_OF_DENIALS,
    ROUND(SUM(F.CLAIM_AMOUNT), 2)              AS REVENUE_AT_RISK
FROM MARTS.FACT_CLAIMS F
JOIN STAGING.STG_CLAIMS S ON F.CLAIM_ID = S.CLAIM_ID
WHERE F.DENIAL_FLAG = 1
GROUP BY F.DENIAL_REASON_CODE
ORDER BY DENIAL_COUNT DESC;

CREATE OR REPLACE VIEW MARTS.V_PRIOR_AUTH_IMPACT AS
SELECT
    F.PRIOR_AUTH_REQUIRED,
    F.PRIOR_AUTH_OBTAINED,
    COUNT(*)                                   AS TOTAL_CLAIMS,
    SUM(F.DENIAL_FLAG)                         AS DENIED_CLAIMS,
    ROUND(AVG(F.DENIAL_FLAG) * 100, 2)         AS DENIAL_RATE_PCT,
    ROUND(SUM(IFF(F.DENIAL_FLAG = 1, F.CLAIM_AMOUNT, 0)), 2) AS REVENUE_AT_RISK
FROM MARTS.FACT_CLAIMS F
GROUP BY F.PRIOR_AUTH_REQUIRED, F.PRIOR_AUTH_OBTAINED
ORDER BY F.PRIOR_AUTH_REQUIRED DESC, F.PRIOR_AUTH_OBTAINED;
