-- Staging: typed/cleaned claims with derived date and processing fields.
-- Source: RAW.CLAIMS (loaded from data/synthetic/claims_50k.csv)

with source as (

    select * from {{ source('raw', 'CLAIMS') }}

),

cleaned as (

    select
        claim_id,
        patient_id,
        provider_id,
        payer_id,
        cast(claim_date as date)                          as claim_date,
        cast(service_date as date)                        as service_date,
        upper(trim(icd10_primary))                        as icd10_primary,
        icd10_description,
        trim(cpt_code)                                    as cpt_code,
        cpt_description,
        cast(claim_amount as number(12,2))                as claim_amount,
        cast(allowed_amount as number(12,2))              as allowed_amount,
        cast(paid_amount as number(12,2))                 as paid_amount,
        lower(trim(claim_status))                         as claim_status,
        nullif(trim(denial_reason_code), '')              as denial_reason_code,
        nullif(trim(denial_reason_description), '')       as denial_reason_description,
        lower(trim(provider_type))                        as provider_type,
        lower(trim(payer_type))                           as payer_type,
        patient_age_bucket,
        upper(trim(state))                                as state,
        cast(prior_auth_required as boolean)              as prior_auth_required,
        cast(prior_auth_obtained as boolean)              as prior_auth_obtained,
        cast(denial_flag as number(1))                    as denial_flag,
        place_of_service,

        -- derived fields
        (lower(trim(claim_status)) = 'denied')            as is_denied,
        year(claim_date)                                  as claim_year,
        month(claim_date)                                 as claim_month,
        quarter(claim_date)                               as claim_quarter,
        datediff('day', service_date, claim_date)         as days_to_process

    from source
    -- filter out test / malformed records
    where claim_id is not null
      and claim_amount > 0
      and service_date <= claim_date

)

select * from cleaned
