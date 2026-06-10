-- Mart: denial KPIs by payer/provider segment with reason distribution.
-- Grain: one row per (payer_type, provider_type, denial_reason_code).

with claims as (

    select * from {{ ref('stg_claims') }}

),

segmented as (

    select
        payer_type,
        provider_type,
        coalesce(denial_reason_code, 'NOT_DENIED')        as denial_reason_code,
        max(denial_reason_description)                    as denial_reason_description,
        count(*)                                          as total_claims,
        sum(denial_flag)                                  as denied_claims,
        round(avg(denial_flag), 4)                        as denial_rate,
        round(avg(claim_amount), 2)                       as avg_claim_amount,
        round(avg(case when claim_status = 'approved' then claim_amount end), 2)
                                                          as avg_approved_claim_amount,
        round(avg(case when claim_status = 'denied' then claim_amount end), 2)
                                                          as avg_denied_claim_amount,
        round(sum(case when denial_flag = 1 then claim_amount else 0 end), 2)
                                                          as revenue_at_risk
    from claims
    group by 1, 2, 3

),

with_share as (

    select
        {{ dbt_utils.generate_surrogate_key(['payer_type', 'provider_type', 'denial_reason_code']) }}
                                                          as denial_analysis_key,
        *,
        -- share of this reason within the payer/provider segment's denials
        round(denied_claims / nullif(sum(denied_claims) over (
            partition by payer_type, provider_type), 0), 4) as pct_of_segment_denials
    from segmented

)

select * from with_share
