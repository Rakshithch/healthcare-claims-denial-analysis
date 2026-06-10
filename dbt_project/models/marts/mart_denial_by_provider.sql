-- Mart: provider-level denial performance.
-- Grain: one row per provider.

with claims as (

    select * from {{ ref('stg_claims') }}

),

providers as (

    select * from {{ ref('stg_providers') }}

)

select
    p.provider_id,
    p.provider_name,
    p.provider_type,
    p.state,
    count(*)                                              as total_claims,
    sum(c.denial_flag)                                    as denied_claims,
    round(avg(c.denial_flag), 4)                          as denial_rate,
    round(avg(c.claim_amount), 2)                         as avg_claim_amount,
    round(sum(case when c.denial_flag = 1 then c.claim_amount else 0 end), 2)
                                                          as total_revenue_at_risk
from claims c
inner join providers p
    on c.provider_id = p.provider_id
group by 1, 2, 3, 4
