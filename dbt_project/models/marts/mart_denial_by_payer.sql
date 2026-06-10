-- Mart: payer-level denial performance with each payer's top denial reason.
-- Grain: one row per payer.

with claims as (

    select * from {{ ref('stg_claims') }}

),

payers as (

    select * from {{ ref('stg_payers') }}

),

payer_stats as (

    select
        c.payer_id,
        count(*)                                          as total_claims,
        sum(c.denial_flag)                                as denied_claims,
        round(avg(c.denial_flag), 4)                      as denial_rate,
        round(avg(c.days_to_process), 2)                  as avg_days_to_decision,
        round(sum(case when c.denial_flag = 1 then c.claim_amount else 0 end), 2)
                                                          as revenue_at_risk
    from claims c
    group by 1

),

reason_ranked as (

    -- most frequent denial reason per payer
    select
        payer_id,
        denial_reason_code as top_denial_reason,
        row_number() over (
            partition by payer_id
            order by count(*) desc, denial_reason_code
        ) as rn
    from claims
    where denial_flag = 1
    group by payer_id, denial_reason_code

)

select
    p.payer_id,
    p.payer_name,
    p.payer_type,
    s.total_claims,
    s.denied_claims,
    s.denial_rate,
    s.avg_days_to_decision,
    s.revenue_at_risk,
    r.top_denial_reason
from payers p
inner join payer_stats s on p.payer_id = s.payer_id
left join reason_ranked r on p.payer_id = r.payer_id and r.rn = 1
