-- Mart: monthly denial trend with 3-month rolling average and MoM delta.
-- Grain: one row per calendar month.

with claims as (

    select * from {{ ref('stg_claims') }}

),

monthly as (

    select
        claim_year                                        as year,
        claim_month                                       as month,
        count(*)                                          as total_claims,
        sum(denial_flag)                                  as denied_claims,
        round(avg(denial_flag), 4)                        as denial_rate,
        round(sum(case when denial_flag = 1 then claim_amount else 0 end), 2)
                                                          as revenue_at_risk
    from claims
    group by 1, 2

),

with_windows as (

    select
        *,
        -- trailing 3-month average (window function)
        round(avg(denial_rate) over (
            order by year, month
            rows between 2 preceding and current row
        ), 4)                                             as denial_rate_3mo_avg,
        -- month-over-month change in denial rate
        round(denial_rate - lag(denial_rate) over (
            order by year, month
        ), 4)                                             as denial_rate_mom_change
    from monthly

)

select * from with_windows
order by year, month
