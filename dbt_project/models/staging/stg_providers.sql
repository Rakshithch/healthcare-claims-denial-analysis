-- Staging: cleaned/standardized provider dimension source.

with source as (

    select * from {{ source('raw', 'PROVIDERS') }}

)

select
    provider_id,
    initcap(trim(provider_name))   as provider_name,
    lower(trim(provider_type))     as provider_type,
    upper(trim(state))             as state
from source
where provider_id is not null
