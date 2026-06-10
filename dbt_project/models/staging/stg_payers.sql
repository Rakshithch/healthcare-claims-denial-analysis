-- Staging: cleaned/standardized payer dimension source.

with source as (

    select * from {{ source('raw', 'PAYERS') }}

)

select
    payer_id,
    trim(payer_name)            as payer_name,
    lower(trim(payer_type))     as payer_type
from source
where payer_id is not null
