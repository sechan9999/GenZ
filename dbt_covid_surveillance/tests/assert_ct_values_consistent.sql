-- tests/assert_ct_values_consistent.sql
-- Custom test to ensure CT value calculations are consistent
-- min <= mean <= max

SELECT
    accession_number,
    ct_min,
    ct_mean,
    ct_max
FROM {{ ref('covid_tests') }}
WHERE TRUE
    AND ct_min IS NOT NULL
    AND ct_mean IS NOT NULL
    AND ct_max IS NOT NULL
    AND (
        ct_min > ct_mean
        OR ct_mean > ct_max
        OR ct_min > ct_max
    )
